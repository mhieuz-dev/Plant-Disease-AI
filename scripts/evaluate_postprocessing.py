"""
evaluate_postprocessing.py
--------------------------
So sánh F1-score của cafe_Leaf_rust và cafe_Miner
TRƯỚC và SAU color post-processing.

Cách chạy:
    python evaluate_postprocessing.py --test_dir path/to/test

Cấu trúc thư mục test (Roboflow standard):
    test/
      cafe_Leaf rust/   <- tên thư mục = tên class
        img1.jpg
        img2.jpg ...
      cafe_Miner/
        img1.jpg ...
      ... (các class khác bỏ qua, chỉ đo Rust vs Miner)

Output: in ra bảng Before / After và lưu vào postproc_results.json
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import cv2
from PIL import Image
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report

# -- Hyperparameter post-processing ------------------------------------------
CONFIDENCE_GAP_THRESHOLD = 0.25   # Chênh lệch confidence < 25% -> kích hoạt post-proc
ORANGE_HUE_MIN  = 10              # HSV Hue tối thiểu (độ) cho màu cam-vàng của Rust
ORANGE_HUE_MAX  = 45              # HSV Hue tối đa
ORANGE_SAT_MIN  = 80              # Saturation tối thiểu (0-255)
ORANGE_RATIO_THRESHOLD = 0.12     # θ: nếu ≥ 12% pixel ROI là cam-vàng -> Rust

TARGET_CLASSES = ["cafe_Leaf rust", "cafe_Miner"]


# -- Load model ---------------------------------------------------------------
def load_model(ckpt_path="best_mobilevit.pth"):
    ckpt    = torch.load(ckpt_path, map_location="cpu")
    classes = ckpt["classes"]
    model   = timm.create_model("mobilevit_s", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes

TRANSFORM = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# -- GradCAM (inline, không gọi gradcam_service để tránh dependency) ----------
def get_gradcam_bbox(model, input_tensor, target_idx, orig_size):
    last_conv = None
    for _, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m

    acts, grads = {}, {}
    fh = last_conv.register_forward_hook(lambda m,i,o: acts.update({"v": o.detach()}))
    bh = last_conv.register_full_backward_hook(lambda m,i,o: grads.update({"v": o[0].detach()}))

    model.zero_grad()
    out = model(input_tensor)
    out[0, target_idx].backward()
    fh.remove(); bh.remove()

    w   = grads["v"].mean(dim=[2,3], keepdim=True)
    cam = F.relu((w * acts["v"]).sum(dim=1, keepdim=True))
    mn, mx = cam.min(), cam.max()
    cam = (cam - mn) / (mx - mn + 1e-8)

    cam_np = cam.squeeze().cpu().numpy()
    cam_resized = cv2.resize(cam_np, orig_size, interpolation=cv2.INTER_LINEAR)
    return cam_resized                # float32 [0,1] kích thước ảnh gốc


# -- Color post-processing ----------------------------------------------------
def orange_yellow_ratio_in_bbox(img_rgb_np, cam_map):
    """Tính tỷ lệ pixel cam-vàng trong vùng GradCAM bbox (ngưỡng 50%)."""
    mask = (cam_map >= 0.5).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return 0.0

    x0, y0 = xs.min(), ys.min()
    x1, y1 = xs.max(), ys.max()
    roi_rgb = img_rgb_np[y0:y1+1, x0:x1+1]
    if roi_rgb.size == 0:
        return 0.0

    roi_bgr = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    # OpenCV HSV: H trong [0,180] (chia đôi so với độ), S/V trong [0,255]
    h_min = int(ORANGE_HUE_MIN / 2)
    h_max = int(ORANGE_HUE_MAX / 2)
    orange_mask = (
        (roi_hsv[:,:,0] >= h_min) &
        (roi_hsv[:,:,0] <= h_max) &
        (roi_hsv[:,:,1] >= ORANGE_SAT_MIN)
    )
    return float(orange_mask.mean())


def postprocess_prediction(pred_class, probs, classes, img_rgb_np,
                            model, input_tensor):
    """
    Áp post-processing khi Rust và Miner đều trong top-2 và confidence gap nhỏ.
    Trả về class sau post-processing (có thể giữ nguyên hoặc đổi).
    """
    rust_idx  = classes.index("cafe_Leaf rust")
    miner_idx = classes.index("cafe_Miner")

    p_rust  = probs[rust_idx].item()
    p_miner = probs[miner_idx].item()

    # Chỉ kích hoạt nếu cả 2 class đều trong top-2 và gap nhỏ
    top2_idxs = probs.topk(2).indices.tolist()
    if rust_idx not in top2_idxs or miner_idx not in top2_idxs:
        return pred_class              # Không phải trường hợp mơ hồ

    gap = abs(p_rust - p_miner)
    if gap >= CONFIDENCE_GAP_THRESHOLD:
        return pred_class              # Gap đủ lớn, tin tưởng model

    # Trường hợp mơ hồ -> dùng color analysis
    h, w = img_rgb_np.shape[:2]
    dominant_idx = rust_idx if p_rust >= p_miner else miner_idx
    cam = get_gradcam_bbox(model, input_tensor, dominant_idx, (w, h))
    ratio = orange_yellow_ratio_in_bbox(img_rgb_np, cam)

    if ratio >= ORANGE_RATIO_THRESHOLD:
        return "cafe_Leaf rust"
    else:
        return "cafe_Miner"


# -- Evaluation loop ----------------------------------------------------------
def evaluate(test_dir, model, classes):
    y_true_before = []
    y_pred_before = []
    y_pred_after  = []

    triggered_count = 0   # bao nhiêu ảnh đã kích hoạt post-proc

    for true_class in TARGET_CLASSES:
        class_dir = os.path.join(test_dir, true_class)
        if not os.path.isdir(class_dir):
            print(f"[WARN] Không tìm thấy thư mục: {class_dir}")
            continue

        image_files = [f for f in os.listdir(class_dir)
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        print(f"  {true_class}: {len(image_files)} ảnh")

        for fname in image_files:
            img_path = os.path.join(class_dir, fname)
            try:
                img_pil = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"    [SKIP] {fname}: {e}")
                continue

            img_rgb_np = np.array(img_pil)
            inp = TRANSFORM(img_pil).unsqueeze(0)

            with torch.no_grad():
                out   = model(inp)
                probs = F.softmax(out, dim=1)[0]

            pred_idx   = probs.argmax().item()
            pred_class = classes[pred_idx]

            # Chỉ ghi nhận nếu prediction là Rust hoặc Miner
            if pred_class not in TARGET_CLASSES:
                # Model dự đoán class khác hẳn -> bỏ qua (không liên quan)
                # Nhưng vẫn ghi là sai cho before (tùy cách tính)
                pred_class_for_eval = pred_class
            else:
                pred_class_for_eval = pred_class

            # Post-processing (cần gradient -> không dùng no_grad)
            inp_grad = TRANSFORM(img_pil).unsqueeze(0)
            inp_grad.requires_grad_(False)
            postproc_class = postprocess_prediction(
                pred_class_for_eval, probs, classes,
                img_rgb_np, model, inp_grad
            )

            if postproc_class != pred_class_for_eval:
                triggered_count += 1

            y_true_before.append(true_class)
            y_pred_before.append(pred_class_for_eval)
            y_pred_after.append(postproc_class)

    return y_true_before, y_pred_before, y_pred_after, triggered_count


# -- Main ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="test",
                        help="Đường dẫn tới thư mục test (có subfolders theo class)")
    parser.add_argument("--ckpt",     default="best_mobilevit.pth")
    args = parser.parse_args()

    print("=== Loading model...")
    model, classes = load_model(args.ckpt)
    print(f"    Classes: {classes}")

    if "cafe_Leaf rust" not in classes or "cafe_Miner" not in classes:
        print("[ERROR] Model không có class cafe_Leaf rust hoặc cafe_Miner!")
        return

    print(f"\n=== Đánh giá trên thư mục: {args.test_dir}")
    y_true, y_before, y_after, triggered = evaluate(args.test_dir, model, classes)

    if not y_true:
        print("[ERROR] Không tìm thấy ảnh nào! Kiểm tra lại --test_dir")
        return

    print(f"\n=== Kết quả ({len(y_true)} ảnh, post-proc triggered: {triggered})")

    # Tính metrics cho target classes
    labels = TARGET_CLASSES
    f1_before    = f1_score(y_true, y_before, labels=labels, average=None, zero_division=0)
    f1_after     = f1_score(y_true, y_after,  labels=labels, average=None, zero_division=0)
    rec_before   = recall_score(y_true, y_before, labels=labels, average=None, zero_division=0)
    rec_after    = recall_score(y_true, y_after,  labels=labels, average=None, zero_division=0)
    prec_before  = precision_score(y_true, y_before, labels=labels, average=None, zero_division=0)
    prec_after   = precision_score(y_true, y_after,  labels=labels, average=None, zero_division=0)

    print("\n+----------------------+------------------------------+------------------------------+")
    print("| Class                | WITHOUT post-processing      | WITH post-processing         |")
    print("|                      | Prec / Recall / F1           | Prec / Recall / F1           |")
    print("+----------------------+------------------------------+------------------------------+")
    for i, cls in enumerate(labels):
        print(f"| {cls:<20} | {prec_before[i]:.3f} / {rec_before[i]:.3f} / {f1_before[i]:.3f}        "
              f"| {prec_after[i]:.3f} / {rec_after[i]:.3f} / {f1_after[i]:.3f}        |")
    print("+----------------------+------------------------------+------------------------------+")

    # Full report
    print("\n--- Full report BEFORE ---")
    print(classification_report(y_true, y_before, labels=labels, zero_division=0))
    print("--- Full report AFTER ---")
    print(classification_report(y_true, y_after, labels=labels, zero_division=0))

    # Lưu JSON
    results = {
        "n_images": len(y_true),
        "post_proc_triggered": triggered,
        "before": {
            cls: {"precision": float(prec_before[i]),
                  "recall":    float(rec_before[i]),
                  "f1":        float(f1_before[i])}
            for i, cls in enumerate(labels)
        },
        "after": {
            cls: {"precision": float(prec_after[i]),
                  "recall":    float(rec_after[i]),
                  "f1":        float(f1_after[i])}
            for i, cls in enumerate(labels)
        }
    }
    with open("postproc_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\n[OK] Đã lưu kết quả vào postproc_results.json")
    print("     -> Dùng các số liệu này điền vào bảng tab:postproc trong paper!")


if __name__ == "__main__":
    main()
