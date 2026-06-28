"""
gradcam_service.py
──────────────────
Tạo GradCAM heatmap để nhận diện và tô màu vùng bệnh trên ảnh cây trồng.
Hỗ trợ cả GradCAM thuần (register_hook) lẫn CBAM (nếu model có).

Cách dùng:
    result = generate_gradcam(image_bytes, model, classes, predicted_idx)
    # result["heatmap_b64"] → ảnh heatmap base64 để trả về Flutter
    # result["bbox"]        → bounding box vùng bệnh nổi bật nhất
    # result["coverage"]    → % diện tích bị ảnh hưởng
"""

import io
import base64
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFilter


# ── TRANSFORM khớp với predict1.py ────────────────────────────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def _get_last_conv_layer(model):
    """
    Tự động tìm lớp Conv2d cuối cùng trong model.
    MobileViT có lớp conv ở cuối encoder — đây là nơi tốt nhất để hook.
    """
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = (name, module)
    if last_conv is None:
        raise ValueError("Không tìm thấy lớp Conv2d nào trong model")
    return last_conv


def generate_gradcam(
    image_bytes: bytes,
    model: torch.nn.Module,
    classes: list[str],
    target_idx: int | None = None,
    alpha: float = 0.5,
) -> dict:
    """
    Sinh GradCAM heatmap cho ảnh đầu vào.

    Parameters
    ----------
    image_bytes : bytes
        Dữ liệu ảnh thô (JPEG/PNG).
    model : nn.Module
        Model đã load (MobileViT hoặc bất kỳ CNN nào).
    classes : list[str]
        Danh sách tên lớp (trùng với ckpt["classes"]).
    target_idx : int | None
        Index lớp muốn visualize. None → dùng lớp có score cao nhất.
    alpha : float
        Độ trong suốt khi overlay heatmap lên ảnh gốc (0–1).

    Returns
    -------
    dict với các key:
        heatmap_b64  : str   – base64 PNG của ảnh overlay
        raw_map_b64  : str   – base64 PNG của heatmap thuần (không overlay)
        bbox         : dict  – {"x","y","w","h"} bounding box vùng nổi bật
        coverage_pct : float – % diện tích ảnh bị đánh dấu (ngưỡng 50%)
        predicted_class : str
        confidence   : float
    """
    # ── 1. Chuẩn bị ảnh ───────────────────────────────────────────────────────
    original_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original_img.size

    input_tensor = TRANSFORM(original_img).unsqueeze(0)  # [1, 3, 256, 256]

    # ── 2. Hook để bắt feature map và gradient ────────────────────────────────
    _, last_conv = _get_last_conv_layer(model)

    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    fh = last_conv.register_forward_hook(forward_hook)
    bh = last_conv.register_full_backward_hook(backward_hook)

    # ── 3. Forward pass ───────────────────────────────────────────────────────
    model.eval()
    output = model(input_tensor)  # [1, num_classes]
    probs = F.softmax(output, dim=1)

    confidence, pred_idx = torch.max(probs, dim=1)
    confidence_val = confidence.item()
    pred_class = classes[pred_idx.item()]

    if target_idx is None:
        target_idx = pred_idx.item()

    # ── 4. Backward pass (tính gradient của score → feature map) ──────────────
    model.zero_grad()
    score = output[0, target_idx]
    score.backward()

    fh.remove()
    bh.remove()

    # ── 5. Tính GradCAM weights ───────────────────────────────────────────────
    # GAP của gradient theo không gian → weight cho mỗi kênh
    grads = gradients["value"]   # [1, C, H, W]
    acts  = activations["value"] # [1, C, H, W]

    weights = grads.mean(dim=[2, 3], keepdim=True)  # [1, C, 1, 1]
    cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, H, W]
    cam = F.relu(cam)  # loại bỏ giá trị âm

    # Normalize về [0, 1]
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    # ── 6. Resize CAM về kích thước ảnh gốc ──────────────────────────────────
    cam_np = cam.squeeze().cpu().numpy()  # [H_feat, W_feat]
    cam_pil = Image.fromarray((cam_np * 255).astype(np.uint8))
    cam_pil = cam_pil.resize((orig_w, orig_h), Image.BILINEAR)
    cam_pil = cam_pil.filter(ImageFilter.GaussianBlur(radius=3))
    cam_array = np.array(cam_pil) / 255.0  # [0, 1] float

    # ── 7. Tạo heatmap màu (COLORMAP_JET kiểu thủ công) ──────────────────────
    heatmap_colored = _apply_jet_colormap(cam_array)  # [H, W, 3] uint8

    # ── 8. Overlay lên ảnh gốc ────────────────────────────────────────────────
    orig_np = np.array(original_img)
    overlay = (
        (1 - alpha) * orig_np.astype(np.float32) +
        alpha * heatmap_colored.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)

    # ── 9. Tính bounding box vùng nổi bật (ngưỡng 50%) ───────────────────────
    threshold = 0.5
    mask = (cam_array >= threshold).astype(np.uint8)
    bbox = _compute_bbox(mask)
    coverage_pct = float(mask.mean() * 100)

    # Vẽ bbox lên overlay
    if bbox:
        overlay_with_box = overlay_img.copy()
        draw = ImageDraw.Draw(overlay_with_box)
        x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 255, 0), width=3)
        # Label confidence
        draw.text((x + 4, y + 4),
                  f"{pred_class}\n{confidence_val:.1%}",
                  fill=(255, 255, 0))
    else:
        overlay_with_box = overlay_img

    # ── 10. Encode base64 ─────────────────────────────────────────────────────
    heatmap_b64 = _img_to_base64(overlay_with_box)
    raw_map_b64 = _img_to_base64(Image.fromarray(heatmap_colored))

    return {
        "heatmap_b64":     heatmap_b64,
        "raw_map_b64":     raw_map_b64,
        "bbox":            bbox,
        "coverage_pct":    round(coverage_pct, 2),
        "predicted_class": pred_class,
        "confidence":      round(confidence_val, 4),
    }


# ── HELPERS ────────────────────────────────────────────────────────────────────

def _apply_jet_colormap(cam: np.ndarray) -> np.ndarray:
    """
    Áp JET colormap thủ công (xanh dương → xanh lá → vàng → đỏ).
    cam: float array [H, W] trong [0, 1]
    Returns: uint8 array [H, W, 3]
    """
    r = np.clip(1.5 - np.abs(4.0 * cam - 3.0), 0, 1)
    g = np.clip(1.5 - np.abs(4.0 * cam - 2.0), 0, 1)
    b = np.clip(1.5 - np.abs(4.0 * cam - 1.0), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255).astype(np.uint8)


def _compute_bbox(mask: np.ndarray) -> dict | None:
    """
    Tính bounding box nhỏ nhất bao quanh vùng mask == 1.
    Returns dict {"x","y","w","h"} hoặc None nếu mask rỗng.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return {
        "x": int(cmin),
        "y": int(rmin),
        "w": int(cmax - cmin),
        "h": int(rmax - rmin),
    }


def _img_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")