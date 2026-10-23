"""
rag_service.py  (v2 - fix đúng class names)
--------------------------------------------
Class names thực tế từ model:
  cafe_Healthy | cafe_Leaf rust | cafe_Miner | cafe_Phoma
  rice_Healthy Rice Leaf | rice_Leaf Blast | rice_Leaf scald | rice_Sheath Blight
  tieu_Healthy | tieu_Footrot | tieu_Pollu_Disease | tieu_Slow-Decline

Logic nhận diện crop: chỉ cần xem PREFIX trước dấu "_" đầu tiên
  cafe_*  -> ca_phe/
  rice_*  -> lua/
  tieu_*  -> tieu/

Tên file PDF trong mỗi folder: KHÔNG CẦN khớp tên class.
RAG tìm theo NỘI DUNG bên trong file, không theo tên file.
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document 
from dotenv import load_dotenv

load_dotenv()

# -- CẤU HÌNH ------------------------------------------------------------------

DOCS_ROOT  = Path("docs")
INDEX_ROOT = Path("faiss_indexes")

# Prefix của class name -> folder tài liệu
# Ví dụ: "cafe_Leaf rust" -> prefix "cafe" -> folder "ca_phe"
PREFIX_TO_CROP: dict[str, str] = {
    "cafe": "ca_phe",
    "rice": "lua",
    "tieu": "tieu",
}

# Tên hiển thị đẹp cho từng crop (dùng trong prompt)
CROP_DISPLAY: dict[str, str] = {
    "ca_phe": "cà phê",
    "lua":    "lúa",
    "tieu":   "hồ tiêu",
}

# Tên bệnh đẹp để hiển thị trong câu trả lời
CLASS_DISPLAY: dict[str, str] = {
    # Cà phê
    "cafe_Healthy":   "Cà phê khoẻ mạnh",
    "cafe_Leaf rust": "Bệnh gỉ sắt cà phê",
    "cafe_Miner":     "Sâu vẽ bùa cà phê",
    "cafe_Phoma":     "Bệnh đốm lá Phoma cà phê",
    # Lúa
    "rice_Healthy Rice Leaf": "Lúa khoẻ mạnh",
    "rice_Leaf Blast":        "Bệnh đạo ôn lá lúa",
    "rice_Leaf scald":        "Bệnh cháy bìa lá lúa",
    "rice_Sheath Blight":     "Bệnh khô vằn lúa",
    # Hồ tiêu
    "tieu_Healthy":       "Hồ tiêu khoẻ mạnh",
    "tieu_Footrot":       "Bệnh chết nhanh / thối cổ rễ hồ tiêu",
    "tieu_Pollu_Disease": "Bệnh thán thư hồ tiêu",
    "tieu_Slow-Decline":  "Bệnh chết chậm hồ tiêu",
}

# Class nào là "khoẻ mạnh" -> không cần RAG tìm tài liệu bệnh
HEALTHY_CLASSES = {
    "cafe_Healthy",
    "rice_Healthy Rice Leaf",
    "tieu_Healthy",
}

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-3-small")

# Cache vector stores đã load
_vector_stores: dict[str, FAISS] = {}


# -- DETECT CROP TỪ CLASS NAME --------------------------------------------------

def detect_crop_from_class(class_name: str) -> Optional[str]:
    """
    Lấy prefix trước '_' đầu tiên, map sang crop key.
    "cafe_Leaf rust" -> "cafe" -> "ca_phe"
    "rice_Sheath Blight" -> "rice" -> "lua"
    "tieu_Footrot" -> "tieu" -> "tieu"
    Trả về None nếu không nhận ra.
    """
    prefix = class_name.split("_")[0].lower().strip()
    return PREFIX_TO_CROP.get(prefix)


def is_healthy(class_name: str) -> bool:
    return class_name in HEALTHY_CLASSES


def get_display_name(class_name: str) -> str:
    """Trả về tên tiếng Việt đẹp của class, fallback về class_name gốc."""
    return CLASS_DISPLAY.get(class_name, class_name)


# -- BUILD INDEX ----------------------------------------------------------------

def build_index_for_crop(crop_key: str) -> None:
    """
    Đọc toàn bộ PDF/TXT trong docs/<crop_key>/, chunk và tạo FAISS index.
    Tên file không quan trọng - chỉ cần đúng folder và nội dung đúng cây.
    """
    folder     = DOCS_ROOT / crop_key
    index_path = INDEX_ROOT / crop_key

    if not folder.exists():
        print(f"[RAG] Thư mục {folder} chưa có - bỏ qua. Tạo folder và thêm PDF vào nhé.")
        return

    # Load index sẵn có nếu đã build rồi
    if index_path.exists() and any(index_path.iterdir()):
        print(f"[RAG] Load index sẵn có: {index_path}")
        _vector_stores[crop_key] = FAISS.load_local(
            str(index_path), EMBEDDINGS, allow_dangerous_deserialization=True
        )
        return

    print(f"[RAG] Building index cho '{crop_key}' từ {folder} ...")

    docs: list[Document] = []
    for file_path in sorted(folder.glob("**/*")):
        if not file_path.is_file():
            continue
        try:
            if file_path.suffix.lower() == ".pdf":
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() in (".txt", ".md"):
                loader = TextLoader(str(file_path), encoding="utf-8")
            else:
                continue
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_file"] = file_path.name
                doc.metadata["crop"]        = crop_key
            docs.extend(loaded)
            print(f"  [OK] {file_path.name} ({len(loaded)} trang)")
        except Exception as e:
            print(f"  [ERROR] {file_path.name}: {e}")

    if not docs:
        print(f"[RAG] Không có tài liệu nào trong {folder} - bỏ qua.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(docs)
    print(f"[RAG] {crop_key}: {len(docs)} trang -> {len(chunks)} chunks")

    vectorstore = FAISS.from_documents(chunks, EMBEDDINGS)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    _vector_stores[crop_key] = vectorstore
    print(f"[RAG] [OK] Lưu index: {index_path}")


def build_all_indexes() -> None:
    """Gọi khi server start - build/load tất cả FAISS indexes."""
    INDEX_ROOT.mkdir(exist_ok=True)
    for crop_key in PREFIX_TO_CROP.values():
        build_index_for_crop(crop_key)


# -- RETRIEVE CONTEXT -----------------------------------------------------------

def retrieve_context(
    class_name: str,
    user_query: str = "",
    top_k: int = 3,
) -> dict:
    """
    Tìm chunks tài liệu liên quan dựa trên class_name từ model.

    Returns dict:
        crop        : str | None
        is_healthy  : bool
        chunks      : list[dict]
        context_str : str   - nhét thẳng vào prompt
        citations   : list[str]
        display_name: str   - tên bệnh tiếng Việt đẹp
    """
    display_name = get_display_name(class_name)
    healthy      = is_healthy(class_name)
    crop_key     = detect_crop_from_class(class_name)

    base = {
        "crop":         crop_key,
        "is_healthy":   healthy,
        "chunks":       [],
        "context_str":  "",
        "citations":    [],
        "display_name": display_name,
    }

    # Cây khoẻ -> không cần tìm tài liệu bệnh
    if healthy:
        return base

    # Không nhận ra crop prefix
    if crop_key is None:
        print(f"[RAG] Không nhận ra crop từ class '{class_name}'")
        return base

    # Chưa có index (chưa thêm tài liệu)
    if crop_key not in _vector_stores:
        print(f"[RAG] Index '{crop_key}' chưa có - chưa thêm PDF vào docs/{crop_key}/")
        return base

    vectorstore  = _vector_stores[crop_key]
    search_query = f"{display_name} {user_query}".strip()

    results = vectorstore.similarity_search_with_score(search_query, k=top_k)
    
    print("\n========== DEBUG RAG ==========")
    print("CLASS:", class_name)
    print("QUERY:", search_query)
    print("RESULT COUNT:", len(results))
    print("===============================\n")

    chunks, citations, context_parts = [], [], []
    for i, (doc, score) in enumerate(results, 1):
        source  = doc.metadata.get("source_file", "unknown")
        page    = doc.metadata.get("page", "?")
        content = doc.page_content.strip()

        chunks.append({
            "content":     content,
            "source_file": source,
            "page":        page,
            "score":       round(float(score), 4),
        })
        citations.append(f"[{i}] {source}, trang {page}")
        context_parts.append(
            f"--- Tài liệu {i} ({source}, trang {page}) ---\n{content}"
        )

    return {
        **base,
        "chunks":      chunks,
        "context_str": "\n\n".join(context_parts),
        "citations":   citations,
    }


# -- BUILD PROMPT ---------------------------------------------------------------

def build_rag_prompt(
    class_name: str,
    user_question: str,
    rag_result: dict,
    confidence: float = 1.0,
) -> str:
    """
    Tạo prompt gửi GPT, có nhúng tài liệu RAG.
    Nếu cây khoẻ hoặc chưa có tài liệu -> prompt tự chuyên môn, không bịa nguồn.
    """
    display_name = rag_result["display_name"]
    crop_display = CROP_DISPLAY.get(rag_result.get("crop", ""), "cây trồng")
    is_healthy_  = rag_result["is_healthy"]
    context_str  = rag_result.get("context_str", "")
    citations    = rag_result.get("citations", [])

    # -- Cây khoẻ mạnh --------------------------------------------------------
    if is_healthy_:
        return f"""Cây {crop_display} được AI chẩn đoán là **khoẻ mạnh** (độ tin cậy: {confidence:.1%}).

Hãy trả lời ngắn gọn:
✅ Thông báo cây khoẻ mạnh
🌱 Vài lời khuyên chăm sóc phòng ngừa bệnh cho {crop_display}
⏰ Thời điểm nên kiểm tra lại

Câu hỏi thêm của người dùng: {user_question if user_question else "(không có)"}
"""

    # -- Có tài liệu RAG -------------------------------------------------------
    if context_str:
        citations_text = "\n".join(citations)
        return f"""Cây {crop_display} được AI chẩn đoán mắc bệnh: **{display_name}**
Độ tin cậy mô hình: {confidence:.1%}

📚 TÀI LIỆU THAM KHẢO (trích từ cơ sở dữ liệu {crop_display}):
{context_str}

Nguồn:
{citations_text}

Dựa trên tài liệu trên, hãy trả lời theo cấu trúc:

🔬 **Bệnh:** {display_name}
🌱 **Nguyên nhân:** (ưu tiên dùng thông tin từ tài liệu)
⚠️ **Dấu hiệu nhận biết:** (triệu chứng cụ thể)
💊 **Cách điều trị:** (tên thuốc, liều lượng, thời gian phun — lấy từ tài liệu)
🛡️ **Phòng tránh:** (biện pháp ngăn ngừa)
⏰ **Mức độ khẩn cấp:** Thấp / Trung bình / Cao
📖 **Nguồn tham khảo:** (ghi rõ "Theo tài liệu [số], trang X...")

Câu hỏi thêm: {user_question if user_question else "(không có)"}
"""

    # -- CHƯA CÓ tài liệu - trả lời bằng kiến thức, nhưng ghi rõ không có tài liệu --
    return f"""Cây {crop_display} được AI chẩn đoán mắc bệnh: **{display_name}**
Độ tin cậy mô hình: {confidence:.1%}

⚠️ Lưu ý: Hệ thống chưa có tài liệu chuyên khảo về bệnh này trong cơ sở dữ liệu.
Câu trả lời dưới đây dựa trên kiến thức nông nghiệp tổng quát — nên tham khảo thêm chuyên gia địa phương.

Hãy trả lời theo cấu trúc:

🔬 **Bệnh:** {display_name}
🌱 **Nguyên nhân:** (kiến thức chung)
⚠️ **Dấu hiệu nhận biết:**
💊 **Cách điều trị:** (hướng dẫn chung, ghi rõ nên xác nhận với cán bộ khuyến nông)
🛡️ **Phòng tránh:**
⏰ **Mức độ khẩn cấp:** Thấp / Trung bình / Cao
📖 **Nguồn:** Kiến thức nông nghiệp tổng quát (chưa có tài liệu chính thức trong hệ thống)

Câu hỏi thêm: {user_question if user_question else "(không có)"}
"""