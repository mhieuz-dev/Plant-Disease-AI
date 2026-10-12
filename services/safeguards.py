"""Safeguards for retrieval-grounded advisory generation (configuration 'rag_safe').

Three deterministic measures, each one answering a reviewer request:
 (1) disease-scoped retrieval: candidate passages are re-ranked so that documents tagged with the
     diagnosed disease come first (cross-document contamination);
 (2) grounding rules appended to the prompt: products and doses only from the passages, with a
     source number after every recommendation, and an explicit statement when the passages are silent;
 (3) post-generation check: every pesticide mentioned in the answer is looked up in the retrieved passages
     and in the Vietnamese registry (Circular 75/2025/TT-BNNMT as amended by 28/2026/TT-BNNMT);
     recommendations of banned, deregistered or withdrawn products are withheld, and products absent from
     the passages or not registered for the crop are annotated for confirmation by an extension officer.

Self-contained (standard library only) so that the same file can be dropped into the server as
services/safeguards.py with data/disease_tags.json and data/pesticide_registry_vn.json.
"""
import json, re
from pathlib import Path

CROP_OF = {"cafe": ["cà phê"], "rice": ["lúa"], "tieu": ["hồ tiêu", "tiêu"]}
PESTS = {"cafe_Leaf rust": ["gỉ sắt", "rỉ sắt"], "cafe_Miner": ["vẽ bùa"], "cafe_Phoma": ["đốm lá", "phoma", "khô cành"],
         "rice_Leaf Blast": ["đạo ôn"], "rice_Leaf scald": ["cháy bìa lá", "bạc lá", "cháy lá"], "rice_Sheath Blight": ["khô vằn"],
         "tieu_Footrot": ["chết nhanh", "thối gốc", "thối rễ", "phytophthora"], "tieu_Pollu_Disease": ["thán thư"],
         "tieu_Slow-Decline": ["chết chậm", "tuyến trùng", "thối rễ"]}
DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|g|gam|gram|kg|lít|lit|l|%)\s*(?:/|\s*(?:cho|trên|mỗi|pha)\s*)?\s*"
                  r"(ha|bình\s*\d*\s*(?:lít|lit|l)?|\d+\s*(?:lít|lit|l)(?:\s*nước)?|lít\s*nước|sào|1000\s*m2|gốc|cây|trụ)?", re.I)
MARK = "Kiểm tra tự động"
TREATMENT_QUERY = re.compile(r"thuốc|liều|hoạt chất|phun|điều trị|trị bệnh|xử lý")

GROUNDING_RULES = (
    "[Quy tắc trả lời] Chỉ nêu tên thuốc, hoạt chất, liều lượng, số lần và thời điểm phun CÓ TRONG tài liệu tham khảo "
    "ở trên; sau mỗi khuyến cáo ghi số tài liệu trong ngoặc vuông, ví dụ [2]. Không bổ sung thuốc hay liều lượng từ "
    "kiến thức ngoài tài liệu. Nếu tài liệu không đề cập điều được hỏi, nói rõ \"Tài liệu trong hệ thống chưa đề cập\" "
    "và khuyên hỏi cán bộ bảo vệ thực vật địa phương. Phần nguyên nhân, triệu chứng và biện pháp canh tác có thể dùng "
    "kiến thức chung nhưng không được mâu thuẫn với tài liệu.")


def crop_status(target_strings, class_name):
    """Registry status of an active for the crop and disease of a class: scan 'pests/ crop' pairs."""
    crop_kw = CROP_OF[class_name.split("_")[0]]; pest_kw = PESTS.get(class_name, [])
    crop_re = re.compile("|".join(r"(?<![a-zà-ỹ])" + re.escape(k) + r"(?![a-zà-ỹ])" for k in crop_kw))
    best = "registered_other_crops"
    for tg in target_strings:
        for seg in tg.lower().split(";"):
            pieces = seg.split("/")
            for k in range(len(pieces) - 1):
                if crop_re.search(pieces[k + 1]):
                    best = "registered_crop_other_pest" if best != "registered_crop_pest" else best
                    if any(p in pieces[k] for p in pest_kw): return "registered_crop_pest"
    return best


class Safeguards:
    def __init__(self, tags_path, registry_path):
        self.tags = json.loads(Path(tags_path).read_text(encoding="utf-8"))
        reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self.active_targets = reg["active_targets"]; self.pat = reg["patterns"]; self.trade = reg["trade"]
        self.banned = set(reg["banned"]); self.withdrawn = set(reg["withdrawn_trade_first_word"])
        self._pat_re = {v: re.compile(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])") for v in self.pat}
        self._trade_re = {t: re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![a-z])") for t in self.trade}

    # (1) disease-scoped retrieval -------------------------------------------------------------
    def scoped_retrieve(self, vectorstore, crop_key, class_name, display_name, user_query, top_k=4, pool=100):
        query = f"{display_name} {user_query}".strip()
        results = vectorstore.similarity_search_with_score(query, k=pool)
        tagged, others = [], []
        for doc, score in results:
            key = f"{crop_key}/{doc.metadata.get('source_file')}"
            (tagged if class_name in self.tags.get(key, []) else others).append((doc, score))
        if TREATMENT_QUERY.search(user_query.lower()):
            # for questions about products or doses, passages that name a pesticide come first among the tagged ones
            tagged = [x for x in tagged if self.find(x[0].page_content)[0]] + [x for x in tagged if not self.find(x[0].page_content)[0]]
        chosen = (tagged + others)[:top_k]
        chunks, citations, parts = [], [], []
        for i, (doc, score) in enumerate(chosen, 1):
            src = doc.metadata.get("source_file", "unknown"); page = doc.metadata.get("page", "?")
            chunks.append({"content": doc.page_content.strip(), "source_file": src, "page": page, "score": round(float(score), 4),
                           "disease_tagged": class_name in self.tags.get(f"{crop_key}/{src}", [])})
            citations.append(f"[{i}] {src}, trang {page}")
            parts.append(f"--- Tài liệu {i} ({src}, trang {page}) ---\n{doc.page_content.strip()}")
        return {"crop": crop_key, "is_healthy": False, "chunks": chunks, "context_str": "\n\n".join(parts),
                "citations": citations, "display_name": display_name, "n_tagged_in_pool": len(tagged)}

    # (3) post-generation check ----------------------------------------------------------------
    def status_of(self, canon, class_name):
        if canon in self.banned or any(canon.startswith(b) for b in self.banned): return "banned"
        tg = self.active_targets.get(canon)
        return "not_registered" if not tg else crop_status(tg, class_name)

    def find(self, text):
        low = text.lower(); found = {}
        for v, rx in self._pat_re.items():
            if rx.search(low): found.setdefault(self.pat[v], set()).add(v)
        allv = {v: c for c, vs in found.items() for v in vs}
        for c, vs in list(found.items()):
            if all(any(v != w and v in w and allv[w] != c for w in allv) for v in vs): del found[c]
        trades = [t for t, rx in self._trade_re.items() if rx.search(text)]
        return found, trades

    def post_check(self, answer, chunks, class_name):
        chunk_text = "\n".join(c["content"] for c in chunks).lower()
        flags = {"hidden": [], "unsupported": [], "other_crop": [], "dose_unsupported": []}
        out_lines = []
        for line in answer.splitlines():
            found, trades = self.find(line)
            hide, notes = [], []
            for canon, vs in found.items():
                st = self.status_of(canon, class_name)
                if st in ("banned", "not_registered"):
                    hide.append(canon); continue
                in_ctx = any(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", chunk_text) for v in vs | {canon, canon.replace("-", " ")})
                if not in_ctx:
                    flags["unsupported"].append(canon); notes.append(f"{canon}: chưa có trong tài liệu hệ thống, cần cán bộ BVTV xác nhận")
                elif st == "registered_other_crops":
                    flags["other_crop"].append(canon); notes.append(f"{canon}: chưa đăng ký cho cây này tại Việt Nam")
            for t in trades:
                if t.split()[0].lower() in self.withdrawn:
                    hide.append(t); continue
                if t.lower() not in chunk_text:
                    flags["unsupported"].append(t); notes.append(f"{t}: chưa có trong tài liệu hệ thống, cần cán bộ BVTV xác nhận")
            for m in DOSE.finditer(line):
                if not m.group(3) and m.group(2) != "%": continue
                if m.group(2) == "%" and float(m.group(1).replace(",", ".")) > 10: continue   # 60% is an efficacy, not a dose
                num = m.group(1); alt = num.replace(",", ".") if "," in num else num.replace(".", ",")
                unit = m.group(2).lower(); units = {unit, unit.replace("lit", "lít")} | ({"g", "gam", "gram"} if unit in ("g", "gam", "gram") else set())
                if not any(re.search(re.escape(n) + r"\s*" + re.escape(u) + r"(?![a-z])", chunk_text) for n in (num, alt) for u in units):
                    flags["dose_unsupported"].append(m.group(0).strip()); notes.append(f"liều \"{m.group(0).strip()}\": chưa có trong tài liệu hệ thống")
            if hide:
                flags["hidden"] += hide
                out_lines.append(f"⚠️ {MARK}: đã ẩn một khuyến cáo vì {', '.join(hide)} không còn trong Danh mục thuốc bảo vệ "
                                 f"thực vật được phép sử dụng tại Việt Nam (Thông tư 75/2025/TT-BNNMT, sửa đổi 28/2026/TT-BNNMT).")
            elif notes:
                out_lines.append(line + " [" + MARK + ": " + "; ".join(dict.fromkeys(notes)) + "]")
            else:
                out_lines.append(line)
        return "\n".join(out_lines), flags

    def summary_notes(self, flags):
        """For the streaming endpoint, where the text is already sent: one block appended at the end."""
        parts = []
        if flags["hidden"]:
            parts.append("KHÔNG dùng " + ", ".join(dict.fromkeys(flags["hidden"])) + ": không còn trong Danh mục thuốc bảo vệ thực vật được phép sử dụng tại Việt Nam (Thông tư 75/2025/TT-BNNMT, sửa đổi 28/2026/TT-BNNMT)")
        if flags["unsupported"]:
            parts.append(", ".join(dict.fromkeys(flags["unsupported"])) + ": chưa có trong tài liệu hệ thống, cần cán bộ BVTV xác nhận trước khi dùng")
        if flags["other_crop"]:
            parts.append(", ".join(dict.fromkeys(flags["other_crop"])) + ": chưa đăng ký cho cây này tại Việt Nam")
        if flags["dose_unsupported"]:
            parts.append("liều lượng " + "; ".join(dict.fromkeys(flags["dose_unsupported"])) + ": chưa có trong tài liệu hệ thống")
        return ("\n\n⚠️ " + MARK + ": " + ". ".join(parts) + ".") if parts else ""
