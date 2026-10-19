"""Safeguards for retrieval-grounded advisory generation (configuration 'rag_safe'), version 2.

Three deterministic measures, each one answering a reviewer request:
 (1) disease-scoped retrieval: the 100 nearest chunks are re-ranked so that documents tagged with the
     diagnosed disease come first and, for questions about products or doses, chunks that name a
     pesticide come first among them; lines that name a banned or deregistered product are removed from
     the passages before generation; citations are shown with human-readable document titles;
 (2) grounding rules appended to the prompt: products and doses only from the passages, a source number
     after every recommendation, doses written in Vietnamese with the units of the passage, and an
     explicit statement when the passages are silent;
 (3) post-generation check: every pesticide mentioned in the answer is looked up in the retrieved passages
     and in the Vietnamese registry (Circular 75/2025/TT-BNNMT as amended by 28/2026/TT-BNNMT).
     A recommendation of a banned, deregistered or withdrawn product is redacted (the product and its
     dose are removed from the sentence, the rest is kept); products absent from the passages or not
     registered for the crop are tagged in place; one closing block explains the tags. Fertilizers and
     biostimulants matched by the lexicon are ignored; biological control agents get a soft note only.

Self-contained (standard library only) so that the same file can be dropped into the server as
services/safeguards.py with data/disease_tags.json, data/pesticide_registry_vn.json and data/docs_meta.json.
"""
import json, re
from pathlib import Path

CROP_OF = {"cafe": ["cà phê"], "rice": ["lúa"], "tieu": ["hồ tiêu", "tiêu"]}
PESTS = {"cafe_Leaf rust": ["gỉ sắt", "rỉ sắt"], "cafe_Miner": ["vẽ bùa"], "cafe_Phoma": ["đốm lá", "phoma", "khô cành"],
         "rice_Leaf Blast": ["đạo ôn"], "rice_Leaf scald": ["leaf scald", "bỏng lá"], "rice_Sheath Blight": ["khô vằn"],
         "tieu_Footrot": ["chết nhanh", "thối gốc", "thối rễ", "phytophthora"], "tieu_Pollu_Disease": ["thán thư"],
         "tieu_Slow-Decline": ["chết chậm", "tuyến trùng", "thối rễ"]}
DOSE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|g|gam|gram|kg|lít|lit|l|%)\s*(?:/|\s*(?:cho|trên|mỗi|pha)\s*)?\s*"
                  r"(ha|bình\s*\d*\s*(?:lít|lit|l)?|\d+\s*(?:lít|lit|l)(?:\s*nước)?|lít\s*nước|sào|1000\s*m2|gốc|cây|trụ)?", re.I)
TREATMENT_QUERY = re.compile(r"thuốc|liều|hoạt chất|phun|điều trị|trị bệnh|xử lý")
MARK = "Kiểm tra tự động"
REDACT = "đã lược bỏ"
# substances in the registry lexicon that are fertilizers or biostimulants, not pesticides: never checked
NON_PESTICIDE = {"amino acid", "fulvic acid", "humic acid", "seaweed extract", "chitosan", "vitamin b1", "salicylic acid",
                 "oligosaccharins", "gibberellic acid", "cytokinin", "brassinolide"}
# biological control agents: registration is product-specific, so only a soft note is given
BIOCONTROL = ("bacillus", "beauveria", "metarhizium", "trichoderma", "paecilomyces", "purpureocillium", "pseudomonas",
              "streptomyces", "verticillium", "chaetomium", "isaria", "azadirachtin")
DELIM = re.compile(r"(?<!\d),|;|\+|\(|\)|(?<!\d)\.(?!\d)|\s(?:hoặc|và|kết hợp|với|hay)\s|\s-\s|$")

GROUNDING_RULES = (
    "[Quy tắc trả lời] Chỉ nêu tên thuốc, hoạt chất, liều lượng, số lần và thời điểm phun CÓ TRONG tài liệu tham khảo "
    "ở trên; sau mỗi khuyến cáo ghi số tài liệu trong ngoặc vuông, ví dụ [2]; không liệt kê thêm các số tài liệu rời ở "
    "cuối câu trả lời. Không bổ sung thuốc hay liều lượng từ kiến thức ngoài tài liệu và không tự quy đổi liều. Viết liều "
    "bằng tiếng Việt với đúng con số và đơn vị trong tài liệu, ví dụ \"pha 1,5 ml cho mỗi lít nước\", không dùng ký hiệu @. "
    "Nếu tài liệu không đề cập điều được hỏi, nói rõ \"Tài liệu trong hệ thống chưa đề cập\" và khuyên hỏi cán bộ bảo vệ "
    "thực vật địa phương. Phần nguyên nhân, triệu chứng và biện pháp canh tác có thể dùng kiến thức chung nhưng không "
    "được mâu thuẫn với tài liệu.")


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
    def __init__(self, tags_path, registry_path, meta_path=None):
        self.tags = json.loads(Path(tags_path).read_text(encoding="utf-8"))
        reg = json.loads(Path(registry_path).read_text(encoding="utf-8"))
        self.active_targets = reg["active_targets"]; self.pat = reg["patterns"]; self.trade = reg["trade"]
        self.banned = set(reg["banned"]); self.withdrawn = set(reg["withdrawn_trade_first_word"])
        self.meta = json.loads(Path(meta_path).read_text(encoding="utf-8")) if meta_path and Path(meta_path).exists() else {}
        self._pat_re = {v: re.compile(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])") for v in self.pat}
        self._trade_re = {t: re.compile(r"(?<![A-Za-z0-9])" + re.escape(t) + r"(?![a-z])") for t in self.trade}

    # (1) disease-scoped retrieval -------------------------------------------------------------
    def cite(self, crop_key, source_file, page):
        title = self.meta.get(f"{crop_key}/{source_file}", {}).get("title", source_file)
        return title + (f", trang {page + 1}" if isinstance(page, int) else "")

    def scoped_retrieve(self, vectorstore, crop_key, class_name, display_name, user_query, top_k=6, pool=100):
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
            content, removed = self.sanitize(doc.page_content.strip(), class_name)
            chunks.append({"content": content, "source_file": src, "page": page, "score": round(float(score), 4),
                           "disease_tagged": class_name in self.tags.get(f"{crop_key}/{src}", []), "removed_lines": removed})
            citations.append(f"[{i}] {self.cite(crop_key, src, page)}")
            parts.append(f"--- Tài liệu {i} ({self.cite(crop_key, src, page)}) ---\n{content}")
        return {"crop": crop_key, "is_healthy": False, "chunks": chunks, "context_str": "\n\n".join(parts),
                "citations": citations, "display_name": display_name, "n_tagged_in_pool": len(tagged)}

    def sanitize(self, text, class_name):
        """Drop the lines of a passage that name a banned or deregistered product, before generation."""
        kept, removed = [], []
        for line in text.splitlines():
            found, trades = self.find(line)
            bad = [c for c in found if c not in NON_PESTICIDE and self.status_of(c, class_name) in ("banned", "not_registered")]
            bad += [t for t in trades if t.split()[0].lower() in self.withdrawn]
            (removed if bad else kept).append(line)
        return "\n".join(kept), removed

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

    def _redact(self, line, patterns, label):
        """Remove the product and the dose that follows it (same clause, or in parentheses right after it)."""
        rx = re.compile("|".join(re.escape(p) for p in sorted(patterns, key=len, reverse=True)), re.I)
        marker = f"[{REDACT}: {label}]"
        out, pos, guard = line, 0, 0
        while guard < 5:
            m = rx.search(out, pos)
            if not m: break
            end = DELIM.search(out, m.end()); e = end.start() if end else len(out)
            pm = re.match(r"\s*\(([^()]*)\)", out[e:])
            if pm and DOSE.search(pm.group(1)) and not self.find(pm.group(1))[0]:
                e += pm.end()   # a parenthesized dose right after the product belongs to it
            out = out[:m.start()] + marker + out[e:]
            pos = m.start() + len(marker); guard += 1
        return out

    def post_check(self, answer, chunks, class_name):
        chunk_text = "\n".join(c["content"] for c in chunks).lower()
        flags = {"hidden": [], "unsupported": [], "other_crop": [], "dose_unsupported": [], "biocontrol": []}
        out_lines = []
        for line in answer.splitlines():
            if re.search(r"\d", line) and re.fullmatch(r"\s*\[?[\d,\s\[\]]+\]?\s*", line):
                continue  # orphan citation-only line such as "[4], [3]"
            found, trades = self.find(line)
            tags, hidden_here = [], []
            for canon, vs in found.items():
                if canon in NON_PESTICIDE: continue
                st = self.status_of(canon, class_name)
                if st in ("banned", "not_registered"):
                    line = self._redact(line, vs | {canon, canon.replace("-", " ")}, canon); flags["hidden"].append(canon); hidden_here.append(canon); continue
                in_ctx = any(re.search(r"(?<![a-z0-9])" + re.escape(v) + r"(?![a-z0-9])", chunk_text) for v in vs | {canon, canon.replace("-", " ")})
                if not in_ctx:
                    flags["unsupported"].append(canon); tags.append(f"chưa có trong tài liệu: {canon}")
                elif canon.startswith(BIOCONTROL):
                    if st == "registered_other_crops": flags["biocontrol"].append(canon)
                elif st == "registered_other_crops":
                    flags["other_crop"].append(canon); tags.append(f"{canon} chưa đăng ký cho cây này")
            for t in trades:
                if t.split()[0].lower() in self.withdrawn:
                    line = self._redact(line, {t}, t); flags["hidden"].append(t); hidden_here.append(t); continue
                if t.lower() not in chunk_text and not any(t.lower() in x for x in tags):
                    flags["unsupported"].append(t); tags.append(f"chưa có trong tài liệu: {t}")
            for m in DOSE.finditer(line):
                before = line[:m.start()]
                if before.rfind(REDACT) > before.rfind("]"): continue   # dose inside a redaction marker
                if not m.group(3) and m.group(2) != "%": continue
                if m.group(2) == "%" and float(m.group(1).replace(",", ".")) > 10: continue
                num = m.group(1); alt = num.replace(",", ".") if "," in num else num.replace(".", ",")
                unit = m.group(2).lower(); units = {unit, unit.replace("lit", "lít")} | ({"g", "gam", "gram"} if unit in ("g", "gam", "gram") else set())
                if not any(re.search(re.escape(n) + r"\s*" + re.escape(u) + r"(?![a-z])", chunk_text) for n in (num, alt) for u in units):
                    flags["dose_unsupported"].append(m.group(0).strip()); tags.append(f"liều chưa có trong tài liệu: {m.group(0).strip()}")
            if hidden_here:
                body = re.sub(r"\[" + REDACT + r"[^\]]*\]", "", line)
                if len(re.findall(r"[a-zà-ỹ]", body.lower())) < 12:
                    line = f"⚠️ [{REDACT}: {', '.join(dict.fromkeys(hidden_here))}, không còn được phép sử dụng tại Việt Nam]"
            if tags:
                line = line + " [" + "; ".join(dict.fromkeys(tags)) + "]"
            out_lines.append(line)
        notes = self.summary_notes(flags)
        return "\n".join(out_lines).rstrip() + notes, flags

    def summary_notes(self, flags):
        """One closing block that explains redactions and tags (also used by the streaming endpoint)."""
        parts = []
        if flags["hidden"]:
            parts.append("Đã lược bỏ " + ", ".join(dict.fromkeys(flags["hidden"])) + " vì không còn trong Danh mục thuốc bảo vệ thực vật "
                         "được phép sử dụng tại Việt Nam (Thông tư 75/2025/TT-BNNMT, sửa đổi 28/2026/TT-BNNMT); hỏi cán bộ bảo vệ thực vật về thuốc thay thế.")
        if flags["unsupported"] or flags["dose_unsupported"]:
            parts.append("Mục đánh dấu \"chưa có trong tài liệu\" không nằm trong tài liệu của hệ thống; cần cán bộ bảo vệ thực vật xác nhận trước khi dùng.")
        if flags["other_crop"]:
            parts.append("Mục đánh dấu \"chưa đăng ký cho cây này\" chưa có đăng ký cho cây trồng này tại Việt Nam.")
        if flags["biocontrol"]:
            parts.append("Chế phẩm sinh học (" + ", ".join(dict.fromkeys(flags["biocontrol"])) + "): chọn sản phẩm thương mại đã đăng ký cho cây trồng này.")
        return ("\n\n⚠️ " + MARK + " của hệ thống: " + " ".join(parts)) if parts else ""
