# Plant Disease Diagnosis System — Session Reference

## Project Overview

Mobile AI system for plant disease diagnosis targeting Vietnamese farmers. Detects disease from photo, generates agrochemical advisory with citations, streams advice to Flutter app.

**Three crops / 12 disease classes:**

| Crop | Classes |
|------|---------|
| Cà phê (coffee) | Leaf rust, Miner (Leucoptera coffeella), Phoma (Phoma costarricensis), Healthy |
| Lúa (rice) | Leaf Blast, Leaf Scald, Sheath Blight, Healthy |
| Hồ tiêu (pepper) | Footrot (Phytophthora capsici), Pollu/Anthracnose (Colletotrichum), Slow Decline (Fusarium + Radopholus), Healthy |

---

## Model Selection

| Model | Params | Test Accuracy |
|-------|--------|---------------|
| MobileViT-S | 4.95M | **87.83%** ✓ chosen |
| EfficientNetV2-S | 20.19M | 87.75% |

Chosen: MobileViT-S — fewer parameters, same accuracy.

---

## System Architecture

```
[Flutter App]
      │
      ▼  REST / SSE
[FastAPI Server  app.py]
      │
      ├──/detect──► MobileViT-S ──► GradCAM ──► GPT-4.1-mini (validate + advise)
      │                                               ▲
      ├──/chat────────────────────────────────────────┤
      │                                           RAG pipeline
      └──/chat/stream (SSE) ──────────────────────────┘

[RAG Pipeline]
  PDF/TXT docs ──► RecursiveCharacterTextSplitter (800 chars, 100 overlap)
               ──► OpenAI text-embedding-3-small
               ──► FAISS flat index (per-crop: ca_phe/, lua/, tieu/)
               ──► top_k=4 similarity search
               ──► context injected into GPT-4.1-mini prompt
```

---

## Running the Server

```bash
# Start server
$env:PYTHONIOENCODING="utf-8"
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Rebuild FAISS indexes (after adding/changing documents)
curl -X POST http://localhost:8000/admin/rebuild-indexes?secret=plant123
```

---

## Running Tests

```bash
$env:PYTHONIOENCODING="utf-8"

# RAG evaluation (sends class prefix so RAG triggers)
python test_rag.py    → rag_results.json

# Vanilla LLM evaluation (no prefix, RAG does not trigger)
python test_vanilla.py → vanilla_results.json
```

---

## RAG vs Vanilla LLM Evaluation Results

**20-question evaluation set across all 12 disease classes.**

| Configuration | Citation (%) | Grounded (%) | Hallucination (%) |
|--------------|-------------|-------------|-----------------|
| RAG-enabled  | 100.0       | 80.0        | 20.0            |
| Vanilla LLM  | 0.0         | 70.0        | 30.0            |

**Scoring criteria:**
- **Citation** — response includes at least one `[1] filename, trang X` citation
- **Grounded** — all factual claims are supported by retrieved documents
- **Hallucination** — response contains claims absent from or contradicted by retrieved documents

**Vanilla LLM failure cases (6/20 = 30%):**
- Misidentified Phytophthora capsici as Fusarium solani for pepper foot rot
- Wrong environmental conditions described for rice sheath blight
- No specific drug dosages (general guesses only)

**RAG failure cases (4/20 = 20%) — Q3, Q4, Q5, Q6 (all coffee diseases):**
- Q3/Q4 (cafe_Miner): Model recommended Indoxacarb/Chlorpyrifos/Spinosad — none are in the retrieved Sau_ve_bua_supplement.txt. Supplement recommends Thiamethoxam, Emamectin benzoate, Chlorantraniliprole. Chlorpyrifos is explicitly cautioned against in section 6.
- Q5 (cafe_Phoma symptoms): Answer attributed "bột mịn vàng cam" (orange powder) to Phoma — this is a leaf rust symptom, caused by cross-document contamination (Gi_sat_1.pdf retrieved alongside Phoma_supplement.txt).
- Q6 (cafe_Phoma drugs): Answer recommended SAGOPERFECT 320SC / LUNASA 25EC — these appear in the leaf rust document (Gi_sat_1.pdf), not the Phoma supplement.

**RAG success highlights (16/20 = 80% grounded):**
- Rice questions (Q8–Q14): Specific dosages cited from Dao_on_la.pdf, 11-9-370-345.pdf, Kho_van.pdf ✓
- Pepper questions (Q15–Q20): Correct pathogen IDs and drug protocols after adding 3 supplement docs ✓

---

## Bugs Fixed in app.py

### Bug 1 — Citations always empty in /chat

**Root cause:** `class_name = session.get("class_name")` was `None` because the test script never calls `/detect` first; it goes directly to `/chat`.

**Fix:** Added `_extract_class_from_question()` regex parser:
```python
def _extract_class_from_question(question: str) -> Optional[str]:
    m = re.match(r'\[Cây đang mắc bệnh:\s*(.+?)\]', question)
    return m.group(1).strip() if m else None
```

Test script sends questions prefixed as: `[Cây đang mắc bệnh: cafe_Leaf rust]\nBệnh gỉ sắt...`

### Bug 2 — Wrong class carried across sessions

**Root cause:** `if not class_name:` guard prevented updating `class_name` once it was set. After Q1 set `class_name="cafe_Leaf rust"`, Q8 (rice) kept routing to the coffee FAISS index.

**Fix:** Changed to always override when prefix present:
```python
extracted = _extract_class_from_question(req.question)
if extracted:
    class_name = extracted
    session["class_name"] = class_name
```

Applied in both `/chat` and `/chat/stream` endpoints.

### Improvement — top_k increased 2 → 4

Changed `top_k=4` in both endpoints so more document chunks are retrieved per query, reducing the chance of missing relevant sections.

---

## Documents Added (Real Sources Only — No Fabrication)

### `docs/ca_phe/Sau_ve_bua_supplement.txt`
Coffee leaf miner (Leucoptera coffeella)
- Sources: Rocha et al. (2020) MDPI Sustainability 12(19):8011; Picanço et al. (2002) Bulletin of Entomological Research 92:509–516; Sở NN&PTNT Gia Lai/Lâm Đồng
- Drugs: Thiamethoxam (Actara 25WG) 10–12 g/16L, Imidacloprid 8–10 g/16L, Emamectin benzoate (Proclaim 1.9EC) 10–15 ml/16L, Abamectin (Tervigo 020SC) 5–8 ml/16L, Chlorantraniliprole (Prevathon 5SC) 10–12 ml/16L
- Season: Mùa mưa tháng 5–10, nhiệt độ 20–28°C, độ ẩm > 75%

### `docs/ca_phe/Phoma_supplement.txt`
Coffee brown eye spot (Phoma costarricensis)
- Sources: PestNet Fact Sheet #142; Martins et al. (2019) ResearchGate; dantocphattrien.vietnamnet.vn
- Drugs: Tebuconazole (Folicur 250EC) 10–15 ml/16L → reduces Phoma 50.9% (Martins 2019), Hexaconazole (Chevin 5SC) 40–60 ml/16L, Mancozeb 80WP 25–30 g/16L, Azoxystrobin 250SC 8–10 ml/16L, Copper oxychloride 30–40 g/10L

### `docs/tieu/Pollu_supplement.txt`
Pepper anthracnose / Pollu disease (Colletotrichum gloeosporioides)
- Sources: dost-bentre.gov.vn; agridrone.vn; happyagri.com.vn; santari.com.vn
- Drugs: Carbendazim/Bavistin 50FL 10–15 ml/10L (0.1%), Mancozeb (Dithane M45) 25–30 g/16L, Score 250EC (Difenoconazole) 10–15 ml/16L, Amistar 250SC (Azoxystrobin) 8–10 ml/16L

### `docs/tieu/Footrot_supplement.txt`
Pepper foot rot / chết nhanh (Phytophthora capsici)
- Sources: Dann et al. (2011) Australasian Plant Pathology 40(3); Sarma & Anandaraj (2000); Sở NN&PTNT Bình Phước/Đắk Lắk
- Drugs: Fosetyl-Al (Aliette 80WP) 30–40 g/10L, Potassium phosphonate 600SL 5–10 ml/L, Ridomil Gold 68WG (Metalaxyl+Mancozeb) 20–25 g/10L, Copper Oxychloride 30–40 g/10L
- Includes 6-step emergency protocol for first infected plant

### `docs/tieu/Slow_Decline_supplement.txt`
Pepper slow decline / chết chậm (Fusarium solani + Radopholus similis)
- Sources: Anandaraj et al. (2009) tài liệu 116–120; Chet_Cham.pdf; Sở NN&PTNT Bình Phước/Đắk Lắk
- Treatment: neem cake 1 kg/gốc, Trichoderma harzianum 50 g/gốc, Carbofuran (Furadan 3G) 30–50 g/gốc, Abamectin 3.6EC 10 ml/16L
- Explicitly states: NOT the same as anthracnose (Pollu) or foot rot (chết nhanh). Do NOT use Fosetyl-Al/Metalaxyl for this disease.

---

## FAISS Index Structure

```
faiss_indexes/
├── ca_phe/     ← indexes coffee docs (Gi_sat_1.pdf, Than_thu.pdf, Sau_ve_bua_supplement.txt, Phoma_supplement.txt)
├── lua/        ← indexes rice docs (Dao_on_la.pdf, Kho_van.pdf, 11-9-370-345.pdf, ...)
└── tieu/       ← indexes pepper docs (Pollu_supplement.txt, Footrot_supplement.txt, Slow_Decline_supplement.txt)
```

**Routing logic** in `rag_service.py`:
- `cafe_*` → `ca_phe/` index
- `rice_*` → `lua/` index
- `tieu_*` → `tieu/` index
- Healthy classes → RAG not triggered

**Rebuild:** `POST /admin/rebuild-indexes?secret=plant123` or run `build_all_indexes()` directly.

---

## Key Technical Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Embedding model | text-embedding-3-small | OpenAI |
| LLM | GPT-4.1-mini | Advisory generation |
| Chunk size | 800 chars | RecursiveCharacterTextSplitter |
| Chunk overlap | 100 chars | |
| top_k | 4 | Chunks retrieved per query |
| Max history | 20 turns | Per session |
| GradCAM target | Last conv block | MobileViT-S |

---

## Post-processing: Miner/Rust Disambiguation

**File:** `evaluate_postprocessing.py`

**Concept:** After MobileViT-S outputs probabilities, if both `cafe_Leaf_rust`
and `cafe_Miner` appear in top-2 with confidence gap < 25%, apply color analysis
on the GradCAM bounding box region to distinguish them:
- Rust = orange-yellow urediniospore powder → HSV Hue ∈ [10°, 45°], S ≥ 80
- Miner = whitish/silver galleries → does NOT match the above

**Trigger condition:** top-2 contains both Miner AND Rust, |P_rust - P_miner| < 0.25

**Decision rule:** if orange-yellow pixel ratio in GradCAM ROI ≥ 0.12 → Rust, else → Miner

**Evaluation results (test set, 998 images: 548 Rust + 450 Miner):**

| Class | Recall (before) | F1 (before) | Recall (after) | F1 (after) |
|-------|----------------|-------------|----------------|------------|
| cafe_Leaf rust | 0.995 | 0.921 | 0.995 | 0.921 |
| cafe_Miner | 0.331 | 0.489 | 0.331 | 0.489 |

**Key finding:** Post-processing triggered **0 times**. The model misclassifies Miner with high confidence (gap >> 25%), so the ambiguity band is never entered. This is a high-confidence representational error, not a boundary-case uncertainty. Fix requires either relaxing the trigger condition or adding close-up training images.

**To reproduce:**
```bash
$env:PYTHONIOENCODING="utf-8"
python evaluate_postprocessing.py --test_dir "C:\Users\ASUS\Downloads\dataset1\test"
```
Results saved to `postproc_results.json`.

---

## Paper Structure (`main.txt`)

IEEE LaTeX for Overleaf. Section structure:
- 4.1 Dataset Collection and Statistics
- 4.2 Experimental Environment
- 4.3 Classification Results (F1 table, confusion matrix)
- **4.4 GradCAM Visualization Results** ← NEW (3 figures: coffee, rice, pepper)
  - Placeholders: `ảnh/gradcam_coffee.png`, `ảnh/gradcam_rice.png`, `ảnh/gradcam_pepper.png`
- **4.5 Advisory Quality: RAG vs Vanilla LLM** ← moved from 4.4
- **4.X Failure Case Analysis** ← updated with post-processing section + tab:postproc
- Appendix: full 20-question evaluation set

Key tables:
```
RAG-enabled  | 100.0% | 80.0%  | 20.0%
Vanilla LLM  | 0.0%   | 70.0%  | 30.0%

MobileViT-S      | 4.95M  | 87.83% ← selected
EfficientNetV2-S | 20.19M | 87.75%
```

**Images needed for paper (open app to capture):**
- `ảnh/gradcam_coffee.png` — 2×2 grid: Leaf Rust (original + GradCAM), Miner (original + GradCAM)
- `ảnh/gradcam_rice.png`   — 2×2 grid: Leaf Blast (original + GradCAM), Sheath Blight (original + GradCAM)
- `ảnh/gradcam_pepper.png` — 2×2 grid: Footrot (original + GradCAM), Pollu (original + GradCAM)

---

## Known Issues / Next Steps

1. **Q3/Q4 (cafe_Miner RAG):** Model ignores correct drugs in supplement, recommends Indoxacarb/Spinosad instead of Thiamethoxam/Emamectin benzoate. Fix: stricter attribution prompting.
2. **Q5/Q6 (cafe_Phoma RAG):** Cross-document contamination — leaf rust symptoms bleed into Phoma answer. Fix: per-disease FAISS sub-index.
3. **Miner vision confusion:** F1=0.489, Recall≈0.33 — post-processing in `evaluate_postprocessing.py` is proposed fix. Run to get actual numbers.
