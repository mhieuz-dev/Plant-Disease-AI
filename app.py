"""
app_v3.py — Plant Doctor AI (RAG + GradCAM)
"""

import os
import json
import re
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from models.predict import predict_disease_detailed
from services.chatgpt_service import ask_chatgpt, ask_chatgpt_stream, validate_plant_image
from services.gradcam_service import generate_gradcam
from services.rag_service import (
    build_all_indexes,
    retrieve_context,
    build_rag_prompt,
    get_display_name,
    is_healthy,
    _vector_stores,
)

import torch, timm

def _load_model():
    ckpt    = torch.load("best_mobilevit.pth", map_location="cpu")
    classes = ckpt["classes"]
    model   = timm.create_model("mobilevit_s", pretrained=False, num_classes=len(classes))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, classes

_model, _classes = _load_model()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Building RAG indexes ...")
    build_all_indexes()
    print("[Startup] Ready!")
    yield

app = FastAPI(title="Plant Doctor AI", version="3.0", lifespan=lifespan)


# ── SESSION ────────────────────────────────────────────────────────────────────

sessions: dict[str, dict] = {}
MAX_HISTORY = 20

def get_session(cid: str) -> dict:
    if cid not in sessions:
        sessions[cid] = {"class_name": None, "crop": None, "history": []}
    return sessions[cid]

def trim(h: list) -> list:
    return h[-MAX_HISTORY:] if len(h) > MAX_HISTORY else h


# ── MODELS ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    conversation_id: str
    question: str


# ── /conversation/new ──────────────────────────────────────────────────────────

@app.post("/conversation/new")
async def new_conversation():
    cid = str(uuid.uuid4())
    sessions[cid] = {"class_name": None, "crop": None, "history": []}
    return {"conversation_id": cid}


# ── /detect ────────────────────────────────────────────────────────────────────

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conversation_id: Optional[str] = None,
):
    cid     = conversation_id or str(uuid.uuid4())
    session = get_session(cid)

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(400, "File ảnh rỗng")

    # 1. Validate ảnh
    try:
        validation = validate_plant_image(image_bytes)
    except Exception as e:
        raise HTTPException(500, f"Lỗi validate ảnh: {e}")

    if not validation["is_plant"]:
        return {
            "conversation_id": cid,
            "disease": None,
            "heatmap_b64": None,
            "bbox": None,
            "coverage_pct": None,
            "solution": validation["message"],
            "citations": [],
        }

    # 2. Predict
    try:
        detail = predict_disease_detailed(image_bytes)
    except Exception as e:
        raise HTTPException(500, f"Lỗi nhận diện: {e}")

    class_name   = detail["disease"]          # vd: "cafe_Leaf rust"
    confidence   = detail["confidence"]
    display_name = get_display_name(class_name)  # vd: "Bệnh gỉ sắt cà phê"

    session["class_name"] = class_name

    # 3. GradCAM
    gradcam = {}
    try:
        gradcam = generate_gradcam(
            image_bytes=image_bytes,
            model=_model,
            classes=_classes,
        )
    except Exception as e:
        print(f"[GradCAM] {e}")

    # 4. RAG — truyền class_name thực tế, không phải display_name
    rag_result = retrieve_context(
        class_name=class_name,
        user_query="",
        top_k=3,
    )
    session["crop"] = rag_result.get("crop")

    # 5. Build prompt + gọi LLM
    prompt = build_rag_prompt(
        class_name=class_name,
        user_question="",
        rag_result=rag_result,
        confidence=confidence,
    )

    session["history"].append({"role": "user", "content": prompt})

    try:
        solution = ask_chatgpt(session["history"])
    except Exception as e:
        raise HTTPException(500, f"Lỗi ChatGPT: {e}")

    session["history"].append({"role": "assistant", "content": solution})
    session["history"] = trim(session["history"])

    return {
        "conversation_id": cid,
        "class_name":      class_name,
        "disease":         display_name,
        "confidence":      confidence,
        "top3":            detail.get("top3", []),
        "heatmap_b64":     gradcam.get("heatmap_b64"),
        "bbox":            gradcam.get("bbox"),
        "coverage_pct":    gradcam.get("coverage_pct"),
        "crop":            rag_result.get("crop"),
        "citations":       rag_result.get("citations", []),
        "solution":        solution,
    }


# ── /chat/stream ───────────────────────────────────────────────────────────────

def _extract_class_from_question(question: str) -> Optional[str]:
    m = re.match(r'\[Cây đang mắc bệnh:\s*(.+?)\]', question)
    return m.group(1).strip() if m else None


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Câu hỏi trống")

    session    = get_session(req.conversation_id)
    class_name = session.get("class_name")
    extracted  = _extract_class_from_question(req.question)
    if extracted:
        class_name = extracted
        session["class_name"] = class_name

    rag_result = {}
    if class_name and not is_healthy(class_name):
        rag_result = retrieve_context(
            class_name=class_name,
            user_query=req.question,
            top_k=4,
        )

    if rag_result.get("context_str"):
        user_content = (
            f"[Tài liệu tham khảo]\n{rag_result['context_str']}\n\n"
            f"[Câu hỏi] {req.question}"
        )
    elif class_name:
        display_name = get_display_name(class_name)
        user_content = f"[Cây đang mắc bệnh: {display_name}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})
    snapshot = list(session["history"])

    async def event_gen():
        full = []
        try:
            for chunk in ask_chatgpt_stream(snapshot):
                full.append(chunk)
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            session["history"].append({"role": "assistant", "content": "".join(full)})
            session["history"] = trim(session["history"])
        except Exception as e:
            yield f"data: {json.dumps('⚠️ Lỗi: ' + str(e), ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
            if session["history"]:
                session["history"].pop()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── /chat (fallback) ───────────────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Câu hỏi trống")

    session    = get_session(req.conversation_id)
    class_name = session.get("class_name")
    extracted  = _extract_class_from_question(req.question)
    if extracted:
        class_name = extracted
        session["class_name"] = class_name

    rag_result = {}
    if class_name and not is_healthy(class_name):
        rag_result = retrieve_context(class_name=class_name, user_query=req.question, top_k=4)

    if rag_result.get("context_str"):
        user_content = f"[Tài liệu tham khảo]\n{rag_result['context_str']}\n\n[Câu hỏi] {req.question}"
    elif class_name:
        user_content = f"[Cây đang mắc: {get_display_name(class_name)}]\n{req.question}"
    else:
        user_content = req.question

    session["history"].append({"role": "user", "content": user_content})

    try:
        answer = ask_chatgpt(session["history"])
    except Exception as e:
        session["history"].pop()
        raise HTTPException(500, f"Lỗi ChatGPT: {e}")

    session["history"].append({"role": "assistant", "content": answer})
    session["history"] = trim(session["history"])

    return {
        "conversation_id": req.conversation_id,
        "answer":          answer,
        "citations":       rag_result.get("citations", []),
    }


# ── Admin & Health ─────────────────────────────────────────────────────────────

@app.post("/admin/rebuild-indexes")
async def rebuild(secret: str = ""):
    if secret != os.getenv("ADMIN_SECRET", "plant123"):
        raise HTTPException(403, "Sai secret")
    build_all_indexes()
    return {"status": "rebuilt"}

@app.delete("/conversation/{cid}")
async def delete_conv(cid: str):
    sessions.pop(cid, None)
    return {"status": "deleted"}

@app.get("/health")
async def health():
    return {
        "status":          "ok",
        "sessions_active": len(sessions),
        "rag_loaded":      list(_vector_stores.keys()),
    }