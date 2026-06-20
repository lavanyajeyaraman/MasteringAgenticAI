from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from semanticats.graph.graph import build_graph
from semanticats.state import RecruitingState
from semanticats.tools.pdf_parser import parse_pdf
from semanticats.tools.session_overrides import parse_override

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="semanticATSRecruit", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "semanticATSRecruit"}


@app.post("/match")
async def match_candidates(
    jd_text: str = Form(...),
    resumes: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    candidates = []
    for resume in resumes:
        content = await resume.read()
        if resume.filename and resume.filename.lower().endswith(".pdf"):
            path = f"/tmp/{resume.filename}"
            with open(path, "wb") as handle:
                handle.write(content)
            text = parse_pdf(path)
        else:
            text = content.decode("utf-8", errors="ignore")
        candidates.append({"filename": resume.filename, "text": text})
    state: RecruitingState = {
        "jd_text": jd_text,
        "jd_approved": True,
        "candidates_raw": candidates,
        "recruiter_filters": {},
        "conversation_history": [],
    }
    result = build_graph().invoke(state, config={"configurable": {"thread_id": "api"}})
    return JSONResponse(result)


@app.post("/override")
def override(command: dict[str, Any]) -> dict[str, Any]:
    text = str(command.get("command", ""))
    return {"filters": parse_override(text)}
