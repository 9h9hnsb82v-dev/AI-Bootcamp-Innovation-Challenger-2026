"""챗봇 문서 검색(RAG, 2026-11-18 사용자 승인). 법령·폐기물 분류번호·데이터 출처·이미지 판정 모델 비교·공공데이터(가격·통계)·앱 사용 설명.

- 문서: data/docs/**/*.md(원본 폴더 제외) + reports 의 모델 비교·데이터 감사 보고서(목록 REPORTS).
- 조각: '## ' 머리글(법령은 조문) 단위, 길면 문단으로 나눔(최대 MAX_CHARS). 조각 앞에 '문서명 > 섹션'을 붙인다.
- 검색: 키워드(BM25, 한국어 글자 2-gram + 영문·숫자·분류번호 토큰) + 의미(Ollama bge-m3 임베딩) 순위를 RRF 로 합친다.
  임베딩이 안 되면(Ollama 꺼짐·모델 없음) 키워드 검색만 쓴다.
- 색인 캐시: data/docs/.index/ (원본 파일 크기·수정시각이 바뀌면 다시 만든다). 미리 만들기: python -m services.doc_search --build
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import threading
from collections import Counter
from pathlib import Path

import numpy as np

from services.config import ROOT, ServiceError

DOCS = ROOT / "data/docs"
INDEX = DOCS / ".index"
REPORTS = ["reports/vision/dino_sd_eval.md", "reports/vision/sd_eval.md", "reports/vision/model_scoreboard.md",
           "reports/vision/app_model_comparison.md", "reports/vision/full_data_retrain.md",
           "reports/vision/model_comparison.md", "reports/vision/part_segmentation.md", "reports/data_audit_summary.md", "reports/vision/gap_dino_eval.md"]
TITLES = {"reports/vision/dino_sd_eval.md": "이미지 판정 모델 비교: DINOv2 vs YOLO11m vs EfficientNet (세션 비겹침 시험셋)",
          "reports/vision/sd_eval.md": "이미지 판정 세션 비겹침 재평가 (이전 조합 EfficientNet+YOLO11m, 현재 앱은 EfficientNet+DINOv2)",
          "reports/vision/model_scoreboard.md": "이미지 판정 모델 성능표 (8:2:1 분할)",
          "reports/vision/app_model_comparison.md": "이미지 판정 이전 앱 vs 현재 앱 비교",
          "reports/vision/full_data_retrain.md": "이미지 판정 원본 전체 재학습 결과",
          "reports/vision/model_comparison.md": "근접 사진 판정 모델 비교 (EfficientNet / YOLOv7 / YOLO11)",
          "reports/vision/part_segmentation.md": "차량 부품 위치(분할) 모델",
          "reports/data_audit_summary.md": "데이터 감사 요약",
          "reports/vision/gap_dino_eval.md": "휀더 단차 판정 모델 비교: EfficientNet vs DINOv2 (현재 앱은 DINOv2)"}
MAX_CHARS = 900
HIT_CHARS = 900  # 모델에 넘기는 조각 길이 = 조각 최대 길이(표 조각 끝 행이 잘리지 않게, 2026-11-19)
EMBED_MODEL = os.environ.get("RESOURCE_EMBED_MODEL", "bge-m3")
_LOCK = threading.Lock()


def _sources() -> list[Path]:
    files = sorted(p for p in DOCS.rglob("*.md") if ".index" not in p.parts and "원본" not in p.parts)
    return files + [ROOT / r for r in REPORTS if (ROOT / r).exists()]


def _signature(files: list[Path]) -> str:
    h = hashlib.sha256()  # 내용 기준(수정시각 아님) → 색인 폴더를 VM 에 그대로 복사해도 다시 만들지 않는다
    for p in files:
        h.update(p.relative_to(ROOT).as_posix().encode() + b"|" + hashlib.sha256(p.read_bytes()).digest())
    h.update(f"{MAX_CHARS}|{EMBED_MODEL}".encode())
    return h.hexdigest()[:16]


def _title(p: Path, text: str) -> str:
    rel = p.relative_to(ROOT).as_posix()
    if rel in TITLES:
        return TITLES[rel]
    m = re.search(r"^# (.+)$", text, flags=re.M)
    return m.group(1).strip() if m else p.stem


def _split(body: str) -> list[str]:
    if len(body) <= MAX_CHARS:
        return [body]
    out, cur = [], ""
    for para in re.split(r"\n(?=\S)", body):  # 줄 단위로 모아 자른다(표 행이 섞여도 행 단위 유지)
        while len(para) > MAX_CHARS:
            out.append(para[:MAX_CHARS])
            para = para[MAX_CHARS:]
        if len(cur) + len(para) + 1 > MAX_CHARS and cur:
            out.append(cur)
            cur = ""
        cur = (cur + "\n" + para) if cur else para
    if cur:
        out.append(cur)
    return out


def chunk_file(p: Path) -> list[dict]:
    text = p.read_text(encoding="utf-8")
    title = _title(p, text)
    parts = re.split(r"^(?=## )", text, flags=re.M)
    head, chunks = parts[0], []
    meta_lines = [ln for ln in head.splitlines() if ln.startswith("- 출처") or ln.startswith("- 시행일")
                  or "단위" in ln or ln.startswith("- 가격 시점")]
    meta = " ".join(ln.lstrip("- ").strip() for ln in meta_lines)[:300]
    sections = [("개요", head)] + [(s.splitlines()[0].lstrip("# ").strip(), s) for s in parts[1:]]
    table_head = ""
    for sec, body in sections:
        body = body.strip()
        if not body or (sec == "개요" and len(parts) > 1 and len(body) < 40):
            continue
        th = re.search(r"^(\|.+\|\n\|[-:| ]+\|)", body, flags=re.M)  # 표 머리글: 잘린 조각에도 붙인다
        if th:
            table_head = th.group(1)
        for i, piece in enumerate(_split(body)):
            if i and table_head and piece.lstrip().startswith("|") and table_head not in piece:
                piece = table_head + "\n" + piece
            label = sec if sec != "개요" else "개요"
            chunks.append({"doc": title, "section": label[:80], "path": p.relative_to(ROOT).as_posix(),
                           "meta": meta, "text": piece})
    return chunks


_TOK = re.compile(r"\d{2}-\d{2}-\d{2}|\d{2}-\d{2}|[A-Za-z]+|\d+(?:\.\d+)?|[가-힣]+")


def tokens(s: str) -> list[str]:
    out = []
    for t in _TOK.findall(s.lower()):
        if re.fullmatch(r"[가-힣]+", t):
            out += [t[i:i + 2] for i in range(len(t) - 1)] or [t]
        else:
            out.append(t)
    return out


class _BM25:
    def __init__(self, docs: list[list[str]], k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.tf = [Counter(d) for d in docs]
        self.len = np.array([len(d) for d in docs], dtype=float)
        self.avg = float(self.len.mean() or 1)
        df = Counter(t for d in self.tf for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, q: list[str]) -> np.ndarray:
        s = np.zeros(len(self.tf))
        for t in set(q):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, tf in enumerate(self.tf):
                f = tf.get(t)
                if f:
                    s[i] += idf * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
        return s


def _embed(texts: list[str], timeout: float = 120) -> np.ndarray | None:
    import requests
    from agent.ollama_client import BASE_URL
    url = BASE_URL.replace("//localhost", "//127.0.0.1")  # Windows 에서 localhost 는 IPv6 를 먼저 시도해 매번 약 2초 지연(실측)
    out = []
    body = {"model": EMBED_MODEL}
    if os.environ.get("RESOURCE_EMBED_CPU") == "1":  # 대회 체험(2026-11-20): GPU 메모리는 챗봇 동시 처리에 양보, 짧은 질문 임베딩은 CPU로
        body["options"] = {"num_gpu": 0}
    try:
        for i in range(0, len(texts), 32):
            r = requests.post(f"{url}/api/embed", json={**body, "input": texts[i:i + 32]}, timeout=timeout)
            r.raise_for_status()
            out += r.json()["embeddings"]
    except Exception:  # noqa: BLE001 - 임베딩 실패 시 키워드 검색만
        return None
    v = np.asarray(out, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


_STATE: dict = {}


def build(force: bool = False, with_embeddings: bool = True) -> dict:
    """색인을 만들거나 캐시에서 읽는다. 반환: {chunks, bm25, emb(없으면 None), signature}."""
    with _LOCK:
        files = _sources()
        sig = _signature(files)
        if not force and _STATE.get("signature") == sig and (_STATE.get("emb") is not None or not with_embeddings):
            return _STATE
        INDEX.mkdir(parents=True, exist_ok=True)
        meta_p, emb_p = INDEX / "chunks.json", INDEX / "emb.npy"
        chunks, emb = None, None
        if not force and meta_p.exists():
            saved = json.loads(meta_p.read_text(encoding="utf-8"))
            if saved.get("signature") == sig:
                chunks = saved["chunks"]
                if emb_p.exists():
                    emb = np.load(emb_p)
                    if len(emb) != len(chunks):
                        emb = None
        if chunks is None:
            chunks = [c for p in files for c in chunk_file(p)]
            meta_p.write_text(json.dumps({"signature": sig, "chunks": chunks}, ensure_ascii=False), encoding="utf-8")
            if emb_p.exists():
                emb_p.unlink()
        if emb is None and with_embeddings:
            emb = _embed([f"{c['doc']} > {c['section']}\n{c['text']}" for c in chunks])
            if emb is not None:
                np.save(emb_p, emb)
        _STATE.clear()
        _STATE.update(signature=sig, chunks=chunks, emb=emb,
                      bm25=_BM25([tokens(f"{c['doc']} {c['section']} {c['text']}") for c in chunks]))
        return _STATE


def search(query: str, top_k: int = 3) -> dict:
    q = (query or "").strip()
    if len(q) < 2:
        raise ServiceError("INVALID_INPUT", "검색어를 두 글자 이상 넣으세요.")
    top_k = max(1, min(int(top_k or 3), 6))
    st = build()
    chunks = st["chunks"]
    bm = st["bm25"].scores(tokens(q))
    ranks = [np.argsort(-bm)[:50]]
    mode = "키워드"
    if st["emb"] is not None:
        qv = _embed([q], timeout=30)
        if qv is not None:
            ranks.append(np.argsort(-(st["emb"] @ qv[0]))[:50])
            mode = "키워드+의미"
    rrf: dict[int, float] = {}
    for r in ranks:
        for pos, i in enumerate(r):
            if r is ranks[0] and bm[i] <= 0:
                continue
            rrf[int(i)] = rrf.get(int(i), 0) + 1 / (60 + pos)
    best = sorted(rrf, key=rrf.get, reverse=True)[:top_k]
    hits = [{"doc": chunks[i]["doc"], "section": chunks[i]["section"][:60], "source_meta": chunks[i]["meta"][:160],
             "text": chunks[i]["text"][:HIT_CHARS]} for i in best]
    return {"query": q, "mode": mode, "hits": hits, "provenance": ["DOC_SEARCH"],
            "note": "문서 조각 원문. 답변에는 문서명·조문(섹션)을 밝히고, 문서에 없는 내용은 모른다고 한다."}


if __name__ == "__main__":
    if "--build" in sys.argv:
        s = build(force=True)
        print(f"조각 {len(s['chunks'])}개, 임베딩 {'있음 ' + str(s['emb'].shape) if s['emb'] is not None else '없음(키워드만)'}")
    else:
        for h in search(" ".join(a for a in sys.argv[1:] if not a.startswith("--")) or "전자인계")["hits"]:
            print(f"- {h['doc']} > {h['section']}\n  {h['text'][:160]!r}")
