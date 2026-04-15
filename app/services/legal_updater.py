"""Legal data updater: crawls 판례 and 법령 from law.go.kr and upserts into Qdrant."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import structlog
from bs4 import BeautifulSoup
from qdrant_client.http.models import PointStruct

from app.config import settings
from app.services.embeddings import embed
from app.services.rag import CASES_COLLECTION_NAME, _get_client, ensure_cases_collection

log = structlog.get_logger()

_API_BASE = "https://www.law.go.kr/DRF"
# 판례 검색 키워드 (전체 텍스트 검색)
_PREC_QUERIES = [
    "불공정계약",
    "손해배상",
    "계약해지",
    "약관",
    "위약금",
    "기밀유지",
    "지식재산권",
]
# 법령 검색 키워드
# section=all 파라미터를 사용하지 않음 — target=law 시 section=all이 빈 결과를 반환함
# (law.go.kr DRF API: target=law 기본값이 이미 법령명+본문 검색)
_LAW_QUERIES = [
    "약관규제법",
    "하도급거래",
    "공정거래",
    "개인정보보호법",
    "손해배상",
    "지식재산기본법",
    "전자상거래",
]
_DISPLAY = 10


def _to_uuid(source_id: str, ref_type: str) -> str:
    """Deterministic UUID from source_id — used for upsert deduplication."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ref_type}:{source_id}"))


async def _fetch(client: httpx.AsyncClient, path: str, params: dict[str, Any]) -> dict:
    resp = await client.get(f"{_API_BASE}/{path}", params=params, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


async def _crawl_cases(client: httpx.AsyncClient, oc: str) -> list[dict]:
    seen: set[str] = set()
    docs: list[dict] = []

    for query in _PREC_QUERIES:
        try:
            data = await _fetch(
                client,
                "lawSearch.do",
                {
                    "OC": oc,
                    "target": "prec",
                    "type": "JSON",
                    "query": query,
                    "display": _DISPLAY,
                },
            )
            items = data.get("PrecSearch", {}).get("prec", [])
            if isinstance(items, dict):
                items = [items]
            for item in items:
                seq = str(item.get("판례일련번호", ""))
                if not seq or seq in seen:
                    continue
                seen.add(seq)
                try:
                    detail = await _fetch(
                        client,
                        "lawService.do",
                        {
                            "OC": oc,
                            "target": "prec",
                            "type": "JSON",
                            "ID": seq,
                        },
                    )
                    d = detail.get("PrecService", {})
                    raw = d.get("판례내용", "") or d.get("전문", "")
                    content = BeautifulSoup(raw, "html.parser").get_text()[:3000]
                    if not content.strip():
                        continue
                    docs.append(
                        {
                            "type": "prec",
                            "source_id": seq,
                            "title": d.get("사건명", ""),
                            "content": content,
                            "date": d.get("선고일자", ""),
                            "court": d.get("법원명", ""),
                        }
                    )
                except Exception as exc:
                    log.warning("판례 상세 조회 실패", seq=seq, error=str(exc))
        except Exception as exc:
            log.warning("판례 검색 실패", query=query, error=str(exc))

    return docs


async def _crawl_laws(client: httpx.AsyncClient, oc: str) -> list[dict]:
    seen: set[str] = set()
    docs: list[dict] = []

    for query in _LAW_QUERIES:
        try:
            # section=all 파라미터 제거: target=law 시 해당 파라미터가 빈 결과를 유발
            data = await _fetch(
                client,
                "lawSearch.do",
                {
                    "OC": oc,
                    "target": "law",
                    "type": "JSON",
                    "query": query,
                    "display": _DISPLAY,
                },
            )
            law_search = data.get("LawSearch", {})
            total_cnt = law_search.get("totalCnt", 0)
            raw_items = law_search.get("law", [])
            log.debug(
                "법령 검색 응답",
                query=query,
                total_cnt=total_cnt,
                has_law_key="law" in law_search,
                raw_type=type(raw_items).__name__,
            )
            if isinstance(raw_items, dict):
                items: list[dict] = [raw_items]
            elif isinstance(raw_items, list):
                items = raw_items
            else:
                log.warning(
                    "법령 검색 응답 law 키 타입 이상",
                    query=query,
                    raw_type=type(raw_items).__name__,
                )
                items = []
            for item in items:
                law_id = str(item.get("법령일련번호", ""))
                if not law_id or law_id in seen:
                    continue
                seen.add(law_id)
                try:
                    detail = await _fetch(
                        client,
                        "lawService.do",
                        {
                            "OC": oc,
                            "target": "law",
                            "type": "JSON",
                            "ID": law_id,
                        },
                    )
                    d = detail.get("LawService", {})
                    log.debug(
                        "법령 상세 응답 키",
                        law_id=law_id,
                        top_keys=list(d.keys())[:10],
                    )
                    # law.go.kr DRF 응답에서 조문 데이터는 '조문' 또는 '법령본문' 키에 위치할 수 있음
                    articles = d.get("조문") or d.get("법령본문") or []
                    if isinstance(articles, dict):
                        articles = [articles]
                    elif not isinstance(articles, list):
                        articles = []
                    content_parts: list[str] = []
                    for a in articles[:20]:
                        title = a.get("조문제목", "") or a.get("조문번호", "")
                        body = a.get("조문내용", "") or a.get("조문", "")
                        part = f"{title}\n{body}".strip()
                        if part:
                            content_parts.append(part)
                    content = "\n\n".join(content_parts)[:3000]
                    if not content.strip():
                        # 조문이 없을 경우 법령명만이라도 남겨 index에 포함
                        law_name = d.get("법령명한글", "") or item.get("법령명한글", "")
                        if not law_name:
                            log.debug("법령 조문 없음, 건너뜀", law_id=law_id)
                            continue
                        content = law_name
                    docs.append(
                        {
                            "type": "law",
                            "source_id": law_id,
                            "title": d.get("법령명한글", "")
                            or item.get("법령명한글", ""),
                            "content": content,
                            "date": d.get("공포일자", ""),
                            "court": "",
                        }
                    )
                except Exception as exc:
                    log.warning("법령 상세 조회 실패", law_id=law_id, error=str(exc))
        except Exception as exc:
            log.warning("법령 검색 실패", query=query, error=str(exc))

    return docs


async def run_update() -> None:
    """Crawl 판례 + 법령 and upsert into Qdrant cases collection."""
    oc = settings.law_api_oc
    if not oc:
        log.warning("LAW_API_OC not configured — skipping legal data update")
        return

    log.info("legal data update started")
    await ensure_cases_collection()

    async with httpx.AsyncClient() as http:
        cases = await _crawl_cases(http, oc)
        laws = await _crawl_laws(http, oc)

    all_docs = cases + laws
    log.info(
        "crawled legal documents", cases=len(cases), laws=len(laws), total=len(all_docs)
    )

    if not all_docs:
        log.warning("no legal documents crawled — nothing to upsert")
        return

    texts = [f"{d['title']}\n{d['content']}" for d in all_docs]
    vectors = await embed(texts)

    qdrant = _get_client()
    points = [
        PointStruct(
            id=_to_uuid(d["source_id"], d["type"]),
            vector=v,
            payload={
                "type": d["type"],
                "source_id": d["source_id"],
                "title": d["title"],
                "content": d["content"][:1000],
                "date": d.get("date", ""),
                "court": d.get("court", ""),
            },
        )
        for d, v in zip(all_docs, vectors)
    ]

    await qdrant.upsert(collection_name=CASES_COLLECTION_NAME, points=points)
    log.info("legal data upserted", count=len(points))
