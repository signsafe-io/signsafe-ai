"""Legal data updater: crawls 판례 and 법령 from law.go.kr and upserts into Qdrant.

law.go.kr DRF API 주의사항:
- type=JSON 은 빈 {} 를 반환함 — 반드시 type=XML 사용
- 법령 상세 조회: ID 파라미터가 아닌 MST=법령일련번호 사용
- 판례 상세 조회: ID=판례일련번호 사용 (기존 유지)
"""

from __future__ import annotations

import asyncio
import uuid
import xml.etree.ElementTree as ET
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
    # 계약 일반
    "불공정계약",
    "계약위반",
    "계약무효",
    "계약해제",
    "계약해지",
    "불이행",
    "이행강제",
    # 손해배상
    "손해배상",
    "손해배상예정",
    "손해배상제한",
    "손해배상청구",
    # 약관/불공정
    "약관",
    "불공정약관",
    "부당조항",
    "부당이득",
    # 위약금
    "위약금",
    "위약벌",
    "손해배상예정액감액",
    # 계약 해지권
    "일방적해지",
    "해지통보",
    "해지요건",
    # 기밀/비밀/경업
    "기밀유지",
    "비밀유지",
    "경업금지",
    "영업비밀",
    "전직금지",
    # IP
    "지식재산권",
    "저작권양도",
    "직무발명",
    "특허귀속",
    # 하도급/용역
    "하도급",
    "용역계약",
    "도급계약",
    "용역비",
    # 근로계약
    "근로계약",
    "프리랜서",
    "근로기준",
    "임금",
    # 임대차
    "임대차계약",
    "임대차보증금",
    "전세계약",
    # 불가항력
    "불가항력",
    "이행불능",
    "천재지변",
    # 준거법/관할
    "관할합의",
    "전속관할",
    "재판관할",
    # 대금지급
    "대금지급지연",
    "지체상금",
    "연체이자",
    # 면책
    "면책조항",
    "면책특약",
    "책임제한",
    # 계약갱신
    "계약갱신",
    "갱신거절",
    "계약기간",
    # 보증/담보
    "연대보증",
    "이행보증",
    "담보책임",
    # 상사/기업
    "주식매매",
    "영업양도",
    "합병계약",
    # 소비자
    "소비자보호",
    "환불",
    "청약철회",
    # 부동산/건설
    "도급계약위반",
    "건설계약",
    "분양계약",
    # 금융
    "대출계약",
    "보험계약",
    "리스계약",
    # 정보통신
    "소프트웨어계약",
    "IT용역",
    "데이터계약",
]

_LAW_QUERIES = [
    "약관규제",
    "하도급거래",
    "공정거래",
    "개인정보보호",
    "손해배상",
    "지식재산",
    "전자상거래",
    "근로기준",
    "저작권",
    "상법",
    "민법",
    "독점규제",
    "중소기업",
    "소비자보호",
    "부정경쟁방지",
    "발명진흥",
    "가맹사업",
    "대리점거래",
    "집합건물",
    "주택임대차",
    "상가임대차",
    "전자금융",
    "정보통신망",
]

_DISPLAY = 100  # 키워드당 최대 100개 (API 허용 최대치)


def _to_uuid(source_id: str, ref_type: str) -> str:
    """Deterministic UUID from source_id — used for upsert deduplication."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{ref_type}:{source_id}"))


async def _fetch_xml(
    client: httpx.AsyncClient, path: str, params: dict[str, Any]
) -> ET.Element:
    """Fetch XML response from law.go.kr DRF API.

    type=JSON returns empty {} for both lawSearch.do and lawService.do.
    type=XML returns proper structured data.
    """
    params = {**params, "type": "XML"}
    resp = await client.get(f"{_API_BASE}/{path}", params=params, timeout=30.0)
    resp.raise_for_status()
    return ET.fromstring(resp.text)


def _xml_text(el: ET.Element | None, tag: str, default: str = "") -> str:
    """Safely extract text from a child element."""
    if el is None:
        return default
    child = el.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _extract_law_content(root: ET.Element, law_id: str) -> tuple[str, str]:
    """Extract law name and article content from XML response.

    Returns (law_name, content). content may be empty if no articles found.
    XML structure:
      <법령>
        <기본정보>
          <법령명_한글>...</법령명_한글>
          ...
        </기본정보>
        <조문>
          <조문단위>
            <조문번호>...</조문번호>
            <조문제목>...</조문제목>
            <조문내용>...</조문내용>
            <항>
              <항번호>...</항번호>
              <항내용>...</항내용>
              <호>
                <호번호>...</호번호>
                <호내용>...</호내용>
              </호>
            </항>
          </조문단위>
        </조문>
      </법령>
    """
    # 오류 응답: <Law>일치하는 법령이 없습니다...</Law>
    if root.tag in ("Law", "법령") and root.find("기본정보") is None:
        if root.text and "일치하는" in root.text:
            log.debug("법령 상세 조회 — 일치하는 법령 없음", law_id=law_id)
            return "", ""

    basic = root.find("기본정보")
    law_name = _xml_text(basic, "법령명_한글")

    jomun_el = root.find("조문")
    if jomun_el is None:
        log.debug("법령 상세 응답에 조문 요소 없음", law_id=law_id, law_name=law_name)
        return law_name, ""

    units = jomun_el.findall("조문단위")
    log.debug(
        "법령 상세 응답 조문 수",
        law_id=law_id,
        law_name=law_name,
        article_count=len(units),
    )

    content_parts: list[str] = []
    for unit in units[:20]:
        num = _xml_text(unit, "조문번호")
        title = _xml_text(unit, "조문제목")
        body = _xml_text(unit, "조문내용")

        # 항 내용 수집
        sub_parts: list[str] = []
        for hang in unit.findall("항"):
            hang_body = _xml_text(hang, "항내용")
            if hang_body:
                sub_parts.append(hang_body)
            # 호 내용 수집
            for ho in hang.findall("호"):
                ho_body = _xml_text(ho, "호내용")
                if ho_body:
                    sub_parts.append(f"  {ho_body}")

        # 조문 텍스트 구성
        header = title or (f"제{num}조" if num else "")
        article_text = body
        if sub_parts:
            article_text = (body + "\n" + "\n".join(sub_parts)).strip()
        part = f"{header}\n{article_text}".strip() if header else article_text.strip()
        if part:
            content_parts.append(part)

    return law_name, "\n\n".join(content_parts)[:3000]




_DETAIL_CONCURRENCY = 10  # law.go.kr 상세 조회 동시 요청 수


async def _fetch_prec_detail(
    client: httpx.AsyncClient,
    oc: str,
    seq: str,
    sem: "asyncio.Semaphore",
) -> dict | None:
    """Fetch a single 판례 detail with concurrency control."""
    async with sem:
        try:
            detail_root = await _fetch_xml(
                client,
                "lawService.do",
                {"OC": oc, "target": "prec", "ID": seq},
            )
            d = detail_root if detail_root.tag == "PrecService" else detail_root
            raw_content = _xml_text(d, "판례내용") or _xml_text(d, "전문")
            content = BeautifulSoup(raw_content, "html.parser").get_text()[:3000]
            if not content.strip():
                return None
            return {
                "type": "prec",
                "source_id": seq,
                "title": _xml_text(d, "사건명"),
                "content": content,
                "date": _xml_text(d, "선고일자"),
                "court": _xml_text(d, "법원명"),
            }
        except Exception as exc:
            log.warning("판례 상세 조회 실패", seq=seq, error=str(exc))
            return None


async def _crawl_cases(client: httpx.AsyncClient, oc: str) -> list[dict]:
    seen: set[str] = set()
    # 키워드별 검색은 순차 실행, 상세 조회는 키워드 내에서 병렬 실행
    sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)
    docs: list[dict] = []

    for query in _PREC_QUERIES:
        try:
            root = await _fetch_xml(
                client,
                "lawSearch.do",
                {"OC": oc, "target": "prec", "query": query, "display": _DISPLAY},
            )
            items = root.findall("prec")
            log.debug(
                "판례 검색 응답",
                query=query,
                total_cnt=_xml_text(root, "totalCnt"),
                item_count=len(items),
            )
            # 이미 수집한 판례 제외 후 상세 병렬 조회
            new_seqs = []
            for item in items:
                seq = _xml_text(item, "판례일련번호")
                if seq and seq not in seen:
                    seen.add(seq)
                    new_seqs.append(seq)

            if new_seqs:
                results = await asyncio.gather(
                    *[_fetch_prec_detail(client, oc, seq, sem) for seq in new_seqs]
                )
                docs.extend(r for r in results if r is not None)
                log.info(
                    "판례 키워드 완료",
                    query=query,
                    fetched=len(new_seqs),
                    added=sum(1 for r in results if r is not None),
                    total_so_far=len(docs),
                )
        except Exception as exc:
            log.warning("판례 검색 실패", query=query, error=str(exc))

    return docs


async def _fetch_law_detail(
    client: httpx.AsyncClient,
    oc: str,
    law_id: str,
    law_name_from_search: str,
    pub_date_from_search: str,
    sem: "asyncio.Semaphore",
) -> dict | None:
    """Fetch a single 법령 detail with concurrency control."""
    async with sem:
        try:
            detail_root = await _fetch_xml(
                client,
                "lawService.do",
                {"OC": oc, "target": "law", "MST": law_id},
            )
            law_name, content = _extract_law_content(detail_root, law_id)
            law_name = law_name or law_name_from_search
            if not content.strip():
                if not law_name:
                    return None
                content = law_name
            basic = detail_root.find("기본정보")
            pub_date = _xml_text(basic, "공포일자") or pub_date_from_search
            return {
                "type": "law",
                "source_id": law_id,
                "title": law_name,
                "content": content,
                "date": pub_date,
                "court": "",
            }
        except Exception as exc:
            log.warning("법령 상세 조회 실패", law_id=law_id, error=str(exc))
            return None


async def _crawl_laws(client: httpx.AsyncClient, oc: str) -> list[dict]:
    """Crawl 법령 using XML API.

    법령 검색: lawSearch.do?target=law&type=XML
    법령 상세: lawService.do?target=law&type=XML&MST=법령일련번호
      - ID 파라미터는 동작하지 않음 — MST(법령일련번호)를 사용해야 함
    """
    seen: set[str] = set()
    sem = asyncio.Semaphore(_DETAIL_CONCURRENCY)
    docs: list[dict] = []

    for query in _LAW_QUERIES:
        try:
            root = await _fetch_xml(
                client,
                "lawSearch.do",
                {
                    "OC": oc,
                    "target": "law",
                    "query": query,
                    "display": _DISPLAY,
                },
            )
            items = root.findall("law")
            total_cnt = _xml_text(root, "totalCnt")
            log.debug(
                "법령 검색 응답",
                query=query,
                total_cnt=total_cnt,
                item_count=len(items),
            )
            new_items = []
            for item in items:
                law_id = _xml_text(item, "법령일련번호")
                if law_id and law_id not in seen:
                    seen.add(law_id)
                    new_items.append(
                        (
                            law_id,
                            _xml_text(item, "법령명한글"),
                            _xml_text(item, "공포일자"),
                        )
                    )

            if new_items:
                results = await asyncio.gather(
                    *[
                        _fetch_law_detail(client, oc, lid, lname, pdate, sem)
                        for lid, lname, pdate in new_items
                    ]
                )
                docs.extend(r for r in results if r is not None)
                log.info(
                    "법령 키워드 완료",
                    query=query,
                    fetched=len(new_items),
                    added=sum(1 for r in results if r is not None),
                    total_so_far=len(docs),
                )
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
