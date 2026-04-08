import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE_URL = "https://www.law.go.kr"
API_BASE = f"{BASE_URL}/DRF/lawSearch.do"
CASE_BASE = f"{BASE_URL}/DRF/lawService.do"

_OC = os.environ["LAW_API_OC"]


def search_cases(query: str, display: int = 10) -> list[str]:
    """판례 검색 → precSeq 목록 반환"""
    res = requests.get(API_BASE, params={
        "OC": _OC,
        "target": "prec",
        "type": "JSON",
        "query": query,
        "display": display,
    })
    res.raise_for_status()
    data = res.json()
    items = data.get("PrecSearch", {}).get("prec", [])
    if isinstance(items, dict):
        items = [items]
    return [item["판례일련번호"] for item in items if "판례일련번호" in item]


def parse_case(prec_seq: str) -> dict:
    """판례 상세 조회"""
    res = requests.get(CASE_BASE, params={
        "OC": _OC,
        "target": "prec",
        "type": "JSON",
        "ID": prec_seq,
    })
    res.raise_for_status()
    data = res.json().get("PrecService", {})

    # 본문은 HTML 포함 → 태그 제거
    raw_content = data.get("판례내용", "") or data.get("전문", "")
    content = BeautifulSoup(raw_content, "html.parser").get_text()[:5000]

    return {
        "type": "prec",
        "source_id": prec_seq,
        "title": data.get("사건명", ""),
        "case_number": data.get("사건번호", ""),
        "court": data.get("법원명", ""),
        "date": data.get("선고일자", ""),
        "content": content,
    }


def search_laws(query: str, display: int = 10) -> list[str]:
    """법령 검색 → 법령ID 목록 반환"""
    res = requests.get(API_BASE, params={
        "OC": _OC,
        "target": "law",
        "type": "JSON",
        "query": query,
        "display": display,
    })
    res.raise_for_status()
    data = res.json()
    items = data.get("LawSearch", {}).get("law", [])
    if isinstance(items, dict):
        items = [items]
    return [item["법령ID"] for item in items if "법령ID" in item]


def parse_law(law_id: str) -> dict:
    """법령 상세 조회"""
    res = requests.get(CASE_BASE, params={
        "OC": _OC,
        "target": "law",
        "type": "JSON",
        "ID": law_id,
    })
    res.raise_for_status()
    data = res.json().get("LawService", {})

    articles = data.get("조문", [])
    if isinstance(articles, dict):
        articles = [articles]
    content = "\n\n".join(
        f"{a.get('조문제목', '')}\n{a.get('조문내용', '')}".strip()
        for a in articles[:20]
        if a.get("조문제목") or a.get("조문내용")
    )[:5000]

    return {
        "type": "law",
        "source_id": law_id,
        "title": data.get("법령명", ""),
        "content": content,
        "date": data.get("공포일자", ""),
        "court": "",
    }
