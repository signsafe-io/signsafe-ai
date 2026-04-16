"""Bootstrap crawler for law.go.kr DRF API.

주의: type=JSON 은 빈 {} 를 반환함 — 반드시 type=XML 사용.
법령 상세 조회는 ID 파라미터가 동작하지 않으므로 MST=법령일련번호 사용.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

BASE_URL = "https://www.law.go.kr"
API_BASE = f"{BASE_URL}/DRF/lawSearch.do"
CASE_BASE = f"{BASE_URL}/DRF/lawService.do"

_OC = os.environ["LAW_API_OC"]


def _get_xml(url: str, params: dict) -> ET.Element:
    params = {**params, "type": "XML"}
    res = requests.get(url, params=params, timeout=30)
    res.raise_for_status()
    return ET.fromstring(res.text)


def _xml_text(el: ET.Element | None, tag: str, default: str = "") -> str:
    if el is None:
        return default
    child = el.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def search_cases(query: str, display: int = 20) -> list[str]:
    """판례 검색 → precSeq 목록 반환"""
    root = _get_xml(
        API_BASE, {"OC": _OC, "target": "prec", "query": query, "display": display}
    )
    return [
        _xml_text(item, "판례일련번호")
        for item in root.findall("prec")
        if _xml_text(item, "판례일련번호")
    ]


def parse_case(prec_seq: str) -> dict:
    """판례 상세 조회"""
    root = _get_xml(CASE_BASE, {"OC": _OC, "target": "prec", "ID": prec_seq})
    d = root if root.tag == "PrecService" else root
    raw_content = _xml_text(d, "판례내용") or _xml_text(d, "전문")
    content = BeautifulSoup(raw_content, "html.parser").get_text()[:5000]
    return {
        "type": "prec",
        "source_id": prec_seq,
        "title": _xml_text(d, "사건명"),
        "case_number": _xml_text(d, "사건번호"),
        "court": _xml_text(d, "법원명"),
        "date": _xml_text(d, "선고일자"),
        "content": content,
    }


def search_laws(query: str, display: int = 20) -> list[str]:
    """법령 검색 → 법령일련번호(MST) 목록 반환"""
    root = _get_xml(
        API_BASE, {"OC": _OC, "target": "law", "query": query, "display": display}
    )
    return [
        _xml_text(item, "법령일련번호")
        for item in root.findall("law")
        if _xml_text(item, "법령일련번호")
    ]


def parse_law(law_id: str) -> dict:
    """법령 상세 조회 (MST=법령일련번호 사용)"""
    root = _get_xml(CASE_BASE, {"OC": _OC, "target": "law", "MST": law_id})

    basic = root.find("기본정보")
    law_name = _xml_text(basic, "법령명_한글")
    pub_date = _xml_text(basic, "공포일자")

    jomun_el = root.find("조문")
    content = ""
    if jomun_el is not None:
        parts = []
        for unit in jomun_el.findall("조문단위")[:20]:
            num = _xml_text(unit, "조문번호")
            title = _xml_text(unit, "조문제목")
            body = _xml_text(unit, "조문내용")
            sub = []
            for hang in unit.findall("항"):
                if t := _xml_text(hang, "항내용"):
                    sub.append(t)
            header = title or (f"제{num}조" if num else "")
            article = (body + "\n" + "\n".join(sub)).strip() if sub else body
            part = f"{header}\n{article}".strip() if header else article.strip()
            if part:
                parts.append(part)
        content = "\n\n".join(parts)[:5000]

    return {
        "type": "law",
        "source_id": law_id,
        "title": law_name,
        "content": content or law_name,
        "date": pub_date,
        "court": "",
    }
