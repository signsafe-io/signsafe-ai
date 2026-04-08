from crawler import parse_case, parse_law, search_cases, search_laws
from llm import structure_case
from vector_store import save_case

BOOTSTRAP_QUERIES = ["약관", "불공정계약", "계약해지", "손해배상", "위약금", "기밀유지", "지식재산권"]
BOOTSTRAP_LIMIT = 5  # 쿼리당 최대 수


def bootstrap() -> None:
    seen_cases: set[str] = set()
    seen_laws: set[str] = set()

    print("판례 수집 시작...")
    for query in BOOTSTRAP_QUERIES:
        for seq in search_cases(query, display=BOOTSTRAP_LIMIT):
            if seq in seen_cases:
                continue
            seen_cases.add(seq)
            try:
                case = parse_case(seq)
                structured = structure_case(case["content"])
                save_case(case, structured)
                print(f"  [판례] [{case['court']}] {case['title'] or seq}")
            except Exception as e:
                print(f"  [판례] 실패 ({seq}): {e}")

    print("법령 수집 시작...")
    for query in BOOTSTRAP_QUERIES:
        for law_id in search_laws(query, display=BOOTSTRAP_LIMIT):
            if law_id in seen_laws:
                continue
            seen_laws.add(law_id)
            try:
                law = parse_law(law_id)
                if not law["content"].strip():
                    continue
                structured = structure_case(law["content"])
                save_case(law, structured)
                print(f"  [법령] {law['title'] or law_id}")
            except Exception as e:
                print(f"  [법령] 실패 ({law_id}): {e}")

    print(f"초기 데이터 완료 — 판례 {len(seen_cases)}건, 법령 {len(seen_laws)}건")
