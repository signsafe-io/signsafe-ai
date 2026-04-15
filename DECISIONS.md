# Architecture Decision Records

## ADR-001: 조항 분절 전략 (Clause Segmentation Strategy)

**날짜**: 2026-03-27

**결정**:
한국어 계약서 특성에 맞춘 규칙 기반 분절 전략을 채택합니다.

**배경**:
계약서 조항은 단순한 문단 단위가 아닌 법적 의미 단위로 분절되어야 합니다.
한국 계약서는 `제N조`, `제N항` 같은 명시적 조항 번호를 사용하는 경우가 많습니다.

**분절 규칙** (우선순위 순):

1. **한국어 조항 헤더 패턴** (최우선):
   - `제N조`, `제N항`, `제N목`, `제N절` (아라비아 숫자 및 한글 숫자)
   - 예: `제1조 (목적)`, `제이조`, `제3항`

2. **순번 패턴**:
   - `숫자.` + 공백 + 한글/영문 시작: `1. 계약의 목적`
   - `숫자)` + 공백: `1) 갑은...`
   - `(숫자)` + 공백: `(1) 본 계약은...`
   - 원문자: `①`, `②`, ...`⑳`

3. **영문 헤더**:
   - `Article N`, `Section N`, `Clause N`

4. **폴백**: 헤더가 없으면 빈 줄 기준으로 단락 분절 후 병합

**크기 제한**:
- 최소 30자 (너무 짧은 조항은 이전 조항에 병합)
- 최대 3000자 (초과 시 강제 분할)

**이유**:
- 규칙 기반 접근은 LLM 호출 없이 빠르고 결정적(deterministic)
- 한국 계약서 패턴에 특화
- 페이지/좌표 정보를 유지하여 UI 하이라이트 지원 가능

---

## ADR-002: 임베딩 모델 선택

**날짜**: 2026-03-27

**결정**: OpenAI `text-embedding-3-small` (1536차원)

**이유**:
- 비용 효율적 ($0.02/1M 토큰)
- 한국어 지원 품질이 충분
- Qdrant 기본 설정과 호환되는 1536 차원
- 배치 처리(최대 100개) 지원으로 처리량 최적화

**대안 검토**:
- `text-embedding-3-large` (3072차원): 더 높은 정확도이나 2배 비용
- 로컬 모델 (KR-SBERT): 추가 인프라 불필요, 그러나 품질 불확실

---

## ADR-003: LLM 모델 선택

**날짜**: 2026-03-27
**수정**: 2026-04-15 (ADR-003-rev1 — 실제 구현 기준으로 정정)

**결정**: OpenAI `gpt-4o`

**이유**:
- 한국어 법률 문서 이해 및 생성 품질 우수
- 구조화된 JSON 출력 일관성 높음 (response_format=json_object 지원)
- 비용 대비 성능 최적
- 128K 컨텍스트로 긴 조항도 처리 가능

**대안 검토**:
- Anthropic claude-3-5-sonnet: 유사한 품질, 동등한 비용 (초기 설계 시 검토)
- gpt-4-turbo: gpt-4o 대비 느리고 비용 높음

**정정 이유**:
- 초기 ADR-003은 Anthropic Claude로 기재되었으나 실제 구현(`app/services/llm.py`,
  `app/workers/analysis.py`)에서는 OpenAI GPT-4o와 `openai.APIStatusError`를 사용.
- 코드와 문서의 불일치를 해소하기 위해 ADR을 실제 구현 기준으로 정정함.

---

## ADR-004: Qdrant 컬렉션 스키마

**날짜**: 2026-03-27

**결정**: 단일 컬렉션 `clauses`, cosine similarity

**페이로드 필드**:
```json
{
  "clause_id":     "string (DB FK)",
  "contract_id":   "string",
  "label":         "string | null",
  "org_id":        "string | null",
  "created_at":    "ISO 8601 string",
  "created_at_ts": "float (unix timestamp, for range filter)"
}
```

**포인트 ID**: clause_id → UUID5(OID namespace, clause_id)로 변환
(Qdrant는 UUID 또는 unsigned int만 ID로 허용)

**필터 지원**:
- `org_id` (exact match)
- `created_at_ts` (range)

---

## ADR-005: 동시성 제어

**날짜**: 2026-03-27

**결정**: Analysis 워커에서 `asyncio.Semaphore(5)` 사용

**이유**:
- LLM API rate limit 준수 (Claude: 50 RPM tier1)
- 동시 임베딩 요청 과부하 방지
- DB 연결 풀 과부하 방지 (pool max_size=10)

---

## ADR-006: 에러 처리 및 DLQ

**날짜**: 2026-03-27

**결정**: `aio-pika`의 `message.process(requeue=False)` 사용

**동작**:
1. 처리 성공 → ack (메시지 삭제)
2. 처리 실패 → nack (requeue=False) → DLQ(`*.dlq`)로 자동 라우팅
3. DB에 `failed` 상태와 에러 메시지 기록
4. 워커는 재시작 후 DLQ 메시지를 재처리하지 않음 (수동 확인 필요)

**DLQ 큐 이름**:
- `ingestion.jobs.dlq`
- `analysis.jobs.dlq`

---

## ADR-007: LLM 응답 confidence score 추가

**날짜**: 2026-04-01

**결정**: LLM 프롬프트에 `confidence: 0.0~1.0` 필드를 요청하고, `ClauseAnalysisResult`에 포함하여 DB에 저장한다.

**이유**:
- risk_level만으로는 판단의 확실성을 알 수 없음
- UI에서 "낮은 신뢰도" 조항을 다르게 표시하거나 사용자 검토 유도 가능
- 프롬프트 수정만으로 추가 가능하며 LLM 호출 횟수 증가 없음

**설계**:
- 파싱 실패 또는 범위 초과 시 `_normalize_confidence()`가 0.5로 클램핑
- DB 컬럼: `clause_results.confidence FLOAT NOT NULL DEFAULT 0.5` (migration 000006)
- 이전에 저장된 행은 DEFAULT 0.5로 초기화됨

**영향**: signsafe-api 마이그레이션 필요 (000006 추가, 완료)

---

## ADR-008: 문서 전체 요약 및 RETRIEVE_EVIDENCE 구현

**날짜**: 2026-04-07

**결정**:
1. 조항 분석 완료 후 LLM을 한 번 더 호출하여 문서 수준 리스크 요약 생성
2. `RETRIEVE_EVIDENCE` 메시지를 no-op에서 실제 RAG 재조회로 교체

**배경**:
- R-001 요구사항: "문서 아래 문서 요약" — 조항별 분석만으로는 전체 리스크 파악 어려움
- R-003 요구사항: RAG 근거 패널에서 출처 URL 클릭 네비게이션 필요

**문서 요약 설계**:
- `llm.summarize_document(clause_results)` — 성공한 조항 결과 목록을 받아 전체 요약 생성
- 출력: `overall_risk`, `summary`, `key_issues`
- DB 저장: `risk_analyses.document_summary`, `overall_risk`, `key_issues` (JSONB)
- 요약 실패는 non-fatal — 로그 후 continue, 분석 완료 상태 유지

**RETRIEVE_EVIDENCE 설계**:
- `evidenceSetId` → DB에서 연결된 조항 텍스트 조회
- RAG 재조회 → 유사 조항 top-k 검색
- 각 citation에 `/contracts/{contractId}/clauses/{clauseId}` URL 포함
- DB `evidence_sets.citations` 및 `retrieved_at` 갱신

**영향**:
- signsafe-api 마이그레이션 필요:
  ```sql
  ALTER TABLE risk_analyses
    ADD COLUMN document_summary TEXT,
    ADD COLUMN overall_risk VARCHAR(10),
    ADD COLUMN key_issues JSONB;
  ```

---

## ADR-009: 법령 크롤링 section=all 파라미터 제거

**날짜**: 2026-04-15

**결정**: `_crawl_laws()`에서 `section=all` 파라미터를 제거하고, 검색 쿼리를 법령명에 특화된 키워드로 교체한다.

**배경**:
배포 로그에서 `laws=0`이 반복됨. 국가법령정보 DRF API (`target=law`)에서
`section=all`을 추가하면 빈 결과(`totalCnt=0`, `law` 키 없음)를 반환한다.
로컬 테스트에서 `section=all` 없이 동일 쿼리를 보내면 정상 결과가 반환됨을 확인.

**변경 내용**:
1. `section=all` 파라미터 제거 — `target=law`의 기본 동작이 이미 법령명+본문 검색
2. 검색 쿼리를 법령명에 가까운 구체적인 키워드로 교체
   (예: `약관` → `약관규제법`, `개인정보` → `개인정보보호법`)
3. 검색 응답에 `totalCnt`/`law` 키 존재 여부 debug 로깅 추가 (향후 디버깅 용이)
4. 상세 조회 응답의 조문 키 탐색 순서 강화: `조문` → `법령본문` 순서로 폴백
5. 조문 내 필드 폴백: `조문제목` 없으면 `조문번호`, `조문내용` 없으면 `조문` 키 사용
6. 조문이 전혀 없을 때 법령명으로 최소 컨텐츠 구성하여 Qdrant 인덱싱 유지

**이유**:
- 원인이 단일 파라미터(`section=all`)로 명확히 확인됨
- 구체적인 법령명 키워드가 관련성 높은 법령을 더 잘 반환함

---

## ADR-010: 법령 API type=JSON→XML 전환 및 MST 파라미터 사용

**날짜**: 2026-04-15

**결정**:
1. `_crawl_laws()` 및 `_crawl_cases()` 모두 `type=XML` 사용으로 전환
2. 법령 상세 조회 파라미터를 `ID=법령일련번호` → `MST=법령일련번호`로 변경
3. JSON 파싱 로직을 `xml.etree.ElementTree` 기반으로 전면 교체

**배경**:
ADR-009 이후에도 배포 로그에서 `top_keys=[]`가 지속됨.
직접 API 호출 테스트 결과:
- `lawSearch.do?type=JSON` → 빈 `{}` 반환 (검색, 상세 모두)
- `lawSearch.do?type=XML` → 정상 XML 데이터 반환
- `lawService.do?ID=법령일련번호` → `{"Law": "일치하는 법령이 없습니다."}` 반환
- `lawService.do?MST=법령일련번호` → 정상 XML 응답 반환

**XML 응답 구조 (법령 상세)**:
```xml
<법령>
  <기본정보>
    <법령명_한글>...</법령명_한글>
    <공포일자>...</공포일자>
  </기본정보>
  <조문>
    <조문단위>
      <조문번호>1</조문번호>
      <조문제목>목적</조문제목>
      <조문내용>제1조(목적) ...</조문내용>
      <항>
        <항번호>①</항번호>
        <항내용>① ...</항내용>
        <호><호번호>1.</호번호><호내용>1. ...</호내용></호>
      </항>
    </조문단위>
  </조문>
</법령>
```

**변경 내용**:
1. `_fetch()` → `_fetch_xml()` — `type=XML` 고정, `ET.fromstring()` 반환
2. `_xml_text()` 헬퍼 — 안전한 XML 자식 요소 텍스트 추출
3. `_extract_law_content()` — XML 트리에서 조문번호/제목/내용/항/호 텍스트 추출
4. `_crawl_cases()` — XML 파싱으로 전환 (판례 상세: `ID=판례일련번호` 유지)
5. `_crawl_laws()` — XML 파싱으로 전환, `MST=법령일련번호` 사용
6. 법령명 폴백 로직 유지 (조문 없을 경우 법령명만 저장)

**이유**:
- law.go.kr DRF API의 JSON 응답이 환경 또는 서버 설정 문제로 빈 `{}`를 반환함
- XML은 동일 엔드포인트에서 정상적으로 구조화된 데이터를 반환함
- `xml.etree.ElementTree`는 Python 표준 라이브러리로 추가 의존성 없음
