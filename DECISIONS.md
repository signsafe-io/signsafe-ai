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

**결정**: Anthropic `claude-3-5-sonnet-20241022`

**이유**:
- 한국어 법률 문서 이해 및 생성 품질 우수
- 구조화된 JSON 출력 일관성 높음
- 비용 대비 성능 최적 (claude-3-opus 대비 저렴하고 빠름)
- 128K 컨텍스트로 긴 조항도 처리 가능

**대안 검토**:
- GPT-4o: 유사한 품질, 동등한 비용
- claude-3-opus: 최고 품질이나 3-5배 비용 증가

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

## ADR-008: update_risk_analysis_summary 컬럼 부재 시 graceful skip

**날짜**: 2026-04-14

**결정**: `update_risk_analysis_summary()`에서 `asyncpg.exceptions.UndefinedColumnError` 발생 시 경고 로그만 남기고 skip한다. 에러를 전파하지 않으므로 분석 완료 흐름에 영향 없다.

**배경**:
- `risk_analyses` 테이블에 `document_summary`, `overall_risk`, `key_issues` 컬럼 추가는 signsafe-api 측 마이그레이션으로 처리
- 마이그레이션 전까지 해당 컬럼이 없어 호출마다 DB 에러 발생
- 기존 코드에서 `analysis.py`의 `try/except Exception`으로 잡혀 비치명적이었으나, 의도가 명확하지 않았음

**구현**:
- `db.py`의 `update_risk_analysis_summary` 내부에서 `UndefinedColumnError`를 직접 잡아 처리
- 마이그레이션 완료 후 동일 코드가 자동으로 정상 동작 (코드 변경 불필요)

**대안 검토**:
- `analysis.py`의 `try/except` 유지: 의도가 불명확하고 모든 예외를 삼킴
- 컬럼 존재 여부 사전 체크: 매 호출마다 `information_schema` 조회 → 오버헤드

---

## ADR-009: anthropic_api_key 설정 필드 제거

**날짜**: 2026-04-14

**결정**: `config.py`에서 `anthropic_api_key` 필드 제거

**이유**:
- ADR-003에서 LLM을 OpenAI gpt-4o로 전환하여 Anthropic SDK 미사용
- 미사용 환경변수를 선언하면 운영자 혼란 유발
- xquare Vault에서도 해당 키 주입 불필요
