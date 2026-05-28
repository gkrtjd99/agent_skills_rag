# skill_rag

`~/.skills/` 에 모아둔 스킬들을 자연어로 검색해서 필요한 것만 에이전트 컨텍스트에 올리는 로컬 RAG.

세션 시작 시 메타-스킬 1개만 자동 로드되고, 나머지 N개는 매 사용자 메시지마다
MCP로 검색해서 적합한 본문만 가져옴. 따라서 처음부터 모든 스킬을 읽느라 컨텍스트
소모하지 않음.

## 핵심 동작

```
사용자 메시지
   │
   ▼
에이전트 → search_skills(query)  ─→ top-k 메타 (name, desc, score)
                                       │
                                       ▼ 적합한 것만
                                  get_skill(name) ─→ SKILL.md 본문
```

- 임베딩: `all-MiniLM-L6-v2` 로컬 모델 (외부 API 호출 없음)
- 벡터 DB: LanceDB
- 인덱스: `search_skills` 호출 시 TTL 30s 캐시로 자동 sync

## 설치

```bash
git clone <this repo>
cd skill_rag
uv sync
bash scripts/install.sh
```

`install.sh`가 하는 일:
1. `~/.skills/` 디렉토리 생성
2. 부트스트랩 메타-스킬 `~/.skills/using-skill-rag/` 설치
3. 각 하네스(`~/.claude/skills/`, `~/.codex/skills/`)에 심볼릭 링크
4. MCP 서버 등록 가이드 출력

각 하네스 설정에 MCP 서버 추가 (안내 메시지 참고) 후 재시작.

## 스킬 추가

`~/.skills/<name>/SKILL.md` 형식으로 파일 작성:

```markdown
---
name: my-skill
description: 한 줄 설명. 검색 정확도가 여기 품질에 좌우됨.
---

# 본문
스킬 사용법을 자세히 적음.
```

다음 `search_skills` 호출 시 30초 이내에 자동 인덱싱됨.

## CLI

| 명령 | 설명 |
| --- | --- |
| `uv run skill-rag sync` | 인덱스 수동 동기화 |
| `uv run skill-rag query "<text>"` | 검색 결과 확인 |
| `uv run skill-rag list-skills` | 인덱스된 스킬 목록 |
| `uv run skill-rag eval` | 평가셋으로 recall@5 측정 |
| `uv run skill-rag reset` | 인덱스 초기화 |
| `uv run skill-rag mcp` | MCP 서버 실행 |

## 환경 변수

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `SKILL_RAG_CORPUS_PATH` | `~/.skills` | corpus 경로 |
| `SKILL_RAG_INDEX_PATH` | `./var/index.lance` | LanceDB 경로 |
| `SKILL_RAG_MODEL` | `all-MiniLM-L6-v2` | 임베딩 모델 |
| `SKILL_RAG_SCORE_THRESHOLD` | `0.25` | 매칭 임계값 (eval 셋 기준 calibration) |
| `SKILL_RAG_SYNC_TTL` | `30` | sync 캐시 TTL (초) |

## 문서

- `AGENTS.md` — 에이전트가 첫 작업 전 읽을 순서
- `ARCHITECTURE.md` — 모듈 구조
- `docs/product-specs/skill-rag.md` — 무엇을, 왜
- `docs/design-docs/` — 설계 결정 로그
- `docs/superpowers/specs/` — 기능별 설계 스펙
