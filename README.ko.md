# skill_rag

[English](README.md) | **한국어**

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

### 1) 코드 + 부트스트랩 스킬

```bash
git clone <repo-url>
cd skill_rag
uv sync
bash scripts/install.sh
```

`install.sh`가 하는 일:
1. `~/.skills/` 디렉토리 생성
2. 부트스트랩 메타-스킬 `~/.skills/using-skill-rag/` 설치
3. 각 하네스(`~/.claude/skills/`, `~/.codex/skills/`)에 심볼릭 링크
4. 아래 MCP 등록 가이드를 콘솔에 출력

### 2) MCP 서버 등록

리포 경로 (`$REPO`)는 `pwd` 결과로 치환. 모든 하네스에서 MCP 실행 커맨드는 동일:

```
uv --directory $REPO run skill-rag mcp
```

#### Claude Code

CLI로 한 줄 등록 (사용자 스코프, 전역):

```bash
claude mcp add skill-rag --scope user -- uv --directory "$(pwd)" run skill-rag mcp
```

또는 `~/.claude.json`의 `mcpServers`에 직접 추가:

```json
{
  "mcpServers": {
    "skill-rag": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/skill_rag", "run", "skill-rag", "mcp"]
    }
  }
}
```

#### Codex

`~/.codex/config.toml`에 추가:

```toml
[mcp_servers.skill-rag]
command = "uv"
args = ["--directory", "/absolute/path/to/skill_rag", "run", "skill-rag", "mcp"]
```

#### 기타 MCP 호환 클라이언트 (Cursor, Windsurf 등)

각 클라이언트의 MCP 설정에 동일한 `command` / `args`를 등록.

### 3) 재시작 + 동작 확인

하네스를 재시작한 뒤 새 세션에서:

- 시작 시 `using-skill-rag` 메타-스킬이 자동 로드되는지 확인
- 아무 메시지에서나 `mcp__skill-rag__search_skills` 도구가 보이는지 확인
- 직접 호출해 보기: `search_skills(query="...", k=5)` → `{status: "ok"|"no_match", hits, ...}`

CLI에서도 같은 검색이 동작하는지 빠르게 확인:

```bash
uv run skill-rag query "deploy to vercel"
```

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
| `uv run skill-rag eval` | 공개 fixture 평가셋으로 recall@5 측정 |
| `uv run skill-rag reset` | 인덱스 초기화 |
| `uv run skill-rag mcp` | MCP 서버 실행 |

## 환경 변수

| 변수 | 기본 | 설명 |
| --- | --- | --- |
| `SKILL_RAG_CORPUS_PATH` | `~/.skills` | corpus 경로 |
| `SKILL_RAG_INDEX_PATH` | `./var/index.lance` | LanceDB 경로 |
| `SKILL_RAG_MODEL` | `all-MiniLM-L6-v2` | 임베딩 모델 |
| `SKILL_RAG_LOCAL_FILES_ONLY` | `1` | 로컬 캐시에서만 임베딩 모델 로드 |
| `SKILL_RAG_SCORE_THRESHOLD` | `0.25` | 매칭 임계값 (eval 셋 기준 calibration) |
| `SKILL_RAG_SYNC_TTL` | `30` | sync 캐시 TTL (초) |

`skill-rag eval`은 기본적으로 `eval/fixtures/` 아래의 공개 fixture를 사용하므로
GitHub에서 받은 사용자도 같은 기준으로 검증할 수 있음. 개인 코퍼스를 점검하려면
경로를 명시:

```bash
uv run skill-rag eval --corpus ~/.skills --dataset eval/queries.jsonl
```

## 문서

- `AGENTS.md` — 에이전트가 첫 작업 전 읽을 순서
- `ARCHITECTURE.md` — 모듈 구조
- `docs/product-specs/skill-rag.md` — 무엇을, 왜
- `docs/design-docs/` — 설계 결정 로그
- `docs/superpowers/specs/` — 기능별 설계 스펙
