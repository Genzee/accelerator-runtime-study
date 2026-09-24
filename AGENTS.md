# Agent Instructions (Codex / Claude Code 공통)

이 repo는 heterogeneous accelerator runtime 학습 repo다. 전체 계획은 `ROADMAP.md`.

## 매 세션 작업 절차

1. `ROADMAP.md`, `PROGRESS.md`, `QUESTIONS.md` 읽기.
2. 현재 phase에서 아직 완료되지 않은 **가장 작은 task 하나** 선택.
3. 필요한 개념을 먼저 설명 (시스템 프로그래머 관점, 수학 증명보다 실행 관점).
4. 작은 실험/코드 수행 (`experiments/NN-*/`).
5. 결과를 수치와 함께 기록. **[계산] / [측정] / [추측]을 구분**해서 쓴다.
6. 실패/의문점을 숨기지 않고 `QUESTIONS.md`에 저장.
7. 완료 기준을 만족하면 `PROGRESS.md` checklist 업데이트.
8. 다음 task **하나만** 제안.

## 작업 종료 시 반드시 업데이트

- `PROGRESS.md`
- 해당 `notes/*.md`
- 새 질문이 있으면 `QUESTIONS.md`
- 설계 선택이 생기면 `DECISIONS.md`
- 새 용어가 나오면 `glossary/glossary.md`

## 환경

- Apple M4, 24 GB unified memory, macOS
- Python: `.venv/` (프로젝트 전용). 실행은 `.venv/bin/python ...`
- 의존성 추가 시 `requirements.txt`에 기록.

## 금지

- 첫 10개 task(ROADMAP 참조)가 끝나기 전에 custom scheduler 구현 시작 금지.
- MLIR/IREE 소스 파기, 자체 IR/compiler/framework 제작은 Phase 8 이후 판단.
