# accelerator-runtime-study

AI 모델이 CPU/GPU/NPU 위에서 어떻게 실행되는지 "실행 관점"으로 공부하고,
heterogeneous accelerator node runtime / scheduler 아이디어의 타당성을 실험으로 검증하는 개인 학습 repo.

- 계획: [ROADMAP.md](ROADMAP.md)
- 진행: [PROGRESS.md](PROGRESS.md)
- 질문: [QUESTIONS.md](QUESTIONS.md)
- 설계 결정: [DECISIONS.md](DECISIONS.md)
- 용어: [glossary/glossary.md](glossary/glossary.md)
- 에이전트 작업 규칙: [AGENTS.md](AGENTS.md)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python experiments/01-matmul/matmul_basics.py
```

환경: Apple M4, 24 GB unified memory, macOS.
