# AI Knowledge Assistant

> Status: 🚧 Under construction — Phase 1 of 25 (project scaffolding) complete.

A retrieval-augmented generation (RAG) chatbot that answers questions from a
specific document set, with citations, conversation memory, and honest
"I don't know" behavior instead of hallucination.

## Setup (so far)

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your real OPENAI_API_KEY
python main.py
```

## Project Structure

```
app/
├── api/         # FastAPI routes
├── core/        # config, logging, custom exceptions
├── services/    # business logic (conversation manager, RAG orchestration)
├── models/      # Pydantic schemas + DB models
├── retrieval/   # embeddings + vector DB
├── llm/         # LLM client wrappers
├── prompts/     # prompt templates
└── utils/       # helpers
```

*(Full architecture diagram, tech stack rationale, evaluation results, and
screenshots will be added as later phases are completed — see Phase 23.)*
