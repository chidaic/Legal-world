# LEGALWORLD Core Backend

This repository contains the backend core for **LEGALWORLD: A Life-Cycle Interactive Environment for Legal Agents**.

LEGALWORLD models civil litigation as a connected life-cycle process: consultation, document drafting, first-instance trial, appeal, and second-instance proceedings. The backend provides the agent runtime, scenario orchestration, legal Tool/Skill interfaces, WebSocket protocol, API routes, and extension points needed to run or extend the research system.

The public repository is intentionally backend-only. It does **not** include frontend source code, runtime result data, raw evaluation outputs, paper drafts, private deployment files, or model/API credentials. A hosted demo page can be linked separately from the project page.

## What Is Included

- `backend/ws_server.py`: FastAPI and WebSocket application entry point.
- `backend/src/agents/`: client, lawyer, judge, and receptionist agent wrappers.
- `backend/src/scenarios/`: consultation, drafting, court, and appeal scenario logic.
- `backend/src/pipeline/`: life-cycle pipeline and Tool/Skill stage resolution.
- `backend/src/tools/`: legal drafting, retrieval, citation checking, memory, and artifact tools.
- `backend/legal-skillhub/public/`: public Skill instructions used by the legal-agent workflow.
- `backend/gitskill/`: reflective Skill management and growth utilities.
- `examples/status_client.py`: minimal example script for checking a running backend.
- `docs/`: static LEGALWORLD project page for GitHub Pages.

## What Is Not Included

- No frontend application source.
- No generated case trajectories, logs, batch runs, or benchmark results.
- No raw legal judgment corpus or private evaluation materials.
- No law-retrieval vector index files; these are large data assets released separately.
- No internal test files or private development checks.
- No `.env`, API keys, database passwords, model credentials, or deployment secrets.

## Requirements

- Python 3.10 or 3.11.
- PostgreSQL 16 for the full API/runtime service.
- Docker and Docker Compose are recommended for the database-backed backend.
- Model provider credentials for model-backed simulations.

## Quick Start

Clone the repository and create a Python environment:

```bash
git clone https://github.com/chidaic/Legal-world.git
cd Legal-world
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Create local configuration:

```bash
# Windows PowerShell
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Edit `.env` and set at least:

```env
OPENAI_API_KEY=
OPENAI_API_BASE_URL=
OPENAI_MODEL_NAME=
SIMLAW_ENABLE_LAW_RETRIEVAL=false
LAW_RETRIEVAL_INDEX_DIR=
DATABASE_URL=postgresql+psycopg://simlaw:change-this-postgres-password@localhost:5432/simlaw
JWT_SECRET=change-this-jwt-secret
```

Use your own model provider endpoint and credentials. Do not commit `.env`.

## Run With Docker Compose

The easiest full local setup is Docker Compose, which starts PostgreSQL and the backend service together:

```bash
docker compose -f backend/docker-compose.yml up --build
```

After startup, check:

```text
http://127.0.0.1:8000/api/status
```

Stop the stack:

```bash
docker compose -f backend/docker-compose.yml down
```

## Run Backend Directly

If PostgreSQL is already available and `DATABASE_URL` points to it, run:

```bash
python start.py
```

This starts only the backend:

```text
Backend API: http://127.0.0.1:8000
WebSocket:   ws://127.0.0.1:8000/ws
```

You can also run Uvicorn directly:

```bash
cd backend
python -m uvicorn ws_server:app --host 127.0.0.1 --port 8000
```

## Run The Example Script

Once the backend is running, use the example client to verify the service:

```bash
python examples/status_client.py --base-url http://127.0.0.1:8000
```

Expected output is a JSON object similar to:

```json
{
  "status": "running",
  "backend_version": "..."
}
```

This example does not run a case and does not call a model provider. It only checks that the backend API is reachable.

## Skill Library

The public legal Skill library is included under:

```text
backend/legal-skillhub/public/
```

The runtime uses this folder by default. To override it, set:

```env
SIMLAW_MAIN_SKILL_ROOT=/path/to/your/skillhub/public
```

Reflective Skill growth utilities are included under `backend/gitskill/`. They operate on generated case-run outputs, which are not part of this public repository. To run the single-case reflection example, first point it at your own generated case directory:

```bash
SIMLAW_SKILL_GROWTH_CASE_DIR=backend/batch_runs/<your_case_run> python backend/gitskill/run_single_case_skill_growth.py
```

On Windows PowerShell:

```powershell
$env:SIMLAW_SKILL_GROWTH_CASE_DIR="backend/batch_runs/<your_case_run>"
python backend/gitskill/run_single_case_skill_growth.py
```

## Optional Law Retrieval Data

Semantic law-article retrieval is disabled by default in this public repository. The retrieval tool requires a prebuilt local vector index, including:

```text
law_vector_index_manifest.json
law_embeddings.float16.npy
law_metadata.jsonl
```

These files are intentionally not stored in GitHub because they are large data assets. After the project Data page is public, download the law-retrieval index package from the Data page or the linked Hugging Face dataset, unpack it locally, and then enable retrieval:

```env
SIMLAW_ENABLE_LAW_RETRIEVAL=true
LAW_RETRIEVAL_INDEX_DIR=/path/to/cn_law
LAW_EMBEDDING_API_KEY=
LAW_EMBEDDING_API_BASE_URL=
LAW_EMBEDDING_MODEL=
LAW_EMBEDDING_DIMENSIONS=1024
```

`LAW_RETRIEVAL_INDEX_DIR` should point to the directory that contains the three files listed above. The query embedding model should match the model used to build the downloaded index; check the downloaded `law_vector_index_manifest.json` for the model hint.

## Project Page

The static project page is stored in `docs/` so GitHub Pages can serve it from the same repository.

After pushing the repository, open GitHub repository settings and set:

```text
Settings -> Pages -> Build and deployment
Source: Deploy from a branch
Branch: main
Folder: /docs
```

The expected page URL is:

```text
https://chidaic.github.io/Legal-world/
```

## Repository Hygiene

Before publishing or making a release:

- Confirm `.env` is not tracked.
- Confirm no generated data exists under `backend/sandbox_data/`, `backend/batch_runs/`, or debug output folders.
- Confirm law-retrieval index files are not committed; publish them through the project Data page instead.
- Confirm no frontend source directory is present.
- Confirm public Skills under `backend/legal-skillhub/public/` contain only reusable procedural guidance.
- Confirm no private IPs, API keys, access tokens, or local absolute paths are present.
- Add a project license before public release.

## Citation

Citation information will be added after the arXiv release.
