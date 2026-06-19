# LEGALWORLD

This repository contains the public open-source code release for **LEGALWORLD: A Life-Cycle Interactive Environment for Legal Agents**.

Project page: [https://chidaic.github.io/Legal-world/](https://chidaic.github.io/Legal-world/)

Paper: [https://arxiv.org/abs/2606.18728](https://arxiv.org/abs/2606.18728)

Demo: [http://www.fudan-disc.com/legalworld/](http://www.fudan-disc.com/legalworld/)

Dataset: [https://huggingface.co/datasets/Chidaic/legal-world](https://huggingface.co/datasets/Chidaic/legal-world)

LEGALWORLD models civil litigation as a connected life-cycle process: consultation, document drafting, first-instance trial, appeal, and second-instance proceedings. The released code provides the agent runtime, scenario orchestration, legal Tool/Skill interfaces, WebSocket protocol, API routes, dataset-construction utilities, and extension points needed to run or extend the research system.

This public release focuses on reproducible code and reusable components. It does **not** include runtime result data, raw evaluation outputs, paper drafts, private deployment files, or model/API credentials. The hosted demo and dataset are linked above.

## What Is Included

- `backend/ws_server.py`: FastAPI and WebSocket application entry point.
- `backend/src/agents/`: client, lawyer, judge, and receptionist agent wrappers.
- `backend/src/scenarios/`: consultation, drafting, court, and appeal scenario logic.
- `backend/src/pipeline/`: life-cycle pipeline and Tool/Skill stage resolution.
- `backend/src/tools/`: legal drafting, retrieval, citation checking, memory, and artifact tools.
- `backend/legal-skillhub/public/`: public Skill instructions used by the legal-agent workflow.
- `backend/gitskill/`: reflective Skill management and growth utilities.
- `dataset_builder/`: code for constructing the public case dataset from matched first- and second-instance raw cases.
- `examples/status_client.py`: minimal example script for checking a running service.
- `docs/`: static LEGALWORLD project page for GitHub Pages.

## What Is Not Included

- No generated case trajectories, logs, run outputs, batch scripts, or benchmark results.
- No raw legal judgment corpus or private evaluation materials.
- No law-retrieval vector index files; these are large data assets released separately.
- No internal test files or private development checks.
- No `.env`, API keys, database passwords, model credentials, or deployment secrets.

## Requirements

- Python 3.10 or 3.11.
- PostgreSQL 16 for the full API/runtime service.
- Docker and Docker Compose are recommended for the database-backed local service.
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

The easiest full local setup is Docker Compose, which starts PostgreSQL and the LEGALWORLD service together:

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

## Run Service Directly

If PostgreSQL is already available and `DATABASE_URL` points to it, run:

```bash
python start.py
```

This starts the local API and WebSocket service:

```text
API:       http://127.0.0.1:8000
WebSocket: ws://127.0.0.1:8000/ws
```

You can also run Uvicorn directly:

```bash
cd backend
python -m uvicorn ws_server:app --host 127.0.0.1 --port 8000
```

## Run The Example Script

Once the service is running, use the example client to verify it:

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
SIMLAW_SKILL_GROWTH_CASE_DIR=backend/case_runs/<your_case_run> python backend/gitskill/run_single_case_skill_growth.py
```

On Windows PowerShell:

```powershell
$env:SIMLAW_SKILL_GROWTH_CASE_DIR="backend/case_runs/<your_case_run>"
python backend/gitskill/run_single_case_skill_growth.py
```

## Dataset Construction Utilities

The `dataset_builder/` folder contains the open-source case dataset construction pipeline. It converts matched first- and second-instance raw JSON cases into structured case data, adds legal persona profiles, and generates consultation questions with reference answers.

Install the lightweight builder dependencies:

```bash
pip install -r dataset_builder/requirements.txt
```

Configure an OpenAI-compatible model endpoint:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="your_api_base_url"
export OPENAI_MODEL="your_model_name"
```

Run the builder from the `dataset_builder/` folder:

```bash
cd dataset_builder
python -m case_dataset_builder.pipeline <input_raw_json> --output-dir <output_dir>
```

The public Hugging Face dataset already provides released case resources. This builder is included for reproducibility and customization; generated outputs should stay outside the GitHub repository.

## Dataset And Optional Law Retrieval Data

The public dataset is hosted on Hugging Face:

```text
https://huggingface.co/datasets/Chidaic/legal-world
```

It includes raw and processed case data plus `law_metadata.jsonl` for legal-provision retrieval. Semantic law-article retrieval is disabled by default in this code release. To enable retrieval, download the law metadata, generate embeddings locally or attach your own vector database, and prepare a local index directory containing:

```text
law_vector_index_manifest.json
law_embeddings.float16.npy
law_metadata.jsonl
```

These files are intentionally not stored in GitHub because they are large data assets. After preparing the local index, enable retrieval:

```env
SIMLAW_ENABLE_LAW_RETRIEVAL=true
LAW_RETRIEVAL_INDEX_DIR=/path/to/your/cn_law_index
LAW_EMBEDDING_API_KEY=
LAW_EMBEDDING_API_BASE_URL=
LAW_EMBEDDING_MODEL=
LAW_EMBEDDING_DIMENSIONS=1024
```

`LAW_RETRIEVAL_INDEX_DIR` should point to the directory that contains the three files listed above. The query embedding model should match the model used to build the local index; keep the model hint in `law_vector_index_manifest.json`.

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
- Confirm no generated data exists under `backend/sandbox_data/`, `backend/case_runs/`, `backend/batch_runs/`, or debug output folders.
- Confirm law-retrieval index files are not committed; publish them through the project Data page instead.
- Confirm public Skills under `backend/legal-skillhub/public/` contain only reusable procedural guidance.
- Confirm no private IPs, API keys, access tokens, or local absolute paths are present.
- Add a project license before public release.

## Citation

```bibtex
@misc{zuo2026legalworld,
  title={LegalWorld: A Life-Cycle Interactive Environment for Legal Agents},
  author={Songhan Zuo and Shengbin Yue and Tao Chiang and Guanying Li and Yun Song and Xuanjing Huang and Zhongyu Wei},
  year={2026},
  eprint={2606.18728},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2606.18728}
}
```
