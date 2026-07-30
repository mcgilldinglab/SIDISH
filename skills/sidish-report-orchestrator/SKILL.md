---
name: sidish-report-orchestrator
description: Orchestrate SIDISH research-use clinician decision-support workflows - validate a config, run high-risk and target/pathway/drug perturbation analysis, produce the structured report, or launch and monitor training on a new single-cell dataset.
---

# SIDISH Report Orchestrator

## Overview

Run SIDISH as a skill-first workflow. Use direct CLI calls for short tasks (preflight, analysis,
report) and the durable job orchestrator for long tasks (training on a new dataset, genome-wide
perturbation). All state lives under the repo checkout that contains `sidish_cli.py`; job state is
written under `outputs/skill_jobs/<job_id>/`.

SIDISH integrates scRNA-seq + bulk RNA-seq + survival to find high-risk cell populations, rank
target, pathway, and drug hypotheses via in-silico perturbation (single, dual, pathway,
patient-specific) and drug mapping, and produce a clinician-facing report (HTML/PDF). This is a
RESEARCH-USE decision-support tool — never present output as a treatment directive.

## Prerequisites

- One active Python environment with the SIDISH stack (torch, scanpy, pyro, lifelines, ...).
  Run `python skills/sidish-report-orchestrator/scripts/check_python_env.py` first.
- If no environment is detected, stop and ask the user to activate one, then continue.
- macOS has no CUDA: analysis runs on CPU; training should run on a GPU/lab server (`device: cuda:1`).
- Perturbation needs the PPI files (`./PPI/` or `$SIDISH_PPI_DIR`).
- A live LLM report needs an OpenAI-compatible endpoint (`OPENAI_BASE_URL`/`OPENAI_API_KEY`,
  e.g. Gemini `gemini-flash-lite-latest`); otherwise the deterministic report is used.

## Workflow Routing

- Short tasks — direct CLI (from the repo root):
  - `python sidish_cli.py preflight -c configs/sidish_workflow_template.yaml`
  - `python sidish_cli.py run -c configs/sidish_workflow_template.yaml`
  - `python sidish_cli.py report --patient CID3946 --pdf`     # config-free shortcut
- Long tasks — durable job orchestration:
  - `python skills/sidish-report-orchestrator/scripts/sidish_job.py submit --config <cfg> --job-name <name>`
  - `python skills/sidish-report-orchestrator/scripts/sidish_job.py status <job_id>`
  - `python skills/sidish-report-orchestrator/scripts/sidish_job.py list`
  - `python skills/sidish-report-orchestrator/scripts/sidish_job.py tail <job_id> --lines 120`
  - `python skills/sidish-report-orchestrator/scripts/sidish_job.py cancel <job_id>`
  - `python skills/sidish-report-orchestrator/scripts/sidish_job.py resume <job_id>`
- Monitoring — local web console:
  - `python skills/sidish-report-orchestrator/scripts/sidish_monitor_web.py serve --port 36872`

## Data-ingestion policy (before training)

- If the user's single-cell data is breast / lung / pancreatic, SIDISH must use the configured,
  locked manuscript bulk+survival for that type - no user bulk upload is used. Set
  `data.cancer_type` and the matching `SIDISH_*_BULK` environment variable.
- For any other disease, the user MUST provide `data.bulk_csv` in SIDISH format:
  columns `duration, event, <gene1>, <gene2>, ...` (no index column; `event` = 0/1).
  Preflight rejects a wrong format before any heavy work.
- If the single-cell data has no trained SIDISH model yet, finding high-risk cells requires a
  TRAINING run (Phase-1 VAE + Phase-2 DeepCox + iterative training) — a long GPU/lab job. Submit it
  with the job orchestrator using `configs/sidish_training_template.yaml`, then point an analysis
  config at the resulting `run_dir`.

## Quick Start

```bash
# 1) Confirm the environment
python skills/sidish-report-orchestrator/scripts/check_python_env.py

# 2) Validate the default analysis config
python sidish_cli.py preflight -c configs/sidish_workflow_template.yaml

# 3) Produce the report (deterministic; add report.use_llm + llm.* for the AI narrative)
python sidish_cli.py run -c configs/sidish_workflow_template.yaml

# 4) Train SIDISH on a new dataset as a durable job (GPU/lab)
python skills/sidish-report-orchestrator/scripts/sidish_job.py submit \
  --config configs/sidish_training_template.yaml --job-name new-train
python skills/sidish-report-orchestrator/scripts/sidish_job.py status <job_id>

# 5) Monitor jobs + browse reports in the browser
python skills/sidish-report-orchestrator/scripts/sidish_monitor_web.py serve --port 36872
```

## Safety

All therapeutic output is a computational hypothesis for validation. Keep the research-use
disclaimer; never phrase perturbation/drug results as a prescription or a diagnosis. The LLM is
not the source of record; chat and report values must come from the structured case result.
