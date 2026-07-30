# SIDISH Skill Command Cheatsheet

## Environment gate (run first)
```bash
python skills/sidish-report-orchestrator/scripts/check_python_env.py
```
If it fails, ask the user to activate a Python env with the SIDISH stack, then continue.

## Short tasks (direct CLI, from repo root)
```bash
python sidish_cli.py preflight -c configs/sidish_workflow_template.yaml   # validate config + inputs
python sidish_cli.py run       -c configs/sidish_workflow_template.yaml   # analysis -> report
python sidish_cli.py report --patient CID3946 --pdf                       # config-free report + PDF
python sidish_cli.py report --patient CID3946 --survival-only --pdf       # survival-only variant
```

## Long tasks (durable jobs)
```bash
# Train SIDISH on a NEW dataset (GPU/lab). Edit configs/sidish_training_template.yaml first.
python skills/sidish-report-orchestrator/scripts/sidish_job.py submit \
  --config configs/sidish_training_template.yaml --job-name new-train

# Durable report as a job (e.g. genome-wide perturbation)
python skills/sidish-report-orchestrator/scripts/sidish_job.py submit \
  --config configs/sidish_workflow_template.yaml --job-name cohort-report

python skills/sidish-report-orchestrator/scripts/sidish_job.py list
python skills/sidish-report-orchestrator/scripts/sidish_job.py status <job_id>
python skills/sidish-report-orchestrator/scripts/sidish_job.py tail <job_id> --lines 120
python skills/sidish-report-orchestrator/scripts/sidish_job.py cancel <job_id>
python skills/sidish-report-orchestrator/scripts/sidish_job.py resume <job_id>
```

## Monitoring web console
```bash
python skills/sidish-report-orchestrator/scripts/sidish_monitor_web.py serve --port 36872
# open http://127.0.0.1:36872 : jobs dashboard, live log tail, report viewer
```

## Config keys (analysis)
- `data.adata_path / bulk_csv / run_dir / cancer_type`
- `model.device` (cpu | cuda:1) `model.percentile`
- `analysis.patient` (null = auto top-burden), `analysis.patient_specific`
- `analysis.perturbation.scope` (markers | all | explicit), `n_genes`, `genes`
- `report.use_llm / survival_only / make_pdf`, `llm.base_url / api_key / model`

## Interactive conversational agent (separate from this skill)
```bash
python sidish_agent.py            # multi-turn LLM tool-calling chat over the SIDISH tools
```
