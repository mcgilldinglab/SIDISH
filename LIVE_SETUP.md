# Running SIDISH Agent with live data

## 1. Configure approved assets

Copy `.env.example` to `.env` and point each known cancer to its locked
manuscript bulk-survival table and, when available, its trained reference run:

```bash
SIDISH_BREAST_BULK=/approved/BREAST_CANCER/bulk_result.csv
SIDISH_BREAST_RUN=/approved/BREAST_CANCER
SIDISH_LUNG_BULK=/approved/LUNG_CANCER/bulk_result.csv
SIDISH_LUNG_RUN=/approved/LUNG_CANCER
SIDISH_PANCREATIC_BULK=/approved/PANCREAS_CANCER/bulk_result.csv
SIDISH_PANCREATIC_RUN=/approved/PANCREAS_CANCER
SIDISH_TRAINING_DEVICE=cuda:1
```

The bulk table must start with `duration,event`, followed by gene-expression
columns. Known cancers always use their configured manuscript bulk. For another
disease, upload a matched bulk RNA-seq/survival CSV in the same format.

## 2. Start the local workspace

```bash
export SIDISH_APP_PASSWORD='use-a-secret-from-your-secret-manager'
streamlit run app.py --server.address 127.0.0.1
```

Create a pseudonymous case, upload a preprocessed `.h5ad`, and choose the cancer
reference. After validation, choose the `AnnData.obs` patient/sample column and
one identifier from that column. Open **Chat with SIDISH** and ask the first
analysis question; if training is required, SIDISH explains why and asks for
confirmation before submitting the durable training task. Raw cell counts and
other read-only case facts do not trigger training. Once trained, chat questions run only the needed
patient-level or cohort-level capability. Ask **generate the final report** to
complete missing evidence and expose HTML/PDF downloads below the conversation.
On a Mac without CUDA, training falls back to CPU and can be much slower. For
routine new-dataset training, run the app/job worker where `SIDISH_TRAINING_DEVICE`
points to an available CUDA device. Patient questions reuse the loaded model and
cached evidence after the first cold load.

## 3. Chat and report behavior

The chat offers on-demand tools for high-risk cells, marker/pathway context, target
perturbation, whole-pathway perturbation, drug mapping, and report generation.
Named genes, pathways, drug targets, mapping methods, and configured compounds are
passed to the scientific function and validated against the active model. Invalid
entities return close matches instead of silently running a default sweep. Target
and pathway perturbations return an inline transition UMAP and statistics table.
Every answer states the evidence scope. The language model, when enabled, can
choose tools and word the response, but it cannot create measurements or modify
the authoritative case result.

Set `SIDISH_LLM_PROVIDER=auto` to use Gemini when `GEMINI_API_KEY` is present,
OpenAI when `OPENAI_API_KEY` is present, an explicitly configured compatible
endpoint, or local Ollama as the no-key fallback. If the provider is unavailable,
the validated local entity parser remains available as a safe fallback.

The report is deterministic by default. Values and tables remain locked to the
case result and conversation wording never changes computed evidence.

## 4. Perturbation semantics

- `perturbation`: target plus high-confidence PPI-neighbour expression ablation.
  It is not a literal single-gene edit.
- `pathway_perturbation`: ablates the modeled genes in an explicit pathway or
  pathway sweep and measures change in model-labelled high-risk cells.
- `drug_perturbation`: maps targets or a reverse signature to candidate
  perturbagens. These are hypotheses, not treatment recommendations.

Use patient-specific perturbation only when the sample exists in the trained
single-cell run. Genome-wide and pathway sweeps may be long-running jobs.

## 5. Operational safeguards

- Use pseudonymous IDs; do not place direct identifiers in chat.
- Keep clinician mode fail-closed (`SIDISH_DEMO_MODE=0`).
- Restrict app binding, configure authentication, and use encrypted institutional
  storage before handling real health information.
- Review job logs, QC, hashes, model version, evidence scope, and limitations.
- Require both scientific and clinical sign-off before releasing a report.
- Do not use the software for autonomous diagnosis or treatment selection.

## 6. Troubleshooting

- Missing manuscript data: set the cancer-specific environment paths; do not
  substitute an unrelated cohort.
- Other disease rejected: supply a complete matched bulk-survival CSV.
- Perturbation unavailable: set `SIDISH_PPI_DIR` to HIPPIE and STRING files.
- CUDA unavailable: analysis can use CPU; schedule training on a GPU system.
- PDF unavailable: install the declared PDF dependencies and browser runtime.
