# SIDISH Agent architecture and evidence contract

## Case flow

`upload -> route and validate -> select patient column/patient -> consent-gated chat training -> parameterized on-demand analysis -> inline artifacts + structured result -> chat-generated report -> review`

Each case has separate inputs, model outputs, jobs, chat history, artifacts, and
an authoritative `result.json`. Reports and chat read from that result. The LLM
is never the source of record. Scientific capabilities are incremental: a
high-risk-cell question does not automatically run perturbation or report code.
Chat turns add optional `figures` and `table` fields to `role` and `content`, so
older histories still load. Named-entity evidence is stored under both the
report-compatible canonical key and a stable entity-specific provenance key.

## Evidence levels

| Scope | Meaning | Appropriate use |
|---|---|---|
| Sample-specific observation | Direct summary of cells assigned to the active sample | Choose a compartment for confirmation |
| Patient-specific model perturbation | Model change after a perturbation restricted to that sample's cells | Form an experimental hypothesis |
| Cohort-level association | Marker, pathway, or survival result from the analysed cohort | Biological context; not an individual prediction |
| External reference | Versioned knowledge not calculated from the case | Context after source verification |
| Computational hypothesis | Target, pathway, drug, or mechanism nomination | Literature, assay, or trial-screening review |

The report does not collapse these scopes into one confidence score.

## Data boundary

Breast, lung, and pancreatic uploads are trained using their locked manuscript
bulk-survival reference. Other diseases must provide matched bulk and survival
data. The complete CSV is checked before training. A newly uploaded single-cell
dataset never receives patient inference from an unrelated pre-trained
single-cell reference. Gene names must be unique; the workflow verifies overlap
and aligns both modalities to the common genes in bulk-column order.

## Agent boundary

The agent can retrieve case evidence, explain provenance and limitations, run
approved tools, and generate a draft report. It cannot issue orders, diagnose,
claim treatment efficacy, invent missing case data, or bypass QC/review gates.
Deterministic policy checks run on every final response and report narrative.

## Perturbation boundary

Target perturbation currently changes the target plus selected PPI neighbours;
reports name that operation explicitly. Pathway perturbation changes all modeled
genes in a pathway/program. Drug perturbation is target/signature mapping, not a
pharmacologic response model. Per-request target/pathway figures show modeled
high-risk/background label transitions and report flip counts, net reduction,
and statistical tests beside the conversation. All require independent verification.

## Deployment boundary

The supplied authentication is suitable for a protected local prototype, not a
complete clinical identity, authorization, retention, or security system.
Institutional deployment requires the gates in `CLINICAL_RELEASE_GATES.md`.
