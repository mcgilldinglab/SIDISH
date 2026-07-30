# Clinical decision-support release gates

This checklist separates implemented software safeguards from validation that
must be completed by qualified people and the deploying institution.

## Required before any clinician pilot

- [ ] Freeze versioned breast, lung, and pancreatic manuscript bulk cohorts;
      record hashes, consent/data-use basis, endpoints, preprocessing, and splits.
- [ ] Validate each cancer independently on held-out or external cohorts; report
      discrimination, calibration where applicable, uncertainty, failure modes,
      subgroup performance, and sensitivity to thresholds/batch effects.
- [ ] Validate the single-cell input contract, preprocessing, patient/sample
      mapping, cell-type annotations, minimum cell/gene counts, and out-of-
      distribution behavior.
- [ ] Recompute pathway statistics against a versioned gene-set database; do not
      transfer manuscript q-values to a changed marker list.
- [ ] Benchmark target, pathway, and drug hypotheses experimentally or against a
      predefined retrospective validation set.
- [ ] Have an oncologist/pathologist and biostatistician approve report language,
      intended use, contraindicated uses, escalation rules, and reviewer workflow.
- [ ] Conduct a formal regulatory and quality-system assessment for the intended
      jurisdiction and deployment model.
- [ ] Complete privacy/security review: identity and role-based access, encryption,
      secret management, retention/deletion, backups, incident response, audit-log
      integrity, dependency scanning, and threat modeling.
- [ ] Run usability and human-factors testing showing clinicians correctly
      distinguish observations, cohort associations, and hypotheses.
- [ ] Establish model/data change control, locked releases, rollback, monitoring,
      and a process for revalidation after changes.

## Per-case release gate

- [ ] Input provenance and hashes recorded; no direct identifiers.
- [ ] Correct cancer reference or matched user bulk-survival cohort used.
- [ ] Training and analysis jobs completed without unresolved errors.
- [ ] QC reviewed, including sample size, high-risk cells, annotations, batch and
      out-of-distribution checks appropriate to the validated protocol.
- [ ] Evidence scopes and limitations reviewed against underlying artifacts.
- [ ] Candidate drug status and evidence checked against current authoritative
      sources; contraindications and treatment selection remain outside SIDISH.
- [ ] Scientific reviewer and clinical reviewer sign the report.

Until these gates are completed, reports remain research-use drafts and must not
be used as autonomous diagnostic or treatment-selection outputs.
