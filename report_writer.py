"""Turn a SIDISH results payload (JSON) into a clinical-style HTML report.

Two modes:
  * LLM mode (default): an OpenAI-compatible model writes the four section
    bodies from the payload. base_url is configurable, so you can point it at
    OpenAI, Azure, or a LOCAL server (vLLM/Ollama) when data must stay on-prem.
  * --no-llm: a deterministic template fills the sections straight from the
    payload. No API needed. Use this to guarantee a report for the demo.

Both paths run through safety.guard() before rendering.

Usage:
    python report_writer.py payloads/mock_payload_BRCA_CID3946.json
    python report_writer.py payloads/CID3946.json --no-llm
    python report_writer.py payloads/CID3946.json --pdf     # needs weasyprint
"""
import argparse, base64, datetime, json, mimetypes, os, re, sys
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
import safety

HERE = Path(__file__).parent
PROMPT = (HERE / "prompts" / "report_prompt.md").read_text()


# ---------- self-contained figures ----------
def _inline_figures(html: str, keep_only=None) -> str:
    """Embed each <figure>'s image as a base64 data URI; drop the whole <figure>
    block if its image file is missing. Keeps the report a single portable file.
    keep_only: if given (list of substrings), drop any <figure> whose src does not
    contain one of them (e.g. keep_only=['survival'] keeps only the KM plot)."""
    def fig_repl(m):
        block = m.group(0)
        src_m = re.search(r"<img[^>]*src=['\"](?P<src>[^'\"]+)['\"]", block)
        if not src_m:
            return block
        src = src_m.group("src")
        if keep_only is not None and not any(k in src for k in keep_only):
            return ""  # figure not in the keep list -> drop it (text-only variant)
        if src.startswith("data:"):
            return block
        p = Path(src) if os.path.isabs(src) else (HERE / src)
        if not p.exists():
            return ""  # figure not generated -> drop it rather than show a broken image
        mime = mimetypes.guess_type(str(p))[0] or "image/png"
        uri = "data:%s;base64,%s" % (mime, base64.b64encode(p.read_bytes()).decode())
        return block.replace(src, uri)
    return re.sub(r"<figure>.*?</figure>", fig_repl, html, flags=re.DOTALL)


# ---------- LLM section generation ----------
def _llm_sections(payload: dict) -> dict:
    """Call an OpenAI-compatible chat model to write section1..4."""
    from openai import OpenAI  # pip install openai
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", "not-needed-for-local"),
        base_url=os.environ.get("OPENAI_BASE_URL"),  # None -> OpenAI default
    )
    model = os.environ.get("SIDISH_LLM_MODEL", "gpt-4o")
    system, user = PROMPT.split("## USER")
    system = system.replace("## SYSTEM", "").strip()
    messages = [
        {"role": "system", "content": system},
        {"role": "user",
         "content": user.strip() + "\n\nPAYLOAD:\n" + json.dumps(payload, indent=2)},
    ]
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ---------- attach figures to LLM-written prose ----------
def _fig(src, cap):
    return (f"<figure><img src='{src}' alt='{cap}'>"
            f"<figcaption>{cap}</figcaption></figure>")


def _attach_figures(sections: dict, p: dict) -> dict:
    """The LLM writes prose only; we deterministically append the real figures so the
    LLM report is as complete (and self-contained) as the deterministic one."""
    f = p.get("features", {}).get("figures", {})
    pf = p.get("perturbation", {}).get("figures", {})
    pr = p.get("prognosis", {})
    plan = {
        "section2": [(f.get("umap_highrisk"), "High-risk (red) vs background cells"),
                     (f.get("umap_celltype"), "Cell-type composition (celltype_major)"),
                     (f.get("celltype_barplot"), "High-risk cell-type composition for this patient"),
                     (f.get("marker_heatmap"), "Marker gene expression, high-risk vs background")],
        "section3": [(pf.get("single_gene_bar"), "Top single-gene knockouts"),
                     (pf.get("perturbation_umap"), "High-risk to background transition after knockout")],
        "section4": [(pr.get("km_figure"), "Kaplan-Meier survival by SIDISH high-risk signature (TCGA-BRCA)")],
    }
    out = dict(sections)
    for sec, figs in plan.items():
        body = out.get(sec, "") or ""
        # attach unless an actual <img> for this src already exists (the LLM often mentions
        # the filename in prose, which must NOT suppress the real figure)
        add = "".join(_fig(s, c) for s, c in figs
                      if s and ("src='%s'" % s) not in body and ('src="%s"' % s) not in body)
        if not add:
            continue
        if body.rstrip().endswith("</div>"):
            body = body.rstrip()[:-6] + add + "</div>"
        else:
            body = body + add
        out[sec] = body
    return out


# ---------- deterministic, locked-data blocks (used in BOTH LLM and fallback paths) ----------
def _exec_summary(p: dict) -> str:
    """Medical-check-up style summary, up top. Restates locked numbers only."""
    f, pert, prog, pt = p["features"], p["perturbation"], p["prognosis"], p["patient"]
    comp = f.get("celltype_composition", {})
    top_ct = max(comp, key=comp.get) if comp else "stromal"
    acts = p.get("pathways", {}).get("activated", [])
    paths = ", ".join(a["name"] for a in acts[:3]) if acts else "extracellular-matrix and stromal programs"
    singles = pert.get("single_gene_top", [])
    top1 = singles[0] if singles else {"target": "—", "high_risk_reduction_percent": 0}
    reduced = ", ".join(f.get("reduced_vs_control", []) or []) or "immune/effector populations"
    return (
        "<p>Overall, <strong>{pid}</strong> shows a high SIDISH high-risk cell burden "
        "(<strong>{hr:.1f}%</strong>), dominated by {ct} cells, with a relative reduction in {red}. "
        "The high-risk transcriptional program is enriched for {paths}. In-silico perturbation "
        "prioritizes <strong>{tgt}</strong> ({r:.1f}% high-risk reduction) as the strongest single "
        "candidate. The marker signature stratifies TCGA-BRCA survival at p = {km}. These are "
        "computational hypotheses for validation, not treatment recommendations.</p>"
    ).format(pid=pt["patient_id"], hr=f.get("patient_high_risk_fraction", 0) * 100, ct=top_ct,
             red=reduced, paths=paths, tgt=top1["target"],
             r=top1["high_risk_reduction_percent"], km=prog.get("km_pvalue", "n/a"))


def _pathway_block(p: dict) -> str:
    """Pathway-analysis subsection (activated pathways + suppressed gene program)."""
    pw = p.get("pathways", {})
    acts = pw.get("activated", [])
    down = pw.get("suppressed_genes", [])
    if not acts and not down:
        return ""
    rows = "".join(
        f"<tr><td>{a['name']}</td><td>{a.get('source', 'GO')}</td><td>{a['q_value']}</td>"
        f"<td>{', '.join(a.get('genes', [])[:8])}</td></tr>" for a in acts)
    tbl = (f"<table><tr><th>Activated pathway</th><th>Source</th><th>q-value (FDR)</th>"
           f"<th>Key genes</th></tr>{rows}</table>") if acts else ""
    downp = (f"<p><strong>Suppressed program:</strong> genes reduced in high-risk cells include "
             f"{', '.join(down[:10])}. {pw.get('suppressed_note', '')}</p>") if down else ""
    return ("<h4>Pathway analysis</h4>"
            "<p>Biological processes over-represented in the high-risk marker program "
            "(GO enrichment, FDR q-values):</p>" + tbl + downp)


def _drug_table(p: dict) -> str:
    """4-column drug table, populated ONLY from the locked CMap target->compound map."""
    pert = p.get("perturbation", {})
    singles = pert.get("single_gene_top", [])
    dm = pert.get("drug_mapping", {})
    by = {r["target"]: r.get("compounds", []) for r in dm.get("by_target", [])} \
        if isinstance(dm, dict) else {}
    rows = []
    for s in singles:
        tgt, red = s["target"], s.get("high_risk_reduction_percent", 0)
        comps = by.get(tgt)
        if comps is None:                                  # no CMap block (mock): use candidate_drugs
            comps = [{"drug": d, "moa": "", "status": ""} for d in (s.get("candidate_drugs") or [])]
        if comps:
            for c in comps:
                rows.append(
                    f"<tr><td>{tgt}</td><td>{c.get('drug', '—')} — {c.get('moa', '')}</td>"
                    f"<td>top knockout ({red:.1f}% high-risk reduction)</td>"
                    f"<td>{c.get('status', '—')}; requires clinical validation</td></tr>")
        else:
            rows.append(f"<tr><td>{tgt}</td><td>—</td>"
                        f"<td>knockout ({red:.1f}% high-risk reduction)</td>"
                        f"<td>no CMap-mapped compound</td></tr>")
    return ("<table><tr><th>Target / pathway</th><th>Possible drug class</th>"
            f"<th>Rationale</th><th>Note</th></tr>{''.join(rows)}</table>")


def _mechanism_block(p: dict) -> str:
    """Mechanism-of-action subsection (§3): affected gene network -> pathway + TF enrichment
    -> the 'why it works' narrative. All content from the mechanism tool (grounded)."""
    m = p.get("mechanism", {})
    if not m or m.get("error") or not m.get("target"):
        return ""
    paths, tfs = m.get("pathways", []), m.get("tf_enrichment", [])
    ptbl = "".join(f"<tr><td>{a['pathway']}</td><td>{a.get('q_value','')}</td>"
                   f"<td>{a.get('overlap','')}</td><td>{', '.join(a.get('genes', [])[:6])}</td></tr>"
                   for a in paths)
    ttbl = "".join(f"<tr><td>{t['tf']}</td><td>{t['n_targets']}</td>"
                   f"<td>{', '.join(t.get('targets', [])[:6])}</td></tr>" for t in tfs)
    perturbed = m.get("tfs_perturbed", [])
    return ("<h4>Mechanism of action &mdash; why this target</h4>"
            f"<p>{m.get('narrative', '')}</p>"
            f"<p><strong>Affected gene network</strong> (target + interaction partners, "
            f"n={m.get('n_affected')}): {', '.join(m.get('affected_genes', [])[:18])}&hellip;</p>"
            + ("<p><strong>Enriched functions among the affected genes:</strong></p>"
               "<table><tr><th>Pathway / function</th><th>q-value</th><th>genes hit</th>"
               f"<th>key genes</th></tr>{ptbl}</table>" if paths else "")
            + ("<p><strong>Transcription factors implicated</strong> "
               "(regulons over-represented in the affected genes):</p>"
               f"<table><tr><th>TF</th><th>targets hit</th><th>target genes</th></tr>{ttbl}</table>"
               if tfs else "")
            + (f"<p>Transcription factors within the affected network: {', '.join(perturbed)}.</p>"
               if perturbed else ""))


def _cost_block(p: dict) -> str:
    """Clinical-style itemised cost estimate (its own report section)."""
    c = p.get("cost", {})
    if not c or not c.get("items"):
        return ""
    cur = c.get("currency", "")
    rows = "".join(f"<tr><td>{it['item']}</td><td>{it['unit']}</td><td>{it['qty']}</td>"
                   f"<td>{it['unit_price']:,}</td><td>{it['subtotal']:,}</td></tr>" for it in c["items"])
    return ("<div class='section-body'>"
            f"<p>Indicative itemised cost for this precision-medicine workup ({cur}):</p>"
            "<table><tr><th>Item</th><th>Unit</th><th>Qty</th><th>Unit price</th><th>Subtotal</th></tr>"
            f"{rows}<tr><td colspan='4' style='text-align:right'><strong>Estimated total</strong></td>"
            f"<td><strong>{c['total']:,} {cur}</strong></td></tr></table>"
            f"<p><em>{c.get('note', '')}</em></p></div>")


def _attach_tables(sections: dict, p: dict) -> dict:
    """Inject the locked pathway table into §2 and the locked drug table into §3 of the
    LLM-written sections, so those numbers/drugs never come from the model."""
    out = dict(sections)
    plan = [("section2", _pathway_block(p), "Activated pathway"),
            ("section3", "<p><strong>Candidate drug directions</strong> (targets mapped to "
                         "compounds via CMap; '—' where none):</p>" + _drug_table(p)
                         + _mechanism_block(p), "Possible drug class")]
    for sec, add, marker in plan:
        body = out.get(sec, "") or ""
        if not add or marker in body:
            continue
        if body.rstrip().endswith("</div>"):
            body = body.rstrip()[:-6] + add + "</div>"
        else:
            body = body + add
        out[sec] = body
    return out


# ---------- deterministic fallback (no LLM) ----------
def _fallback_sections(p: dict) -> dict:
    pt, cc = p["patient"], p["cohort_context"]
    f, pert, prog = p["features"], p["perturbation"], p["prognosis"]

    def li(items): return "<ul>" + "".join(f"<li>{x}</li>" for x in items) + "</ul>"

    s1 = (
        f"<div class='section-body'><p>{pt['patient_id']} is a {pt['disease'].lower()} "
        f"case of subtype <strong>{pt['cancer_subtype']}</strong>. Available metadata: "
        f"age {pt['age']}, sex {pt['sex']}, stage {pt['stage']}, prior treatment "
        f"{pt['prior_treatment']}.</p>"
        f"<p>Fields marked <em>illustrative</em> are placeholders; scRNA-seq datasets "
        f"often lack full clinical metadata. Data source: {pt['data_source']}.</p>"
        f"<p>Cohort context: {cc['cohort']}, {cc['n_cells_total']:,} cells, "
        f"{cc['n_high_risk_total']:,} SIDISH high-risk cells "
        f"({cc['high_risk_fraction_cohort']*100:.1f}% of the cohort).</p></div>"
    )

    comp_rows = "".join(
        f"<tr><td>{k}</td><td>{v*100:.1f}%</td></tr>"
        for k, v in f["celltype_composition"].items())
    markers = li([f"<strong>{m['gene']}</strong> — {m['note']}" for m in f["marker_genes"]])
    s2 = (
        f"<div class='section-body'>"
        f"<p>SIDISH assigns this patient a high-risk cell burden of "
        f"<strong>{f['patient_high_risk_fraction']*100:.1f}%</strong> "
        f"({f['patient_rank_in_cohort']}). For contrast, {f['contrast_patient']['patient_id']} "
        f"carries {f['contrast_patient']['high_risk_fraction']*100:.1f}%.</p>"
        f"<p>High-risk cell composition:</p>"
        f"<table><tr><th>Cell type</th><th>Share of high-risk cells</th></tr>{comp_rows}</table>"
        f"<p>Enriched vs control: {', '.join(f['enriched_vs_control'])}. "
        f"Reduced: {', '.join(f['reduced_vs_control'])}.</p>"
        f"<p>Biological interpretation: high-risk cells are enriched for "
        f"{', '.join(f['enriched_vs_control'])} and relatively depleted of "
        f"{', '.join(f['reduced_vs_control'])}, indicating the dominant cellular program "
        f"driving the high-risk phenotype in this patient.</p>"
        f"<p>Precision marker genes for the high-risk program:</p>{markers}"
        + _pathway_block(p) +
        f"<figure><img src='{f['figures']['umap_highrisk']}' alt='High-risk UMAP'>"
        f"<figcaption>High-risk (red) vs background cells.</figcaption></figure>"
        f"<figure><img src='{f['figures']['umap_celltype']}' alt='Cell-type UMAP'>"
        f"<figcaption>Cell-type composition (celltype_major).</figcaption></figure>"
        f"<figure><img src='{f['figures']['celltype_barplot']}' alt='Composition bar'>"
        f"<figcaption>High-risk cell-type composition for this patient.</figcaption></figure>"
        f"<figure><img src='{f['figures']['marker_heatmap']}' alt='Marker heatmap'>"
        f"<figcaption>Marker gene expression, high-risk vs background.</figcaption></figure>"
        f"</div>"
    )

    def prow(x):
        drugs = "; ".join(x["candidate_drugs"]) if x.get("candidate_drugs") else "—"
        return (f"<tr><td>{x['target']}</td>"
                f"<td>{x['high_risk_reduction_percent']:.1f}%</td><td>{drugs}</td></tr>")
    single = "".join(prow(x) for x in pert["single_gene_top"])
    dual = "".join(prow(x) for x in pert["dual_gene_top"])
    top1 = pert["single_gene_top"][0]
    topc = pert["dual_gene_top"][0]
    s3 = (
        f"<div class='section-body'>"
        f"<p>Single-gene in-silico knockouts ranked by high-risk reduction:</p>"
        f"<table><tr><th>Target</th><th>High-risk reduction</th><th>Candidate drug(s)</th></tr>{single}</table>"
        f"<p>Combination (dual-gene) knockouts:</p>"
        f"<table><tr><th>Targets</th><th>High-risk reduction</th><th>Candidate drug(s)</th></tr>{dual}</table>"
        f"<p>For this patient, <strong>{top1['target']}</strong> is the strongest single "
        f"candidate ({top1['high_risk_reduction_percent']:.1f}% reduction), and "
        f"<strong>{topc['target']}</strong> is the strongest combination "
        f"({topc['high_risk_reduction_percent']:.1f}%). These are candidate therapeutic "
        f"hypotheses prioritized for validation, not treatment directives.</p>"
        f"<p><strong>Candidate drug directions</strong> (targets mapped to compounds via CMap; "
        f"'—' where none):</p>" + _drug_table(p) + _mechanism_block(p) +
        f"<figure><img src='{pert['figures']['single_gene_bar']}' alt='Perturbation ranking'>"
        f"<figcaption>Top single-gene perturbation scores.</figcaption></figure>"
        f"<figure><img src='{pert['figures']['perturbation_umap']}' alt='Perturbation UMAP'>"
        f"<figcaption>High-risk to background transition after knockout.</figcaption></figure>"
        f"</div>"
    )

    s4 = (
        f"<div class='section-body'>"
        f"<p>{pt['patient_id']} shows a high high-risk burden "
        f"({f['patient_high_risk_fraction']*100:.1f}%) dominated by "
        f"{max(f['celltype_composition'], key=f['celltype_composition'].get)} cells. "
        f"The strongest candidate strategy is <strong>{top1['target']}</strong>; the "
        f"strongest combination is <strong>{topc['target']}</strong>.</p>"
        f"<p>Prognostic support: marker genes from this patient stratify TCGA survival "
        f"at p = {prog['km_pvalue']}. {prog['km_note']}</p>"
        f"<figure><img src='{prog['km_figure']}' alt='Kaplan-Meier'>"
        f"<figcaption>Kaplan-Meier survival by SIDISH high-risk marker signature (TCGA-BRCA).</figcaption></figure>"
        f"<p>Suggested next steps: experimental validation of the prioritized target(s) "
        f"in patient-derived models, review by the molecular tumor board, and correlation "
        f"with standard-of-care biomarkers before any clinical consideration.</p>"
        f"<p><em>{safety.DISCLAIMER}</em></p></div>"
    )
    return {"section1": s1, "section2": s2, "section3": s3, "section4": s4}


def _html_to_pdf(html_path, pdf_path) -> bool:
    """Render an HTML file to PDF. Prefer headless Chromium (playwright) — faithful to the
    browser incl. base64 images + print CSS — and fall back to weasyprint if present."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()
            page.goto("file://" + str(Path(html_path).resolve()))
            page.pdf(path=str(pdf_path), format="A4", print_background=True,
                     margin={"top": "12mm", "bottom": "14mm", "left": "10mm", "right": "10mm"})
            browser.close()
        return True
    except Exception as e:
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(pdf_path))
            return True
        except Exception as e2:
            print(f"[warn] PDF failed (chromium: {e}; weasyprint: {e2}). "
                  f"Open the HTML and Print-to-PDF instead.", file=sys.stderr)
            return False


def build(payload_path: str, use_llm: bool = True, make_pdf: bool = False,
          keep_figures=None, out_suffix: str = "") -> str:
    """Render the 4-section report. keep_figures (list of filename substrings) keeps only
    those figures (e.g. ['survival'] for a text + KM-only variant); out_suffix names the
    output file; make_pdf also writes a PDF alongside the HTML."""
    payload = json.loads(Path(payload_path).read_text())
    try:
        if use_llm:
            sections = _llm_sections(payload)              # LLM writes the narrative...
            sections = _attach_tables(sections, payload)   # ...we inject the locked pathway+drug tables
            sections = _attach_figures(sections, payload)  # ...and the real figures
        else:
            sections = _fallback_sections(payload)
    except Exception as e:  # never let the demo fail — fall back deterministically
        print(f"[warn] LLM path failed ({e}); using deterministic fallback.", file=sys.stderr)
        sections = _fallback_sections(payload)

    sections = safety.guard(sections)
    sections = {k: _inline_figures(v, keep_only=keep_figures) for k, v in sections.items()}
    exec_summary = _inline_figures(safety.sanitize(_exec_summary(payload)), keep_only=keep_figures)
    cost_section = safety.sanitize(_cost_block(payload))       # clinical-style cost estimate

    env = Environment(loader=FileSystemLoader(HERE / "templates"),
                      autoescape=select_autoescape(["html"]))
    tmpl = env.get_template("clinical_report.html.j2")
    html = tmpl.render(
        generated_date=datetime.date.today().isoformat(),
        disclaimer=safety.DISCLAIMER, executive_summary=exec_summary,
        cost_section=cost_section, **payload, **sections,
    )

    out = HERE / "reports" / f"{payload['patient']['patient_id']}_SIDISH_report{out_suffix}.html"
    out.write_text(html)
    print(f"[ok] wrote {out}")

    if make_pdf:
        pdf = out.with_suffix(".pdf")
        if _html_to_pdf(out, pdf):
            print(f"[ok] wrote {pdf}")
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("payload")
    ap.add_argument("--no-llm", action="store_true", help="deterministic, no API")
    ap.add_argument("--pdf", action="store_true", help="also write PDF (headless Chromium)")
    ap.add_argument("--survival-only", action="store_true",
                    help="drop all figures except the survival (KM) plot")
    ap.add_argument("--keep-figures", default=None,
                    help="comma-separated figure name substrings to keep (e.g. survival,umap)")
    ap.add_argument("--suffix", default="", help="suffix for the output filename")
    a = ap.parse_args()
    keep = ["survival"] if a.survival_only else (a.keep_figures.split(",") if a.keep_figures else None)
    suffix = a.suffix or ("_textonly" if a.survival_only else "")
    build(a.payload, use_llm=not a.no_llm, make_pdf=a.pdf, keep_figures=keep, out_suffix=suffix)
