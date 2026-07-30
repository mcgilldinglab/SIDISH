"""SIDISH agent — an LLM that orchestrates the SIDISH tools from plain language.

This is the actual agent: given a chat request, the model DECIDES which tools
in sidish_tools.py to call, reads their results, and loops until it can answer
or has produced the report. The model never computes anything itself.

Works with any OpenAI-compatible endpoint:
  * Gemini (free tier):
      export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
      export OPENAI_API_KEY=<gemini_key> ; export SIDISH_LLM_MODEL=gemini-2.0-flash
  * Qwen3 / local / free / private (recommended for hospital data):
      ollama pull qwen3 ; ollama serve
      export OPENAI_BASE_URL=http://localhost:11434/v1
      export OPENAI_API_KEY=ollama ; export SIDISH_LLM_MODEL=qwen3
  * OpenAI: just set OPENAI_API_KEY

Run:
    python sidish_agent.py "Generate a precision-medicine report for CID3946"
    python sidish_agent.py            # interactive chat
"""
import json, os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # macOS: tolerate duplicate libomp
import sidish_tools as T

SYSTEM = """You are the SIDISH precision-oncology agent. You help clinicians and
researchers analyze a patient's single-cell data by calling SIDISH tools.

The trained BRCA SIDISH model is ALREADY loaded and live — do NOT call init_sidish
unless the user supplies new data paths.

This is a CONVERSATION: use earlier tool results and the user's follow-ups (e.g. "that
sample", "those targets") to decide the next step. Do only what the user asked for in
each turn; don't jump ahead to a full report unless they ask for one.

Rules:
- Use the tools to get every fact. NEVER invent numbers, genes, drugs, or p-values;
  the numbers in your narrative must come verbatim from tool results.
- New dataset: if the user gives their own single-cell file, call load_dataset(adata_path,
  cancer_type, [bulk_path]). If it returns need_bulk / needs_training, relay that clearly
  (SIDISH must be trained on new data with a matched bulk+survival table — a GPU/lab job).
- "Find the high-risk cells / show the cohort": highrisk_overview gives the per-sample
  high-risk distribution; show_figures(kinds=[...]) makes the UMAPs, survival curve, and
  distribution plot; survival_km gives the log-rank p-value.
- "Propose targets for sample X's tumour": perturbation(patient="X") runs the knockout on
  THAT patient's cells (patient-specific); then drug_perturbation for candidate compounds.
  perturbation defaults to the top ~50 marker genes; pass genes=[...] for a specific panel,
  n_genes to change the count, or scope='all' for genome-wide (warn: slow, GPU/lab only).
- "Perturb a pathway for sample X": pathway_perturbation(patient="X", pathway="...") runs
  a patient-specific pathway-program perturbation. Explain that it is a network-model hypothesis.
- "Find drug directions": drug_perturbation maps confirmed model targets to research leads.
  Always preserve indirectness, indication, and evidence limitations; never predict response.
- show_figures returns saved PNG paths — mention them so the user can view the plots.
- "Make a report for patient X": build_report(patient_id="X", use_llm=true,
  patient_specific=true) unless the user explicitly requests a cohort-level report.
  Then tell them the report path.
- Frame therapeutic findings as candidate hypotheses for validation, never as treatment
  directives. This is research-use decision support.
- Be concise. Explain what you found, grounded in tool results."""


def _default_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
                  base_url=os.environ.get("OPENAI_BASE_URL"))


def _default_model():
    """Pick a sensible default model for the configured endpoint."""
    if os.environ.get("SIDISH_LLM_MODEL"):
        return os.environ["SIDISH_LLM_MODEL"]
    base = (os.environ.get("OPENAI_BASE_URL") or "").lower()
    if "generativelanguage" in base:   # Gemini OpenAI-compatible endpoint
        return "gemini-2.0-flash"
    if "11434" in base or "ollama" in base:
        return "qwen3:8b"
    return "gpt-4o"


def _bootstrap(verbose=True):
    """Load the trained model so the analysis tools are live, regardless of whether the
    LLM calls init_sidish. Skip if already live (e.g. a notebook handed off a model)."""
    if T._live():
        if verbose:
            print("  [context] live (already loaded)")
        return
    try:
        boot = T.init_sidish()
        if verbose:
            print(f"  [context] {boot.get('mode')} ({boot.get('device', '-')}, "
                  f"{boot.get('n_cells', '?')} cells)")
    except Exception as e:
        raise RuntimeError(f"live SIDISH initialization failed; no mock substitution: {e}") from e


def _chat_create(client, **kw):
    """LLM call with simple exponential backoff on rate limits (free tiers are tight)."""
    import time
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kw)
        except Exception as e:
            transient = "429" in str(e) or "RateLimit" in type(e).__name__ or "503" in str(e)
            if transient and attempt < 4:
                wait = 2 ** attempt
                print(f"  [LLM busy — retrying in {wait}s]")
                time.sleep(wait)
                continue
            raise


def _run_loop(messages, client, model, max_steps=10, verbose=True):
    """Run the tool-calling loop on an existing message list (mutated in place).
    Returns the final assistant text. Used for both one-shot and multi-turn chat."""
    for _ in range(max_steps):
        resp = _chat_create(client, model=model, messages=messages,
                            tools=T.TOOLS_SPEC, tool_choice="auto")
        msg = resp.choices[0].message
        messages.append(msg)

        if not getattr(msg, "tool_calls", None):
            from sidish_policy import audit_text
            if msg.content and audit_text(msg.content):
                msg.content = ("I cannot release that wording because it crosses the SIDISH "
                               "decision-support boundary. I can describe the evidence, scope, "
                               "limitations, and validation options instead.")
            if verbose and msg.content:
                print("\n🤖", msg.content)
            return msg.content

        for call in msg.tool_calls:                       # the model chose tool(s)
            name = call.function.name
            args = json.loads(call.function.arguments or "{}")
            if verbose:
                print(f"  ↳ tool: {name}({args})")
            try:
                result = T.REGISTRY[name](**args)
            except Exception as e:
                result = {"error": str(e)}
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "name": name, "content": json.dumps(result, default=str)})
    return "Stopped after max steps."


def run(user_msg: str, client=None, model=None, max_steps: int = 10, verbose: bool = True):
    """One-shot: run the agent loop for a single request. Returns the final assistant text."""
    client = client or _default_client()
    model = model or _default_model()
    _bootstrap(verbose)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_msg}]
    return _run_loop(messages, client, model, max_steps, verbose)


def chat():
    """Multi-turn conversation. History is preserved across turns, so follow-ups like
    'propose targets for that sample' resolve against what was already found."""
    client = _default_client()
    model = _default_model()
    _bootstrap(verbose=True)
    print("SIDISH agent — conversational. e.g. 'find the high-risk cells in my cohort',\n"
          "then 'propose targets for the highest-burden sample', then 'make a report for it'.\n"
          "(Ctrl-C to exit)")
    messages = [{"role": "system", "content": SYSTEM}]
    while True:
        try:
            user = input("\nyou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not user:
            continue
        messages.append({"role": "user", "content": user})
        _run_loop(messages, client, model, max_steps=10, verbose=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run(" ".join(sys.argv[1:]))
    else:
        chat()
