# cv-evidence-review

An AI-powered CV technical-competency reviewer. Given a candidate's CV, it identifies
evidence of AI-related technical competencies (Python, LLMs, embeddings, vector
databases, RAG, ML/DL, AI frameworks, model integration, AI solution architecture) and
produces a structured, evidence-backed report.

**Explicitly out of scope by design:** hiring recommendations, pass/fail judgments,
scores, or ratings of any kind. The system's only job is to report what is and isn't
evidenced in the CV text — nothing more. This constraint isn't just a prompt
instruction; it's structural (see Design Decisions below).

## Architecture

```
CV file (.txt/.pdf/.docx)
    → Loader              (app/loaders/)       file -> plain text
    → Extraction           (app/extraction/)     LLM call -> structured CVReview
        - validated against a strict Pydantic schema
        - self-correcting retry loop on validation failure
        - repair step for one known small-model quirk
    → [next: Validation]  (app/validation/)     confirm evidence quotes are real, not hallucinated
    → [next: UI]           (app/ui/)             Streamlit report view
```

```
cv-evidence-review/
├── app/
│   ├── loaders/document_loader.py     Step 1 — file -> text
│   ├── schema/evidence.py              Step 2 — the evidence-tier data model
│   ├── extraction/extractor.py        Step 3 — LLM extraction + validation + repair
│   ├── validation/                     Step 4 — anti-hallucination quote checking (next)
│   └── ui/                             Step 5 — Streamlit app (next)
└── data/sample_cvs/jordan_rivera.txt   Fictional test CV, deliberately mixes evidence
                                         tiers to exercise every code path
```

## Design decisions

### The evidence-tier model

Every competency area gets classified into exactly one of five tiers:

| Tier | Meaning |
|---|---|
| `demonstrated_strong` | Named project/role + specific technical detail + ideally an outcome |
| `demonstrated_moderate` | Named project/role, but shallow detail |
| `mentioned_only` | Appears in a skills list only — no project or role ties it to real work |
| `not_demonstrated` | Doesn't appear anywhere in the CV |
| `insufficient_info` | Present, but too ambiguous to classify confidently (e.g. "familiar with X") |

This closed enum (not free text) is what forces consistent, comparable classification
instead of the model hedging with vague prose ("solid experience", "some familiarity")
that's hard to compare across competency areas.

### No verdict field, anywhere

The schema (`app/schema/evidence.py`) has no score, rating, or recommendation field at
all — not filtered out after the fact, structurally absent. The exercise brief is
explicit that this system must not make a hiring judgment; enforcing that in the data
model is a stronger guarantee than a prompt instruction the model could ignore or drift
from over a long response.

### Evidence quotes are required, not optional

Every tier except `not_demonstrated` requires at least one verbatim (or near-verbatim)
quote from the CV, enforced by a Pydantic `model_validator`. This is what makes the
upcoming Step 4 (hallucination validation) possible at all — there's a concrete string
to check against the source text, not just trust-the-model reasoning prose.

### Why no RAG/vector DB here, despite this being an AI-competency exercise

This is single-document extraction against a full CV that comfortably fits in an LLM's
context window — there's nothing to *retrieve* across. RAG solves search across many/
large documents; forcing it into a single-CV read would add complexity with no benefit.
(Worth stating explicitly if asked in the presentation — shows deliberate scoping, not
an oversight.)

## The debugging journey (worth knowing for the presentation)

Getting reliable structured output from a local ~7B-class model took several real
iterations. Documenting the failures and fixes here because *how* they were diagnosed
and resolved is arguably more interesting than the final code:

### 1. Empty `reasoning` fields everywhere
**Symptom:** first real run produced valid JSON, but every `reasoning` field was `""`.
**Cause:** the schema declared `reasoning: str` with no constraint — an empty string
still validates.
**Fix:** added `min_length=10` to the field, so an empty reasoning now fails validation
and triggers the retry loop instead of silently passing through.

### 2. "AI Frameworks and Libraries" wrongly marked `not_demonstrated`
**Symptom:** TensorFlow/PyTorch/LangChain were sitting right in the CV's skills list,
but the competency area came back `not_demonstrated`, while those tools were dumped
(along with irrelevant non-AI infra tools like Docker/Kubernetes/AWS) into
`additional_technologies_noted`.
**Cause:** the prompt never told the model that named tools should be attributed as
evidence to the fixed competency areas *first* — `additional_technologies_noted` was
meant only for genuine leftovers that don't map anywhere.
**Fix:** added an explicit mapping table to the prompt (which tools map to which of the
9 areas) and restricted `additional_technologies_noted` to non-mapped AI-relevant items
only, explicitly excluding general infrastructure tools.

### 3. `ModuleNotFoundError: No module named 'app'` (twice)
**Symptom:** running `python3 app/extraction/extractor.py` directly failed on the
`from app.schema.evidence import ...` line.
**Cause:** Python only auto-adds the running script's own directory to `sys.path`, not
the project root — so `app.*` imports fail when the script itself lives inside `app/`.
The first fix attempt put the `sys.path.insert` inside `if __name__ == "__main__":`,
which runs *after* the top-level imports already executed — too late.
**Fix:** moved `sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))`
to the very top of the file, before any `app.*` import.

### 4. `ExtractionError` after 3 attempts — missing `reasoning` on the last item
**Symptom:** `competencies.8.reasoning: Field required` — consistently the *last*
competency area (`AI Solution Architecture`) in the list.
**First hypothesis (wrong-ish):** assumed response truncation from Ollama's default
output token limit, since a 9-area structured response is long. Added `num_predict:
4096` to the Ollama call options. This is a real and legitimate fix in general — a
missing default `num_predict` will genuinely truncate long structured output — but it
turned out not to be the actual cause here, since JSON parsing succeeded cleanly (a
true truncation produces invalid JSON, not valid JSON with one key cleanly absent).
**Real cause (found after checking):** a self-inflicted contradiction between the
prompt and the schema. The prompt said reasoning was required "except for
`not_demonstrated`"; the schema required it unconditionally for every tier. The model
was correctly following the (wrong) prompt instruction and getting penalized for it by
validation that disagreed.
**Fix:** corrected the prompt to require reasoning unconditionally, matching the schema.
**Lesson:** the prompt and the validation schema are two independent sources of truth
that don't automatically agree with each other — nothing enforces that they stay in
sync, and this kind of contradiction is easy to introduce silently while iterating on
either one alone.

### 5. Same error persisted after the prompt fix — turned out to be a real model limitation
**Symptom:** identical `competencies.8.reasoning` missing-field error, same competency,
even after the prompt contradiction from #4 was fixed and confirmed correct.
**Diagnosis:** at low temperature (0.1) the model's output is close to deterministic, so
seeing the *exact same failure survive a genuine, verified prompt fix* was the signal
that this wasn't a wording problem anymore — it looks like a real capability limit of a
smaller local model: attention/quality degrading on the last item of a long generated
JSON array, especially a "negative" (`not_demonstrated`) entry with little to say.
**Fix:** rather than keep burning retries chasing a low-stakes field, added a repair
step (`_repair_missing_reasoning`) that runs after JSON parsing but before Pydantic
validation. If a `not_demonstrated` entry is missing `reasoning`, it's auto-filled with
an honest, non-fabricated placeholder ("No evidence of X was found anywhere in the CV
text.") instead of failing and retrying. Deliberately scoped ONLY to `not_demonstrated`
— any other tier missing reasoning still hard-fails and retries, since a "demonstrated"
or "mentioned" claim with no explanation is a much bigger honesty gap than a "not
present" claim needing no elaboration.
**Lesson:** knowing when a failure is a prompt-engineering problem (fixable with better
wording) versus a genuine model-capability limitation (needs a code-level safety net
instead) is a real engineering judgment call — retrying the same class of fix forever
against a wall wastes time and API calls.

### 6. File-sync confusion during iteration
**Symptom:** after supposedly applying the fix from #4/#5, got the *exact same*
traceback at the *exact same* line numbers.
**Cause:** the updated file was never actually saved to the right path on the local
machine — an old version was still being executed.
**Fix:** cross-checked with `wc -l` against the expected line count of the current
version to positively confirm which version of the file was actually running, rather
than inferring from symptoms alone. This turned out to be a more reliable diagnostic
than reading the traceback, since stale files reproduce old tracebacks exactly.

### 7. `insufficient_info` tier validated but with empty evidence
**Symptom:** `ValueError: tier 'INSUFFICIENT_INFO' requires at least one evidence
quote` — for Machine Learning/Deep Learning, despite the CV having a clearly quotable
ambiguous phrase ("Familiar with machine learning concepts through coursework").
**Cause:** the prompt described *when* to use `insufficient_info` but never explicitly
said the ambiguous phrase itself should be quoted as evidence — the model treated
"insufficient information" as license to leave evidence empty.
**Fix:** added an explicit prompt instruction that the vague wording IS the evidence for
this tier, and evidence should never be left empty if any relevant text exists.

### 8. Known, documented limitation (deliberately not fixed with a validator): silently dropped mixed-strength evidence
**Symptom:** a clean run correctly filled every field, but TensorFlow/PyTorch/LangChain
— present in the CV's skills list — vanished from the output entirely. "AI Frameworks
and Libraries" was marked `demonstrated_strong` citing only the strongest match
(sentence-transformers used in a real project), silently dropping the weaker,
skills-list-only items for the same competency area.
**Cause:** the schema allows exactly one tier per competency area. Forced to summarize
mixed-strength evidence into a single classification, the model kept only the strongest
match and dropped the rest — a real information-loss bug that no Pydantic validator can
catch, since nothing in the schema *requires* the model to remember tools it left out.
**Fix applied:** prompt-level — instruct the model to include ALL relevant quotes for an
area when evidence quality is mixed, tier reflects the strongest evidence, but
`reasoning` must name the weaker/mentioned-only items too.
**Alternative considered and deliberately not built:** restructuring the schema so each
individual evidence quote carries its own micro-tier (rather than one tier per
competency area) would represent this more precisely, but adds real complexity for a
48-72hr take-home exercise. Documenting this as a considered tradeoff — not an
oversight — is itself a reasonable answer if asked "what would you improve."

## Setup

```bash
pip install -r requirements.txt
ollama serve
ollama pull llama3.2   # or qwen2.5:7b
```

## Running (current state — extraction only, no UI yet)

```bash
python3 app/extraction/extractor.py data/sample_cvs/jordan_rivera.txt
```

## Future improvements

- **Step 4 (next): Validation** — programmatically confirm every evidence quote the
  model claims actually appears (verbatim or near-verbatim) in the source CV text,
  as a last line of defense against hallucinated evidence even after all the schema
  and prompt hardening above.
- **Step 5: UI (Streamlit)** — upload CV(s), run review, browse results with evidence
  expandable per competency area.
- **Mixed-evidence schema redesign** — see item #8 above; a per-quote micro-tier would
  be more accurate than the current per-area tier, if time allows.
- **Prompt/schema consistency check** — item #4 above happened because the prompt and
  schema drifted independently. A generator that derives the prompt's field
  requirements directly from the schema's `Field(...)` definitions (rather than
  hand-writing both) would make this class of bug structurally impossible.
- **Model comparison** — most of the reliability issues above are specific to smaller
  local models handling long structured output. Worth comparing llama3.2 vs. qwen2.5:7b
  vs. a larger model on the same CV to see how much of this tuning is model-specific.
- **Multi-CV batch review** — currently single-document; batching + a comparison view
  across candidates would be a natural extension.
- **Config management** — model name, retry counts, `num_predict`, and Ollama URL are
  currently hardcoded in `extractor.py`; should move to environment-based settings.
