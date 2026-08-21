"""
Step 5: Streamlit UI
------------------------
The user-facing app: upload a CV, run the review, browse results.

Design decisions:

1. Verification status is shown INLINE with each quote (checkmark or
   warning icon), not buried in a separate tab. The whole point of
   Step 4's validation work is wasted if a reviewer has to go looking
   for it - it should be impossible to read an evidence quote without
   also seeing whether it was confirmed against the source CV.

2. Tiers are color-coded consistently (green=strong evidence,
   blue=moderate, amber=mentioned only, gray=not demonstrated,
   red=insufficient info) so a reviewer can visually scan the whole
   report and immediately spot the well-evidenced areas vs. the gaps,
   without reading every reasoning sentence.

3. No download/export of a "score" or "verdict" - consistent with the
   schema and prompt design from Steps 2-3, the UI has structurally
   nowhere to display a hiring judgment, because the underlying data
   model doesn't have one.

4. Errors (Ollama not running, extraction failing after retries) are
   caught and shown as a clear message in the UI itself, not a raw
   Python traceback - this matters for anyone besides you ever running
   this, and for the live demo not to look broken if something's off.
"""

import sys
import tempfile
from pathlib import Path

# streamlit_app.py lives at app/ui/streamlit_app.py, so parent.parent
# is the project root - must be on sys.path before the app.* imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

from app.loaders.document_loader import load_document
from app.extraction.extractor import Extractor, ExtractionError
from app.validation.validator import validate_review, VerificationStatus
from app.schema.evidence import CompetencyTier


TIER_DISPLAY = {
    CompetencyTier.DEMONSTRATED_STRONG: ("🟢", "Demonstrated (Strong)", "#1a7f37"),
    CompetencyTier.DEMONSTRATED_MODERATE: ("🔵", "Demonstrated (Moderate)", "#0969da"),
    CompetencyTier.MENTIONED_ONLY: ("🟡", "Mentioned Only", "#9a6700"),
    CompetencyTier.NOT_DEMONSTRATED: ("⚪", "Not Demonstrated", "#6e7781"),
    CompetencyTier.INSUFFICIENT_INFO: ("🟠", "Insufficient Info", "#bc4c00"),
}


def render_quote(quote_text: str, location: str, source_text: str) -> None:
    """Render one evidence quote with its verification status inline -
    this is the concrete payoff of Step 4's validation work."""
    from app.validation.validator import verify_quote

    status, score = verify_quote(quote_text, source_text)
    icon = "✅" if status == VerificationStatus.VERIFIED else "⚠️"
    st.markdown(f"{icon} *\"{quote_text}\"*")
    caption = f"— {location}" if location else ""
    if status != VerificationStatus.VERIFIED:
        caption += f"  (⚠️ verification score: {score:.0f}/100 - review this quote manually)"
    if caption:
        st.caption(caption)


def render_competency(comp, source_text: str) -> None:
    icon, label, color = TIER_DISPLAY[comp.tier]
    with st.expander(f"{icon} **{comp.area}** — {label}", expanded=(comp.tier != CompetencyTier.NOT_DEMONSTRATED)):
        st.markdown(f"<span style='color:{color}; font-weight:600'>{label}</span>", unsafe_allow_html=True)
        st.write(comp.reasoning)
        if comp.evidence:
            st.markdown("**Evidence:**")
            for ev in comp.evidence:
                render_quote(ev.quote, ev.location, source_text)


def main():
    st.set_page_config(page_title="CV Evidence Review", layout="centered")
    st.title("CV AI-Competency Evidence Review")
    st.caption(
        "Reviews a CV for demonstrated AI-related technical competencies. "
        "This tool does not make hiring recommendations or produce a score - "
        "it reports evidence only."
    )

    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Ollama model", ["llama3.2", "qwen2.5:7b"], index=0)
        st.caption("Requires Ollama running locally with the selected model pulled.")

    uploaded = st.file_uploader("Upload a CV", type=["txt", "md", "pdf", "docx"])

    if uploaded is not None:
        run = st.button("Run Review", type="primary")

        if run:
            suffix = Path(uploaded.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                with st.spinner("Loading document..."):
                    doc = load_document(tmp_path)

                if not doc.text.strip():
                    st.error(
                        "No text could be extracted from this file. If this is a "
                        "scanned/image-based PDF, text extraction won't work without OCR."
                    )
                    return

                with st.spinner(f"Analyzing CV with {model} (this can take 30-90s)..."):
                    extractor = Extractor(model=model)
                    review = extractor.extract(doc.text, source=uploaded.name)

                with st.spinner("Validating evidence against source text..."):
                    report = validate_review(review, doc.text)

                st.session_state["review"] = review
                st.session_state["report"] = report
                st.session_state["source_text"] = doc.text

            except ExtractionError as e:
                st.error(
                    f"The model couldn't produce a valid structured review after "
                    f"multiple attempts. This can happen with smaller local models "
                    f"on complex CVs. Details: {e}"
                )
                return
            except Exception as e:
                if "Connection" in str(type(e).__name__) or "refused" in str(e).lower():
                    st.error(
                        "Couldn't reach Ollama. Make sure it's running (`ollama serve`) "
                        f"and that the '{model}' model is pulled (`ollama pull {model}`)."
                    )
                else:
                    st.error(f"Something went wrong: {e}")
                return
            finally:
                Path(tmp_path).unlink(missing_ok=True)

    # Render results if we have them (persisted in session_state so they
    # survive Streamlit's rerun-on-interaction behavior, e.g. expanding
    # a section shouldn't wipe the results and force a re-run)
    if "review" in st.session_state:
        review = st.session_state["review"]
        report = st.session_state["report"]
        source_text = st.session_state["source_text"]

        st.divider()
        st.subheader(f"Results: {review.candidate_source}")

        verified_pct = (
            100 * report.verified_count / report.total_quotes if report.total_quotes else 100
        )
        st.caption(
            f"Evidence verification: {report.verified_count}/{report.total_quotes} "
            f"quotes confirmed against source text ({verified_pct:.0f}%)"
        )
        if not report.all_verified:
            st.warning(
                f"{len(report.flagged)} quote(s) did not verify cleanly against the "
                "source CV and are marked with ⚠️ below - please review these manually."
            )

        for comp in review.competencies:
            render_competency(comp, source_text)

        if review.additional_technologies_noted:
            st.divider()
            st.markdown("**Other AI-relevant technologies noted (not in standard list):**")
            st.write(", ".join(review.additional_technologies_noted))


if __name__ == "__main__":
    main()
