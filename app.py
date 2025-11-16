"""
app.py - Streamlit LinkedIn Article Poster (single-file)

- UI accepts Gemini + LinkedIn API keys in-session (password inputs)
- Generates article via google.generativeai (if installed) with robust parsing/fallback
- Posts to LinkedIn UGC API (or dry-run shows payload)
- Never writes API keys to disk
- Use 'streamlit run app.py' to run locally
"""

from __future__ import annotations
import os
import sys
import time
import json
import textwrap
from typing import Optional, Dict, Any

import streamlit as st
from streamlit.components.v1 import html as st_html
from dotenv import load_dotenv

# Optional: the app will try to import linkedin_poster_core.py if present.
try:
    import linkedin_poster_core  # type: ignore
    BACKEND_MODULE = True
except Exception:
    BACKEND_MODULE = False

# Try to import the Gemini client (google.generativeai). Be graceful if missing.
try:
    import google.generativeai as genai  # type: ignore
    GENAI_AVAILABLE = True
except Exception:
    genai = None
    GENAI_AVAILABLE = False

import requests

# Constants
LINKEDIN_API_BASE = "https://api.linkedin.com/v2"
ORGS_ENDPOINT = f"{LINKEDIN_API_BASE}/organizationalEntityAcls?q=roleAssignee&state=APPROVED"
UGC_POSTS_ENDPOINT = f"{LINKEDIN_API_BASE}/ugcPosts"
MAX_LINKEDIN_COMMENTARY_LENGTH = 3000
DEFAULT_MODEL = "gemini-pro-latest"

# Tailwind hero (Play CDN) - for a nicer header (dev only)
TAILWIND_HEADER = """
<script src="https://cdn.tailwindcss.com"></script>
<div class="bg-gradient-to-r from-indigo-600 via-purple-600 to-pink-500 p-6 rounded-b-2xl text-white">
  <div class="max-w-6xl mx-auto flex items-center gap-4">
    <div class="flex-shrink-0">
      <div class="w-14 h-14 bg-white/20 rounded-lg flex items-center justify-center text-2xl font-bold">AI</div>
    </div>
    <div>
      <h1 class="text-2xl font-semibold">LinkedIn Article Poster</h1>
      <p class="text-sm opacity-90">Generate professional articles with Gemini and publish to your LinkedIn Organization — visually and safely.</p>
    </div>
  </div>
</div>
"""

st.set_page_config(page_title="LinkedIn Article Poster", layout="wide")
st_html(TAILWIND_HEADER, height=110)

# Layout
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Article settings")

    prompt = st.text_area(
        "Article prompt",
        value=(
            "Generate a professional and engaging article about the benefits of adopting cloud-native "
            "technologies for enterprise businesses. The article should be approximately 500 words, "
            "include a strong introduction and conclusion, and focus on scalability, cost efficiency, "
            "innovation, and developer experience."
        ),
        height=180,
    )

    model = st.selectbox("Model", options=[DEFAULT_MODEL, "gemini-1.0-mini"], index=0)

    st.subheader("LinkedIn settings")
    org_id_override = st.text_input("Organization ID (optional)", help="Numeric org id overrides auto-detection")

    st.subheader("API Keys (session only)")
    st.markdown("Enter your API keys below. Keys are kept only in this session and are not saved to disk.")
    gemini_key = st.text_input("Gemini API Key", type="password", placeholder="Paste your GEMINI_API_KEY here (session only)")
    linkedin_key = st.text_input("LinkedIn API Key (Bearer token)", type="password", placeholder="Paste your LINKEDIN_API_KEY here (session only)")

    st.subheader("Output options")
    save_path = st.text_input("Save generated article to file (optional)", value="article.txt")
    dry_run = st.checkbox("Dry run (do not post to LinkedIn)", value=True)

    st.markdown("---")
    st.caption("Backend status:")
    if BACKEND_MODULE:
        st.success("Found linkedin_poster_core.py — will try to use it.")
    else:
        st.info("No backend module found — app will use its internal generator/poster or mocks.")

    generate_btn = st.button("Generate Article", use_container_width=True)
    publish_disabled = dry_run or (not linkedin_key)
    post_btn = st.button("Publish to LinkedIn", use_container_width=True, disabled=publish_disabled)

with right_col:
    st.header("Preview")
    preview_area = st.empty()
    st.markdown("---")
    st.header("Activity / Logs")
    log_area = st.empty()

# Session state setup
if "generated_article" not in st.session_state:
    st.session_state.generated_article = ""
if "logs" not in st.session_state:
    st.session_state.logs = []
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = gemini_key or ""
if "linkedin_key" not in st.session_state:
    st.session_state.linkedin_key = linkedin_key or ""

def append_log(msg: str):
    st.session_state.logs.append(msg)
    log_area.text_area("Logs", value="\n".join(st.session_state.logs[-100:]), height=220)

# Mock article for fallback/demo
MOCK_ARTICLE = textwrap.dedent(
    """
    Introduction

    Cloud-native technologies are transforming how enterprises design, deploy, and operate software...
    (This is a placeholder article used when Gemini is not available.)
    """
).strip()

# ---------- Backend helpers (inlined) ----------

def configure_gemini_session(api_key: str):
    """Configure google.generativeai client if available."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("google.generativeai is not installed. Install it or use mock generation.")
    genai.configure(api_key=api_key)

def extract_text_from_gemini_response(resp: Any) -> Optional[str]:
    try:
        if hasattr(resp, "text") and isinstance(resp.text, str) and resp.text.strip():
            return resp.text.strip()
        if isinstance(resp, dict):
            for key in ("text", "content"):
                if key in resp and isinstance(resp[key], str) and resp[key].strip():
                    return resp[key].strip()
            candidates = resp.get("candidates") or resp.get("outputs") or resp.get("choices")
            if isinstance(candidates, (list, tuple)) and candidates:
                first = candidates[0]
                if isinstance(first, dict):
                    for k in ("output", "content", "text"):
                        if k in first and isinstance(first[k], str) and first[k].strip():
                            return first[k].strip()
                if isinstance(first, str) and first.strip():
                    return first.strip()
        if hasattr(resp, "candidates"):
            c = getattr(resp, "candidates")
            if isinstance(c, (list, tuple)) and c:
                first = c[0]
                for attr in ("content", "output", "text"):
                    if hasattr(first, attr):
                        val = getattr(first, attr)
                        if isinstance(val, str) and val.strip():
                            return val.strip()
    except Exception:
        return None
    return None

def generate_article_with_gemini(prompt_text: str, model_name: str = DEFAULT_MODEL, api_key: Optional[str] = None, max_retries: int = 2) -> str:
    """Attempt to generate via Gemini; fallback to mock if failing."""
    if not GENAI_AVAILABLE:
        raise RuntimeError("Gemini client not installed.")
    if api_key:
        configure_gemini_session(api_key)
    attempt = 0
    while attempt <= max_retries:
        attempt += 1
        try:
            model = genai.GenerativeModel(model_name)
            if hasattr(model, "generate_content"):
                resp = model.generate_content(prompt_text)
            else:
                resp = model.generate(prompt_text)
            out = extract_text_from_gemini_response(resp)
            if out:
                return out
            raise ValueError("Could not extract text from Gemini response.")
        except Exception as exc:
            # on last attempt raise
            if attempt > max_retries:
                raise
            time.sleep(1.0 * attempt)
    # shouldn't reach here
    raise RuntimeError("Unexpected generation failure.")

def safe_truncate(text: str, max_len: int = MAX_LINKEDIN_COMMENTARY_LENGTH) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."

def get_linkedin_org_id(headers: Dict[str, str]) -> Optional[str]:
    try:
        resp = requests.get(ORGS_ENDPOINT, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        elements = data.get("elements", [])
        for el in elements:
            org_urn = el.get("organizationalTarget") or el.get("organizationalTarget~")
            if org_urn:
                parts = org_urn.split(":")
                return parts[-1]
        return None
    except Exception as e:
        raise

def build_ugc_payload(org_id: str, text: str) -> Dict[str, Any]:
    truncated = safe_truncate(text, MAX_LINKEDIN_COMMENTARY_LENGTH)
    payload = {
        "author": f"urn:li:organization:{org_id}",
        "lifecycleState": "PUBLISHED",
        "specificContent": {
            "com.linkedin.ugc.ShareContent": {
                "shareCommentary": {"text": truncated},
                "shareMediaCategory": "NONE",
            }
        },
        "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
    }
    return payload

def post_to_linkedin_api(payload: Dict[str, Any], bearer_token: str, timeout: int = 15) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Content-Type": "application/json",
    }
    resp = requests.post(UGC_POSTS_ENDPOINT, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    try:
        return resp.json()
    except ValueError:
        return {"status_code": resp.status_code, "headers": dict(resp.headers)}

# ---------- Handlers (UI button logic) ----------

if generate_btn:
    append_log("[info] Generation started...")
    # store keys in session (do not write to disk)
    st.session_state.gemini_key = gemini_key or st.session_state.get("gemini_key", "")
    st.session_state.linkedin_key = linkedin_key or st.session_state.get("linkedin_key", "")

    try:
        # Prefer a user-provided backend module if available
        if BACKEND_MODULE and hasattr(linkedin_poster_core, "generate_article"):
            append_log("[info] Using external backend module for generation.")
            try:
                # try passing gemini_api_key if supported
                article_text = linkedin_poster_core.generate_article(prompt, model, gemini_api_key=st.session_state.gemini_key)
            except TypeError:
                article_text = linkedin_poster_core.generate_article(prompt, model)
            append_log("[info] Generation completed using backend module.")
        else:
            # Use genai if available and key present; otherwise fallback to mock
            if GENAI_AVAILABLE and st.session_state.gemini_key:
                append_log("[info] Generating via Gemini (remote).")
                try:
                    article_text = generate_article_with_gemini(prompt, model, api_key=st.session_state.gemini_key)
                    append_log("[info] Gemini generation succeeded.")
                except Exception as e:
                    append_log(f"[warn] Gemini generation failed: {e}. Falling back to mock.")
                    article_text = f"# Prompt\n{prompt}\n\n" + MOCK_ARTICLE
            else:
                append_log("[info] Using local mock generation (no Gemini client or key).")
                article_text = f"# Prompt\n{prompt}\n\n" + MOCK_ARTICLE

        st.session_state.generated_article = article_text
        preview_area.code(article_text[:3000], language="markdown")
        if save_path:
            try:
                with open(save_path, "w", encoding="utf-8") as fh:
                    fh.write(article_text)
                append_log(f"[success] Saved generated article to {save_path}")
            except Exception as e:
                append_log(f"[error] Could not save file: {e}")
    except Exception as e:
        append_log(f"[error] Generation failed: {e}")

if post_btn:
    append_log("[info] Publishing started...")
    article_to_post = st.session_state.get("generated_article", "")
    if not article_to_post:
        append_log("[warn] No generated article found. Generate first.")
    else:
        # ensure linkedin_key in session
        st.session_state.linkedin_key = linkedin_key or st.session_state.get("linkedin_key", "")
        if not st.session_state.linkedin_key:
            append_log("[error] LinkedIn API key missing. Provide token in API Keys field.")
        else:
            try:
                if BACKEND_MODULE and hasattr(linkedin_poster_core, "post_to_linkedin"):
                    append_log("[info] Using external backend module for posting.")
                    try:
                        response = linkedin_poster_core.post_to_linkedin(article_to_post, org_id=org_id_override or None, linkedin_api_key=st.session_state.linkedin_key)
                    except TypeError:
                        response = linkedin_poster_core.post_to_linkedin(article_to_post, org_id=org_id_override or None)
                    append_log("[success] Published using backend module. Response (truncated):")
                    append_log(json.dumps(response)[:1000])
                    st.success("Published to LinkedIn (backend module).")
                else:
                    # Build payload with auto org detection
                    headers = {"Authorization": f"Bearer {st.session_state.linkedin_key}", "X-Restli-Protocol-Version": "2.0.0"}
                    org_id = org_id_override
                    if not org_id:
                        try:
                            org_id = get_linkedin_org_id(headers)
                            if not org_id:
                                raise RuntimeError("No organization found for token.")
                            append_log(f"[info] Auto-detected org id: {org_id}")
                        except Exception as e:
                            append_log(f"[error] Could not auto-detect org id: {e}")
                            raise

                    payload = build_ugc_payload(org_id, article_to_post)
                    if dry_run:
                        append_log("[dry-run] Dry run enabled. Payload preview (truncated):")
                        append_log(json.dumps(payload)[:1200])
                        st.success("Dry-run: payload prepared (not posted)")
                    else:
                        append_log("[info] Sending post to LinkedIn API.")
                        resp = post_to_linkedin_api(payload, st.session_state.linkedin_key)
                        append_log("[success] Posted to LinkedIn. Response (truncated):")
                        append_log(json.dumps(resp)[:1000])
                        st.success("Published to LinkedIn.")
            except requests.HTTPError as he:
                body = getattr(he.response, "text", "")
                append_log(f"[http-error] {he} -- {body}")
            except Exception as e:
                append_log(f"[error] Publishing failed: {e}")

# Display full article in an expander for copy/edit
if st.session_state.get("generated_article"):
    with st.expander("Full generated article"):
        st.text_area("Article (editable)", value=st.session_state.generated_article, height=400)

# Footer notes
st.markdown("---")
st.caption("Prototype. Keys are kept only in the Streamlit session. For production use a secrets manager.")
with st.expander("Tips & next steps"):
    st.markdown(
        """
- To enable real Gemini generation, install the google-generativeai package and paste your GEMINI_API_KEY into the UI.
- For actual LinkedIn posting you need a valid LinkedIn OAuth Bearer token with org permissions and an approved organization.
- The app uses the LinkedIn UGC endpoint — refer to LinkedIn docs for required permissions and media upload flows.
"""
    )
