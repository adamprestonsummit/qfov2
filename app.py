import os
import time
import json
import re
import requests
import pandas as pd
import streamlit as st
import google.generativeai as genai
from bs4 import BeautifulSoup

# =============================
# App config
# =============================
st.set_page_config(page_title="Qforia", layout="wide")
st.title("🔍 Qforia: Query Fan‑Out + Content Audit")

# =============================
# Utilities
# =============================
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

@st.cache_data(show_spinner=False)
def fetch_url_snapshot(url: str, timeout: int = 20) -> dict:
    """Fetch a URL and return a lightweight snapshot for LLM analysis.

    Returns:
        dict with keys: url, status, title, meta_description, h1s (list),
        text (truncated), word_count, fetched_ok (bool), error (str|None)
    """
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        status = resp.status_code
        if status >= 400:
            return {
                "url": url,
                "status": status,
                "title": None,
                "meta_description": None,
                "h1s": [],
                "text": "",
                "word_count": 0,
                "fetched_ok": False,
                "error": f"HTTP {status}",
            }

        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        meta_description = meta_desc_tag.get("content", "").strip() if meta_desc_tag else None
        h1s = [h.get_text(strip=True) for h in soup.find_all("h1")]

        # Get visible text quickly (basic approach)
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        # Normalize whitespace
        text = re.sub(r"\n{2,}", "\n", text)
        text = re.sub(r"\s{2,}", " ", text)
        words = text.split()
        word_count = len(words)

        # Truncate to keep prompt sizes manageable
        MAX_CHARS = 16000  # ~4k tokens ballpark
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "\n...[truncated]"

        return {
            "url": url,
            "status": status,
            "title": title,
            "meta_description": meta_description,
            "h1s": h1s,
            "text": text,
            "word_count": word_count,
            "fetched_ok": True,
            "error": None,
        }
    except Exception as e:
        return {
            "url": url,
            "status": None,
            "title": None,
            "meta_description": None,
            "h1s": [],
            "text": "",
            "word_count": 0,
            "fetched_ok": False,
            "error": str(e),
        }


def clean_model_json(text: str) -> str:
    """Remove common fencing and extract the first top‑level JSON object/array."""
    if not text:
        return ""
    t = text.strip()
    # Remove Markdown code fences if present
    if t.startswith("```json"):
        t = t[7:]
    if t.startswith("````json"):
        t = t[8:]
    if t.startswith("```"):
        t = t[3:]
    if t.endswith("```"):
        t = t[:-3]
    t = t.strip()

    # Try to locate the first JSON object/array using a simple bracket balance
    start_idx = None
    for i, ch in enumerate(t):
        if ch in "[{":
            start_idx = i
            break
    if start_idx is not None:
        t = t[start_idx:]
    return t


# =============================
# Gemini setup (via Streamlit secrets)
# =============================
# Expected in .streamlit/secrets.toml:
# GEMINI_API_KEY = "your_key_here"

def get_gemini_key() -> str | None:
    # Prefer Streamlit secrets, fall back to env var
    key = st.secrets.get("GEMINI_API_KEY", None)
    if not key:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    return key


def get_model(model_name: str):
    key = get_gemini_key()
    if not key:
        st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit secrets or environment.")
        st.stop()
    genai.configure(api_key=key)
    return genai.GenerativeModel(model_name)


# =============================
# Prompts
# =============================

def QUERY_FANOUT_PROMPT(q: str, mode: str) -> str:
    min_queries_simple = 10
    min_queries_complex = 20

    if mode == "AI Overview (simple)":
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_simple}. "
            f"For a straightforward query, generating around {min_queries_simple}-{min_queries_simple + 2} may suffice. "
            f"If the query has a few distinct aspects or common follow-ups, aim for {min_queries_simple + 3}-{min_queries_simple + 5}. "
            f"Provide a brief reasoning for why you chose this number. Queries should be tightly scoped and relevant."
        )
    else:
        num_queries_instruction = (
            f"First, analyze the user's query: \"{q}\". Based on its complexity and the '{mode}' mode, "
            f"you must decide on an optimal number of queries to generate. "
            f"This number must be at least {min_queries_complex}. "
            f"For multifaceted queries, generate {min_queries_complex + 5}-{min_queries_complex + 10} or more if warranted. "
            f"Provide a brief reasoning for why you chose this number. Queries should be diverse and in-depth."
        )

    return (
        "You are simulating Google's AI Mode query fan-out process for generative search systems.\n"
        f"The user's original query is: \"{q}\". The selected mode is: \"{mode}\".\n\n"
        "Your first task is to determine the total number of queries to generate and the reasoning for this number, based on the instructions below:\n"
        f"{num_queries_instruction}\n\n"
        "Once you have decided on the number and the reasoning, generate exactly that many unique synthetic queries.\n"
        "Each of the following query transformation types MUST be represented at least once, if total allows:\n"
        "1. Reformulations\n2. Related Queries\n3. Implicit Queries\n4. Comparative Queries\n5. Entity Expansions\n6. Personalized Queries\n\n"
        "The 'reasoning' field for each individual query should explain why that specific query was generated in relation to the original query, its type, and the overall user intent.\n"
        "Do NOT include queries dependent on real-time user history or geolocation.\n\n"
        "Return only a valid JSON object with fields 'generation_details' and 'expanded_queries'."
    )


def CONTENT_AUDIT_PROMPT(expanded_queries: list[dict], page_snapshot: dict) -> str:
    """Prompt to analyze one page against the fan‑out set and output structured recommendations."""
    # Keep prompt size reasonable by trimming queries
    queries_for_prompt = expanded_queries[:30]  # cap to first 30 for token safety

    return (
        "You are an expert SEO/Content strategist. Evaluate the web page against a set of generated queries.\n"
        "Your job: (1) summarize the page, (2) assess coverage for each query, (3) recommend on-page optimizations, and (4) propose new content ideas if gaps exist.\n"
        "Return ONLY JSON with this exact schema: {\n"
        "  \"url\": str,\n"
        "  \"page_summary\": str,\n"
        "  \"coverage\": [ { \"query\": str, \"coverage\": \"good|partial|missing\", \"notes\": str } ],\n"
        "  \"recommended_changes\": [ { \"type\": \"on_page_seo|structure|content_addition|new_article\", \"suggestion\": str, \"impact\": \"high|medium|low\", \"effort\": \"low|medium|high\" } ],\n"
        "  \"new_content_ideas\": [ { \"title\": str, \"angle\": str, \"target_intent\": str, \"outline\": [str] } ]\n"
        "}\n\n"
        f"QUERIES (subset up to 30): {json.dumps(queries_for_prompt, ensure_ascii=False)}\n\n"
        f"PAGE_SNAPSHOT: {json.dumps({k: v for k, v in page_snapshot.items() if k in ['url','title','meta_description','h1s','word_count']}, ensure_ascii=False)}\n\n"
        "PAGE_TEXT (truncated):\n" + page_snapshot.get("text", "")[:12000]
    )


# =============================
# Generation functions
# =============================

def generate_fanout(model, query: str, mode: str) -> tuple[list[dict], dict | None, str | None]:
    prompt = QUERY_FANOUT_PROMPT(query, mode)
    try:
        response = model.generate_content(prompt)
        json_text = clean_model_json(getattr(response, "text", "").strip())
        data = json.loads(json_text)
        generation_details = data.get("generation_details", {})
        expanded_queries = data.get("expanded_queries", [])
        return expanded_queries, generation_details, None
    except json.JSONDecodeError as e:
        raw = getattr(response, "text", "") if 'response' in locals() else ""
        return [], None, f"Failed to parse JSON: {e}. Raw: {raw[:1000]}"
    except Exception as e:
        raw = getattr(response, "text", "") if 'response' in locals() else ""
        return [], None, f"Unexpected error: {e}. Raw: {raw[:1000]}"


def audit_urls_against_queries(model, urls: list[str], expanded_queries: list[dict]) -> list[dict]:
    results = []
    for i, url in enumerate(urls, start=1):
        with st.spinner(f"Fetching {url} ({i}/{len(urls)})..."):
            snapshot = fetch_url_snapshot(url)
        if not snapshot.get("fetched_ok"):
            results.append({
                "url": url,
                "error": snapshot.get("error") or f"HTTP {snapshot.get('status')}",
                "status": snapshot.get("status"),
                "fetched_ok": False,
            })
            continue

        # Call Gemini for this page
        with st.spinner(f"Analyzing content for {url} ..."):
            try:
                prompt = CONTENT_AUDIT_PROMPT(expanded_queries, snapshot)
                resp = model.generate_content(prompt)
                jt = clean_model_json(getattr(resp, "text", ""))
                rec = json.loads(jt)
                rec["status"] = snapshot.get("status")
                rec["title"] = snapshot.get("title")
                rec["meta_description"] = snapshot.get("meta_description")
                rec["word_count"] = snapshot.get("word_count")
                results.append(rec)
            except Exception as e:
                results.append({
                    "url": url,
                    "error": f"Model error: {e}",
                    "fetched_ok": True,
                })
        # Gentle pacing to avoid hammering the site/API
        time.sleep(0.4)
    return results


# =============================
# Sidebar controls
# =============================
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Gemini model",
    [
        "gemini-2.5-flash-latest",
        "gemini-2.5-pro-latest",
    ],
    index=0,
)
mode = st.sidebar.radio("Search Mode", ["AI Overview (simple)", "AI Mode (complex)"])

# Query input
st.sidebar.subheader("Query")
user_query = st.sidebar.text_area("Enter your query", "Enter your query here...", height=120)

# URL inputs
st.sidebar.subheader("URLs to audit")
url_text = st.sidebar.text_area("One URL per line", placeholder="https://example.com/one\nhttps://example.com/two", height=140)
url_file = st.sidebar.file_uploader("Or upload CSV with a column named 'url'", type=["csv"])

# =============================
# Main actions
# =============================
model = get_model(model_choice)

# Session state
if "expanded_queries" not in st.session_state:
    st.session_state.expanded_queries = []
if "generation_details" not in st.session_state:
    st.session_state.generation_details = None

colA, colB = st.columns(2)
with colA:
    if st.button("Run Fan‑Out 🚀", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a query.")
        else:
            with st.spinner("Generating query fan‑out..."):
                eq, details, err = generate_fanout(model, user_query, mode)
            if err:
                st.error(err)
            elif not eq:
                st.warning("Model returned no queries.")
            else:
                st.success("Query fan‑out complete!")
                st.session_state.expanded_queries = eq
                st.session_state.generation_details = details

with colB:
    if st.button("Audit URLs against Prompts 🧪", use_container_width=True):
        urls = []
        if url_text.strip():
            urls.extend([u.strip() for u in url_text.splitlines() if u.strip()])
        if url_file is not None:
            try:
                dfu = pd.read_csv(url_file)
                if "url" in dfu.columns:
                    urls.extend([str(u).strip() for u in dfu["url"].tolist() if str(u).strip()])
                else:
                    st.error("Uploaded CSV has no 'url' column.")
            except Exception as e:
                st.error(f"Failed to read uploaded CSV: {e}")
        urls = list(dict.fromkeys(urls))  # de‑dupe, preserve order

        if not st.session_state.expanded_queries:
            st.warning("Run the fan‑out first to generate prompts.")
        elif not urls:
            st.warning("Provide at least one URL to audit.")
        else:
            results = audit_urls_against_queries(model, urls, st.session_state.expanded_queries)
            st.session_state.audit_results = results
            st.success("Audit complete!")

# =============================
# Results display
# =============================
if st.session_state.get("generation_details") or st.session_state.get("expanded_queries"):
    st.markdown("---")
    st.subheader("🧠 Model's Query Generation Plan")
    details = st.session_state.get("generation_details") or {}
    generated_count = len(st.session_state.get("expanded_queries") or [])
    target_count_model = details.get("target_query_count", "N/A")
    reasoning_model = details.get("reasoning_for_count", "Not provided by model.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Target #", target_count_model)
    col2.metric("Actual #", generated_count)
    col3.write(f"**Reasoning:** {reasoning_model}")

    with st.expander("View expanded queries"):
        df_eq = pd.DataFrame(st.session_state.expanded_queries)
        st.dataframe(df_eq, use_container_width=True, height=min(600, (len(df_eq)+1)*35))
        st.download_button(
            "📥 Download Queries CSV",
            data=df_eq.to_csv(index=False).encode("utf-8"),
            file_name="qforia_expanded_queries.csv",
            mime="text/csv",
        )

# Audit results
if st.session_state.get("audit_results"):
    st.markdown("---")
    st.subheader("🔎 URL Content Audit vs Fan‑Out Prompts")

    audit = st.session_state.audit_results

    # Flatten top recommendations for a summary table
    flat_rows = []
    for rec in audit:
        if rec.get("error"):
            flat_rows.append({
                "url": rec.get("url"),
                "status": rec.get("status"),
                "title": None,
                "impact": None,
                "effort": None,
                "suggestion": f"ERROR: {rec.get('error')}",
            })
            continue
        url = rec.get("url")
        title = rec.get("title")
        changes = rec.get("recommended_changes") or []
        if not changes:
            flat_rows.append({
                "url": url,
                "status": rec.get("status"),
                "title": title,
                "impact": None,
                "effort": None,
                "suggestion": "(No specific changes returned)",
            })
        else:
            for c in changes[:5]:  # top 5 per page for summary
                flat_rows.append({
                    "url": url,
                    "status": rec.get("status"),
                    "title": title,
                    "impact": c.get("impact"),
                    "effort": c.get("effort"),
                    "suggestion": c.get("suggestion"),
                })

    df_flat = pd.DataFrame(flat_rows)
    st.dataframe(df_flat, use_container_width=True, height=min(600, (len(df_flat)+1)*35))

    # Download full JSON & CSV
    st.download_button(
        "📦 Download full audit (JSON)",
        data=json.dumps(audit, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="qforia_content_audit.json",
        mime="application/json",
    )

    # Per‑URL details
    for rec in audit:
        url = rec.get("url")
        st.markdown("---")
        st.markdown(f"### 🧩 {url}")
        if rec.get("error"):
            st.error(f"{rec.get('error')}")
            continue
        meta_cols = st.columns(4)
        meta_cols[0].metric("Status", rec.get("status"))
        meta_cols[1].metric("Word count", rec.get("word_count"))
        meta_cols[2].write(f"**Title:** {rec.get('title')}")
        meta_cols[3].write(f"**Meta description:** {rec.get('meta_description')}")

        with st.expander("Coverage vs queries"):
            cov = rec.get("coverage") or []
            df_cov = pd.DataFrame(cov)
            st.dataframe(df_cov, use_container_width=True, height=min(400, (len(df_cov)+1)*35))

        with st.expander("Recommended on‑page changes"):
            ch = rec.get("recommended_changes") or []
            df_ch = pd.DataFrame(ch)
            st.dataframe(df_ch, use_container_width=True, height=min(400, (len(df_ch)+1)*35))

        with st.expander("New content ideas / net‑new articles"):
            ideas = rec.get("new_content_ideas") or []
            if ideas:
                for i, idea in enumerate(ideas, start=1):
                    st.markdown(f"**Idea {i}: {idea.get('title')}** — _{idea.get('angle')}_ (Intent: {idea.get('target_intent')})")
                    outline = idea.get("outline") or []
                    if outline:
                        st.markdown("- " + "\n- ".join(outline))
            else:
                st.info("No new content ideas suggested.")

