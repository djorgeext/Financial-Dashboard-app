# %%
# Colab setup
!pip -q install yfinance trafilatura chromadb sentence-transformers transformers accelerate huggingface_hub openai

# Optional: install llama-cpp-python only if you prefer GGUF local inference instead of Transformers
# !pip -q install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --no-cache-dir

# %%
import re
import json
import time
import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import yfinance as yf
import trafilatura
import chromadb
from openai import OpenAI, BadRequestError

from sentence_transformers import CrossEncoder
from transformers import pipeline

# Mount Drive in Colab
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
except Exception as e:
    print(f"Drive mount skipped: {e}")

# Global config
BANKS = {
    "BAC": "Bank of America",
    "JPM": "JPMorgan Chase",
    "WFC": "Wells Fargo",
    "C": "Citigroup",
}

TARGET_NEWS_PER_BANK = 10
CHROMA_PATH = "/content/drive/MyDrive/Challenges_ML-DL/chromadb"
REGISTRY_DIR = "/content/drive/MyDrive/Challenges_ML-DL/chromadb/news_registry"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
MIN_EXTRACTED_CHARS = 500

# Groq API config (OpenAI-compatible client)
GROQ_MODEL_ID = "qwen/qwen3-32b"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
GROQ_MAX_USER_CHARS = 2500
GROQ_RETRY_USER_CHARS = 1000
# Recomendado en Colab/Jupyter: usar getpass y variable de entorno (sin hardcodear la key)
# from getpass import getpass; os.environ[GROQ_API_KEY_ENV] = getpass("Groq API Key: " ).strip()
if not os.getenv(GROQ_API_KEY_ENV, "").strip():
    try:
        from getpass import getpass
        os.environ[GROQ_API_KEY_ENV] = getpass("Groq API Key: " ).strip()
    except Exception:
        pass
GROQ_API_KEY = os.getenv(GROQ_API_KEY_ENV, "").strip()

Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
Path(REGISTRY_DIR).mkdir(parents=True, exist_ok=True)

print("Config ready.")

# %%
# Utilities: URL normalization, scraping, filtering, chunking, Chroma persistence

NOISE_PATTERNS = [
    "accept cookies",
    "cookie policy",
    "your privacy choices",
    "consent",
    "gdpr",
    "we use cookies",
    "do not sell or share",
]

OTHER_BIG_BANKS = {
    "BAC": ["jpmorgan", "wells fargo", "citigroup"],
    "JPM": ["bank of america", "wells fargo", "citigroup"],
    "WFC": ["bank of america", "jpmorgan", "citigroup"],
    "C": ["bank of america", "jpmorgan", "wells fargo"],
}

BANK_ALIASES = {
    "BAC": ["bank of america", "bofa", "bac"],
    "JPM": ["jpmorgan", "jpmorgan chase", "jpm"],
    "WFC": ["wells fargo", "wfc"],
    "C": ["citigroup", "citibank", "citi", " c "],
}

ANALYST_PATTERNS = [
    "upgrades", "downgrades", "price target", "target price", "initiates coverage",
    "maintains", "analyst", "research note", "raises target", "cuts target",
    "buy call", "sell call", "overweight", "underweight", "equal weight",
    "outperform", "underperform", "reiterates", "rating", "pt",
]

THIRD_PARTY_CUE_PATTERNS = [
    r"\bon\s+([A-Z][A-Za-z0-9&'\.-]{1,}(?:\s+[A-Z][A-Za-z0-9&'\.-]{1,}){0,3})",
    r"\bat\s+([A-Z][A-Za-z0-9&'\.-]{1,}(?:\s+[A-Z][A-Za-z0-9&'\.-]{1,}){0,3})",
    r"\bfor\s+([A-Z][A-Za-z0-9&'\.-]{1,}(?:\s+[A-Z][A-Za-z0-9&'\.-]{1,}){0,3})",
]

RELEVANCE_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
AMBIGUITY_MODEL_ID = "facebook/bart-large-mnli"

# Score band where cross-encoder is uncertain and requires a second-stage decision.
AMBIGUOUS_SCORE_LOW = -0.2
AMBIGUOUS_SCORE_HIGH = 0.4
AMBIGUOUS_CLASSIFIER_MIN_PROB = 0.45


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_title(title: str) -> str:
    text = (title or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    # Remove tracking params but keep content-identifying params when useful.
    query = parse_qs(parsed.query)
    clean_query = {
        k: v for k, v in query.items()
        if not k.lower().startswith(("utm_", "cmp", "cid", "guccounter", "src", "ref"))
    }
    query_str = urlencode(clean_query, doseq=True)
    clean = parsed._replace(query=query_str, fragment="")
    return urlunparse(clean)


def news_id(ticker: str, canonical_url: str, title_norm: str) -> str:
    base = f"{ticker}|{canonical_url}|{title_norm}".encode("utf-8")
    return hashlib.sha1(base).hexdigest()


def fetch_news_from_yf(ticker: str):
    tk = yf.Ticker(ticker)
    items = tk.news or []
    records = []
    for item in items:
        content = item.get("content", {}) if isinstance(item, dict) else {}
        title = (content.get("title") or item.get("title") or "").strip()
        summary = (content.get("summary") or item.get("summary") or "").strip()

        canonical_url = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
        click_url = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}

        link = canonical_url.get("url") or click_url.get("url") or item.get("link") or ""

        publish_ts = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(publish_ts, (int, float)):
            publish_dt = datetime.fromtimestamp(publish_ts, tz=timezone.utc)
        else:
            try:
                publish_dt = datetime.fromisoformat(str(publish_ts).replace("Z", "+00:00"))
                if publish_dt.tzinfo is None:
                    publish_dt = publish_dt.replace(tzinfo=timezone.utc)
            except Exception:
                publish_dt = datetime.now(timezone.utc)

        records.append({
            "ticker": ticker,
            "bank_name": BANKS[ticker],
            "title": title,
            "title_norm": normalize_title(title),
            "summary": summary,
            "publisher": content.get("provider", {}).get("displayName") if isinstance(content.get("provider"), dict) else item.get("publisher", ""),
            "url": link,
            "url_canonical": canonicalize_url(link),
            "published_at": publish_dt.isoformat(),
            "downloaded_at": now_utc_iso(),
        })

    # newest first and keep extra candidates before strict filtering
    records = sorted(records, key=lambda x: x["published_at"], reverse=True)
    return records[:30]


def looks_like_cookie_noise(text: str) -> bool:
    low = (text or "").lower()
    hits = sum(1 for p in NOISE_PATTERNS if p in low)
    return hits >= 2


def scrape_article_text(url: str, max_retries: int = 3, pause: float = 1.0):
    if not url:
        return "", "missing_url"

    last_error = "unknown"
    for attempt in range(max_retries):
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                last_error = "empty_download"
            else:
                text = trafilatura.extract(
                    downloaded,
                    include_comments=False,
                    include_tables=False,
                    favor_precision=True,
                )
                text = (text or "").strip()
                if len(text) < MIN_EXTRACTED_CHARS:
                    last_error = "too_short"
                elif looks_like_cookie_noise(text):
                    last_error = "cookie_noise"
                else:
                    return text, "ok"
        except Exception as e:
            last_error = f"error:{type(e).__name__}"

        if attempt < max_retries - 1:
            time.sleep(pause * (2 ** attempt))

    return "", last_error


def has_analyst_language(text_low: str) -> bool:
    return any(p in text_low for p in ANALYST_PATTERNS)


def extract_third_party_targets(raw_text: str):
    candidates = []
    for pattern in THIRD_PARTY_CUE_PATTERNS:
        matches = re.findall(pattern, raw_text)
        for m in matches:
            name = (m or "").strip(" .,:;()[]{}\"'")
            if not name:
                continue
            candidates.append(name)
    return candidates


def is_third_party_analyst_call(ticker: str, title: str, summary: str) -> bool:
    raw_text = f"{title}. {summary}".strip()
    text_low = raw_text.lower()
    aliases = BANK_ALIASES[ticker]

    has_bank_mention = any(a.strip() and a in text_low for a in aliases)
    if not has_bank_mention:
        return False

    if not has_analyst_language(text_low):
        return False

    bank_name_low = BANKS[ticker].lower()
    for company in extract_third_party_targets(raw_text):
        company_low = company.lower()
        if company_low in bank_name_low:
            continue
        if any(a.strip() and a.strip() in company_low for a in aliases):
            continue
        # Exclude if the target appears as clearly non-bank company in analyst context.
        return True

    # Extra guard: allow direct exclusion for frequent "BofA on/at/for <ticker/name>" style.
    explicit_third_party_tickers = re.findall(r"\b(?:on|at|for)\s+\(([A-Z]{1,5})\)", raw_text)
    if explicit_third_party_tickers:
        return True

    return False


def build_ambiguity_classifier(model_id: str = AMBIGUITY_MODEL_ID):
    return pipeline("zero-shot-classification", model=model_id)


def classify_ambiguous_news(ticker: str, text: str, classifier, min_prob: float = AMBIGUOUS_CLASSIFIER_MIN_PROB) -> bool:
    bank = BANKS[ticker]
    labels = [
        f"news primarily about {bank} ({ticker}) business results, strategy, regulation, risk or operations",
        f"analyst recommendation by {bank} ({ticker}) about another company",
        "general market or macro news not centered on the bank",
    ]

    result = classifier(
        text[:2200],
        candidate_labels=labels,
        hypothesis_template="This text is {}.",
        multi_label=False,
    )

    top_label = result["labels"][0]
    top_score = float(result["scores"][0])
    return top_label == labels[0] and top_score >= min_prob


def is_relevant_news(
    ticker: str,
    title: str,
    summary: str,
    relevance_model: CrossEncoder,
    ambiguity_classifier=None,
    ambiguity_low: float = AMBIGUOUS_SCORE_LOW,
    ambiguity_high: float = AMBIGUOUS_SCORE_HIGH,
) -> bool:
    bank = BANKS[ticker]
    text = f"{title}. {summary}".strip()
    if not text:
        return False

    # Stage 1: high-precision exclusion for analyst notes focused on third parties.
    if is_third_party_analyst_call(ticker, title, summary):
        return False

    text_low = text.lower()
    query = f"This article is primarily about {bank} ({ticker}) financial actions and business results."
    score = float(relevance_model.predict([(query, text)])[0])

    bank_hits = sum(1 for term in [bank.lower(), ticker.lower()] if term in text_low)
    other_hits = sum(1 for term in OTHER_BIG_BANKS[ticker] if term in text_low)
    analyst_style = has_analyst_language(text_low)

    if bank_hits == 0:
        return False
    if analyst_style and other_hits >= bank_hits:
        return False

    # Stage 2: only ambiguous scores go to zero-shot classifier.
    if score <= ambiguity_low:
        return False
    if score >= ambiguity_high:
        return True

    if ambiguity_classifier is not None:
        return classify_ambiguous_news(ticker, text, ambiguity_classifier)

    return False


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    cleaned = re.sub(r"\n{3,}", "\n\n", (text or "").strip())
    if not cleaned:
        return []

    paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = f"{current}\n\n{para}".strip()
            continue

        if current:
            chunks.append(current)

        # If paragraph is too long, split inside paragraph with overlap.
        if len(para) > chunk_size:
            start = 0
            while start < len(para):
                end = min(start + chunk_size, len(para))
                chunks.append(para[start:end])
                if end >= len(para):
                    break
                start = max(0, end - overlap)
            current = ""
        else:
            current = para

    if current:
        chunks.append(current)

    # Remove near-empty and exact duplicate chunks while preserving order.
    final_chunks = []
    seen = set()
    for ch in chunks:
        c = ch.strip()
        if len(c) < 80:
            continue
        if c in seen:
            continue
        seen.add(c)
        final_chunks.append(c)
    return final_chunks


def get_registry_path(ticker: str) -> Path:
    return Path(REGISTRY_DIR) / f"{ticker.lower()}_registry.json"


def load_registry(ticker: str):
    p = get_registry_path(ticker)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_registry(ticker: str, data: dict):
    p = get_registry_path(ticker)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_collection(client, ticker: str):
    return client.get_or_create_collection(
        name=f"news_{ticker.lower()}",
        metadata={"ticker": ticker, "bank_name": BANKS[ticker]},
    )


def upsert_news_to_chroma(client, ticker: str, news_item: dict):
    collection = get_collection(client, ticker)
    nid = news_item["news_id"]
    chunks = chunk_text(news_item["article_text"])
    if not chunks:
        return 0

    ids, docs, metas = [], [], []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        cid = f"{nid}::chunk::{idx:03d}"
        ids.append(cid)
        docs.append(chunk)
        metas.append({
            "ticker": ticker,
            "bank_name": news_item["bank_name"],
            "news_id": nid,
            "batch_id": nid,
            "chunk_index": idx,
            "total_chunks": total,
            "published_at": news_item["published_at"],
            "source": news_item.get("publisher", ""),
            "url": news_item["url"],
            "url_canonical": news_item["url_canonical"],
            "title": news_item["title"],
            "title_norm": news_item["title_norm"],
            "downloaded_at": news_item["downloaded_at"],
            "scrape_status": news_item.get("scrape_status", "ok"),
            "fallback_used": news_item.get("fallback_used", False),
        })

    collection.upsert(ids=ids, documents=docs, metadatas=metas)
    return total


def delete_news_from_chroma(client, ticker: str, news_ids_to_delete):
    if not news_ids_to_delete:
        return
    collection = get_collection(client, ticker)
    ids = []
    for nid in news_ids_to_delete:
        # We do not know exact chunk count, so query by metadata and delete matching IDs.
        found = collection.get(where={"news_id": nid}, include=[])
        ids.extend(found.get("ids", []))
    if ids:
        collection.delete(ids=ids)


def enforce_retention(client, ticker: str, registry: dict, max_items: int = TARGET_NEWS_PER_BANK):
    items = list(registry.values())
    items.sort(key=lambda x: x["published_at"], reverse=True)

    keep = items[:max_items]
    drop = items[max_items:]

    drop_ids = [it["news_id"] for it in drop]
    delete_news_from_chroma(client, ticker, drop_ids)

    final = {it["news_id"]: it for it in keep}
    return final, len(drop)


def build_groq_client(api_key: str = "", base_url: str = GROQ_BASE_URL):
    resolved_api_key = (api_key or os.getenv(GROQ_API_KEY_ENV, "") or GROQ_API_KEY).strip()
    if not resolved_api_key:
        raise ValueError(
            f"Missing Groq API key. Set environment variable {GROQ_API_KEY_ENV}."
        )
    return OpenAI(api_key=resolved_api_key, base_url=base_url)


def _truncate_chars(text: str, max_chars: int):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _clean_llm_output(text: str) -> str:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    # If the model only returned an unfinished reasoning block, force a retry path.
    if cleaned.lower().startswith("<think>"):
        return ""
    return cleaned


def _groq_chat_completion(
    client,
    system_prompt: str,
    user_prompt: str,
    model_id: str = GROQ_MODEL_ID,
    max_tokens: int = 180,
    max_user_chars: int = GROQ_MAX_USER_CHARS,
    retry_user_chars: int = GROQ_RETRY_USER_CHARS,
):
    safe_user_prompt = _truncate_chars(user_prompt, max_user_chars)
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": safe_user_prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        answer = _clean_llm_output(response.choices[0].message.content or "")
        if answer:
            return answer

        # Strict fallback when models leak reasoning or empty output.
        strict_prompt = _truncate_chars(
            "Return only the final answer. Do not include analysis, thinking, or <think> tags.\n\n" + safe_user_prompt,
            retry_user_chars,
        )
        retry_tokens = max(80, min(max_tokens, 120))
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "Return concise final output only."},
                {"role": "user", "content": strict_prompt},
            ],
            temperature=0.0,
            max_tokens=retry_tokens,
        )
        return _clean_llm_output(response.choices[0].message.content or "")
    except BadRequestError as e:
        if "reduce the length of the messages or completion" not in str(e).lower():
            raise

        retry_prompt = _truncate_chars(safe_user_prompt, retry_user_chars)
        retry_tokens = max(80, min(max_tokens, 120))
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": retry_prompt},
            ],
            temperature=0.0,
            max_tokens=retry_tokens,
        )
        return _clean_llm_output(response.choices[0].message.content or "")


def summarize_news(text: str, bank_name: str, ticker: str, client, model_id: str = GROQ_MODEL_ID, max_tokens: int = 140):
    system_prompt = (
        "You are a financial analyst. Reply in English with final answer only, no reasoning."
    )
    user_prompt = (
        f"Summarize this news item about {bank_name} ({ticker}) in English.\n"
        "Provide exactly 3 short bullets: (1) key facts, (2) business/risk impact, (3) tone (positive/neutral/negative).\n"
        "Maximum 110 words total.\n\n"
        f"Article text:\n{text[:2400]}"
    )
    return _groq_chat_completion(client, system_prompt, user_prompt, model_id=model_id, max_tokens=max_tokens)


def summarize_bank(news_summaries, bank_name: str, ticker: str, client, model_id: str = GROQ_MODEL_ID, max_tokens: int = 160):
    joined = "\n\n".join([f"- {s}" for s in news_summaries])
    system_prompt = (
        "You are a senior financial analyst. Reply in English with final answer only, no reasoning."
    )
    user_prompt = (
        f"You are given news summaries for {bank_name} ({ticker}).\n"
        "Write an executive summary in English with exactly 4 labeled lines: Overview, Risks, Catalysts, Final signal.\n"
        "Maximum 150 words total.\n\n"
        f"Summaries:\n{joined[:3200]}"
    )
    return _groq_chat_completion(client, system_prompt, user_prompt, model_id=model_id, max_tokens=max_tokens)


print("Utilities loaded.")

# %%
# Orchestrator: fetch -> filter -> scrape -> store -> dedupe/retain -> summarize

client = chromadb.PersistentClient(path=CHROMA_PATH)
relevance_model = CrossEncoder(RELEVANCE_MODEL_ID)
ambiguity_classifier = build_ambiguity_classifier(AMBIGUITY_MODEL_ID)
if not os.getenv(GROQ_API_KEY_ENV, "").strip() and not GROQ_API_KEY:
    raise ValueError(f"Set {GROQ_API_KEY_ENV} before running this cell.")
groq_client = build_groq_client()

all_outputs = {}

for ticker, bank_name in BANKS.items():
    print("\n" + "=" * 80)
    print(f"Processing {ticker} - {bank_name}")

    registry = load_registry(ticker)
    fetched = fetch_news_from_yf(ticker)

    # Relevance filtering
    filtered = []
    for item in fetched:
        if is_relevant_news(
            ticker,
            item["title"],
            item["summary"],
            relevance_model,
            ambiguity_classifier=ambiguity_classifier,
        ):
            filtered.append(item)

    # Scraping and dedupe (against existing + within current pull)
    inserted_or_updated = 0
    dedup_skipped = 0
    scrape_failed = 0
    fallback_used = 0

    for item in filtered:
        nid = news_id(ticker, item["url_canonical"], item["title_norm"])
        item["news_id"] = nid

        # Skip exact duplicates already tracked.
        if nid in registry:
            dedup_skipped += 1
            continue

        article_text, status = scrape_article_text(item["url"])
        item["scrape_status"] = status
        item["fallback_used"] = False

        if status != "ok":
            # Fallback for cookie walls / blocked pages: keep provider summary so the news is still processable.
            fallback_text = f"{item['title']}\n\n{item['summary']}".strip()
            if len(fallback_text) >= 150:
                article_text = fallback_text
                item["scrape_status"] = f"{status}|fallback_summary"
                item["fallback_used"] = True
                fallback_used += 1
            else:
                scrape_failed += 1
                continue

        item["article_text"] = article_text

        chunk_count = upsert_news_to_chroma(client, ticker, item)
        if chunk_count == 0:
            continue

        registry[nid] = {
            "news_id": nid,
            "ticker": ticker,
            "bank_name": bank_name,
            "title": item["title"],
            "title_norm": item["title_norm"],
            "url": item["url"],
            "url_canonical": item["url_canonical"],
            "publisher": item["publisher"],
            "published_at": item["published_at"],
            "downloaded_at": item["downloaded_at"],
            "scrape_status": item["scrape_status"],
            "fallback_used": item["fallback_used"],
            "chunk_count": chunk_count,
        }
        inserted_or_updated += 1

    # Enforce top-10 recency retention per bank
    registry, removed_old = enforce_retention(client, ticker, registry, TARGET_NEWS_PER_BANK)
    save_registry(ticker, registry)

    # Build per-news and per-bank summaries from retained news
    collection = get_collection(client, ticker)
    retained_items = sorted(registry.values(), key=lambda x: x["published_at"], reverse=True)

    news_summaries = []
    detailed_rows = []
    for meta in retained_items:
        result = collection.get(where={"news_id": meta["news_id"]}, include=["documents", "metadatas"])
        docs = result.get("documents", [])
        if not docs:
            continue

        docs_with_idx = list(zip(docs, result.get("metadatas", [])))
        docs_with_idx.sort(key=lambda x: int(x[1].get("chunk_index", 0)))
        full_text = "\n\n".join([d for d, _ in docs_with_idx])

        summary = summarize_news(full_text, bank_name, ticker, groq_client)
        news_summaries.append(summary)

        detailed_rows.append({
            "news_id": meta["news_id"],
            "title": meta["title"],
            "published_at": meta["published_at"],
            "scrape_status": meta.get("scrape_status", "unknown"),
            "fallback_used": meta.get("fallback_used", False),
            "summary_en": summary,
            "summary_es": summary,
        })

    bank_summary = summarize_bank(news_summaries, bank_name, ticker, groq_client) if news_summaries else "No summarizable news."

    all_outputs[ticker] = {
        "bank_name": bank_name,
        "metrics": {
            "fetched": len(fetched),
            "filtered": len(filtered),
            "inserted_or_updated": inserted_or_updated,
            "dedup_skipped": dedup_skipped,
            "scrape_failed": scrape_failed,
            "fallback_used": fallback_used,
            "removed_old": removed_old,
            "retained": len(registry),
        },
        "news_summaries": detailed_rows,
        "bank_summary_en": bank_summary,
        "bank_summary_es": bank_summary,
    }

    print("Metrics:", all_outputs[ticker]["metrics"])
    print("Bank summary:\n", bank_summary)

# Persist final report
report_path = Path("/content/drive/MyDrive/Challenges_ML-DL/chromadb/news_pipeline_report.json")
with report_path.open("w", encoding="utf-8") as f:
    json.dump(all_outputs, f, ensure_ascii=False, indent=2)

print("\nPipeline complete. Report saved to:", report_path)


