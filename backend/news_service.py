"""
News service for fetching and analyzing financial news with Groq LLM.
Includes notebook-style ticker flow: fetch -> relevance filter -> per-news summary -> ticker summary.
"""
import json
import logging
import os
import re
import time
import importlib
from hashlib import sha1
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import yfinance as yf

logger = logging.getLogger(__name__)

try:
    from groq import Groq

    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq library not available. Install with: pip install groq")

try:
    chromadb = importlib.import_module("chromadb")
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logger.warning("ChromaDB library not available. Install with: pip install chromadb")


BANK_ALIASES = {
    "BAC": ["bank of america", "bofa", "bac"],
    "JPM": ["jpmorgan", "jpmorgan chase", "jpm"],
    "WFC": ["wells fargo", "wfc"],
    "C": ["citigroup", "citibank", "citi", " c "],
}

OTHER_BIG_BANKS = {
    "BAC": ["jpmorgan", "wells fargo", "citigroup"],
    "JPM": ["bank of america", "wells fargo", "citigroup"],
    "WFC": ["bank of america", "jpmorgan", "citigroup"],
    "C": ["bank of america", "jpmorgan", "wells fargo"],
}

ANALYST_PATTERNS = [
    "upgrades",
    "downgrades",
    "price target",
    "target price",
    "initiates coverage",
    "maintains",
    "analyst",
    "research note",
    "raises target",
    "cuts target",
    "buy call",
    "sell call",
    "overweight",
    "underweight",
    "outperform",
    "underperform",
    "reiterates",
    "rating",
]

POSITIVE_WORDS = {
    "beat",
    "growth",
    "strong",
    "surge",
    "rally",
    "upgrade",
    "outperform",
    "bullish",
    "momentum",
    "expansion",
    "profit",
}

NEGATIVE_WORDS = {
    "miss",
    "weak",
    "downgrade",
    "bearish",
    "decline",
    "drop",
    "plunge",
    "risk",
    "lawsuit",
    "fine",
    "loss",
}

SECTOR_KEYWORDS = {
    "tech": ["ai", "semiconductor", "cloud", "software", "hardware", "chip"],
    "banks": ["credit", "deposit", "loan", "rates", "net interest", "banking"],
    "mining": ["gold", "copper", "silver", "commodity", "ore", "metal"],
}


class NewsService:
    """Service for analyzing financial news with Groq LLM."""

    def __init__(self, api_key: Optional[str] = None, cache_ttl: int = 600, chroma_path: str = "chromadb"):
        self.api_key = api_key
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, Dict] = {}
        self.cache_times: Dict[str, float] = {}
        self.client = None
        self.chroma_path = chroma_path
        self._chroma_client = None
        self._chroma_collection = None

        if GROQ_AVAILABLE and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized successfully")
            except Exception as exc:
                logger.error("Error initializing Groq client: %s", exc)
        elif not GROQ_AVAILABLE:
            logger.warning("Groq library not available - news analysis will return defaults")

        self._init_chroma()

    def _init_chroma(self):
        if not CHROMA_AVAILABLE:
            return

        try:
            os.makedirs(self.chroma_path, exist_ok=True)
            os.environ["ANONYMIZED_TELEMETRY"] = "false"
            # Some posthog versions are incompatible with Chroma's capture signature.
            # Force-disable and no-op capture to avoid startup telemetry errors.
            try:
                posthog_module = importlib.import_module("posthog")
                posthog_module.disabled = True
                posthog_module.capture = lambda *args, **kwargs: None
            except Exception:
                pass
            chroma_settings = chromadb.config.Settings(anonymized_telemetry=False)
            self._chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=chroma_settings,
            )
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="news_item_summaries"
            )
        except Exception as exc:
            self._chroma_client = None
            self._chroma_collection = None
            logger.warning(
                "ChromaDB initialization failed, continuing without persistence: %s",
                exc,
            )

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self.cache_times:
            return False
        age = time.time() - self.cache_times[key]
        return age < self.cache_ttl

    def _get_cached(self, key: str) -> Optional[Dict]:
        if self._is_cache_valid(key):
            return self.cache[key]
        return None

    def _set_cache(self, key: str, value: Dict):
        self.cache[key] = value
        self.cache_times[key] = time.time()

    def _ticker_news_cache_key(
        self,
        ticker: str,
        sector: Optional[str],
        max_items: int,
        bank_name: Optional[str],
    ) -> str:
        ticker_key = (ticker or "").upper().strip()
        sector_key = (sector or "").lower().strip() or "none"
        bank_key = (bank_name or "").upper().strip() or "none"
        safe_max_items = max(1, int(max_items))
        return (
            f"ticker_news_{ticker_key}"
            f"_sector_{sector_key}"
            f"_max_{safe_max_items}"
            f"_bank_{bank_key}"
        )

    def get_safe_default_response(self) -> Dict:
        return {
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "signals": [],
            "summary": "News analysis unavailable",
            "timestamp": datetime.now().isoformat(),
            "status": "unavailable",
        }

    def analyze_news(self, sector: str, news_items: Optional[List[str]] = None) -> Dict:
        """Generic sector-level analysis used by existing endpoint/tests."""
        cache_key = f"news_{sector}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        if not self.client or not news_items:
            result = self.get_safe_default_response()
            result["status"] = "default"
            self._set_cache(cache_key, result)
            return result

        try:
            news_text = "\n".join([f"- {item}" for item in news_items[:5]])
            prompt = f"""
Analyze the following financial news for the {sector} sector.
Provide sentiment (bullish/neutral/bearish) and any trade signals.

News:
{news_text}

Respond ONLY with valid JSON (no markdown, no extra text):
{{
    "sentiment": "bullish|neutral|bearish",
    "sentiment_score": <float -1 to 1>,
    "signals": [<list of identified signals>],
    "summary": "<brief analysis>"
}}
"""
            message = self.client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=500,
            )

            response_text = message.choices[0].message.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            analysis = json.loads(response_text)
            result = {
                "sentiment": analysis.get("sentiment", "neutral").lower(),
                "sentiment_score": float(analysis.get("sentiment_score", 0.0)),
                "signals": analysis.get("signals", []),
                "summary": analysis.get("summary", ""),
                "timestamp": datetime.now().isoformat(),
                "status": "success",
            }
            self._set_cache(cache_key, result)
            return result
        except json.JSONDecodeError:
            result = self.get_safe_default_response()
            result["status"] = "parse_error"
            return result
        except Exception as exc:
            logger.error("Error analyzing news for %s: %s", sector, exc)
            result = self.get_safe_default_response()
            result["status"] = str(type(exc).__name__)
            return result

    def extract_signals(self, sector: str, news_text: Optional[str] = None) -> List[str]:
        if not news_text:
            return []

        signals = []
        text_lower = news_text.lower()

        for keyword in ["surge", "rally", "jump", "soar", "beat", "upgrade", "buy", "strong"]:
            if keyword in text_lower:
                signals.append(f"Bullish signal: {keyword.upper()}")

        for keyword in ["crash", "plunge", "fall", "downgrade", "sell", "weak", "miss"]:
            if keyword in text_lower:
                signals.append(f"Bearish signal: {keyword.upper()}")

        return signals[:5]

    def _normalize_title(self, text: str) -> str:
        cleaned = (text or "").lower().strip()
        cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
        return re.sub(r"\s+", " ", cleaned)

    def _canonicalize_url(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url.strip())
        query = parse_qs(parsed.query)
        clean_query = {
            key: value
            for key, value in query.items()
            if not key.lower().startswith(("utm_", "cmp", "cid", "src", "ref"))
        }
        query_str = urlencode(clean_query, doseq=True)
        return urlunparse(parsed._replace(query=query_str, fragment=""))

    def _fetch_news_from_yf(self, ticker: str, limit: int = 30) -> List[Dict]:
        records = []
        tk = yf.Ticker(ticker)
        items = tk.news or []

        for item in items:
            content = item.get("content", {}) if isinstance(item, dict) else {}
            title = (content.get("title") or item.get("title") or "").strip()
            summary = (content.get("summary") or item.get("summary") or "").strip()

            canonical_url = content.get("canonicalUrl") if isinstance(content.get("canonicalUrl"), dict) else {}
            click_url = content.get("clickThroughUrl") if isinstance(content.get("clickThroughUrl"), dict) else {}
            raw_url = canonical_url.get("url") or click_url.get("url") or item.get("link") or ""

            publish_ts = content.get("pubDate") or item.get("providerPublishTime")
            if isinstance(publish_ts, (int, float)):
                published_dt = datetime.fromtimestamp(publish_ts, tz=timezone.utc)
            else:
                try:
                    published_dt = datetime.fromisoformat(str(publish_ts).replace("Z", "+00:00"))
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    published_dt = datetime.now(timezone.utc)

            records.append(
                {
                    "ticker": ticker,
                    "title": title,
                    "title_norm": self._normalize_title(title),
                    "summary": summary,
                    "url": raw_url,
                    "url_canonical": self._canonicalize_url(raw_url),
                    "publisher": (
                        content.get("provider", {}).get("displayName")
                        if isinstance(content.get("provider"), dict)
                        else item.get("publisher", "")
                    ),
                    "published_at": published_dt.isoformat(),
                }
            )

        records.sort(key=lambda row: row["published_at"], reverse=True)
        return records[:limit]

    def _is_third_party_analyst_call(self, ticker: str, title: str, summary: str) -> bool:
        ticker_upper = ticker.upper()
        aliases = BANK_ALIASES.get(ticker_upper)
        if not aliases:
            return False

        text = f"{title}. {summary}".lower()
        if not any(alias in text for alias in aliases):
            return False
        if not any(pattern in text for pattern in ANALYST_PATTERNS):
            return False

        bank_hits = sum(1 for alias in aliases if alias.strip() and alias in text)
        other_hits = sum(1 for term in OTHER_BIG_BANKS.get(ticker_upper, []) if term in text)
        return other_hits >= max(1, bank_hits)

    def _relevance_score(self, ticker: str, sector: str, title: str, summary: str) -> float:
        text = f"{title}. {summary}".strip()
        if not text:
            return 0.0

        text_low = text.lower()
        score = 0.0

        ticker_token = (ticker or "").strip().lower()
        ticker_pattern = rf"(?<![a-z0-9]){re.escape(ticker_token)}(?![a-z0-9])"
        if ticker_token and re.search(ticker_pattern, text_low):
            score += 0.55

        aliases = BANK_ALIASES.get(ticker.upper(), [])
        if aliases:
            if any(alias in text_low for alias in aliases):
                score += 0.35

        sector_terms = SECTOR_KEYWORDS.get((sector or "").lower(), [])
        if any(term in text_low for term in sector_terms):
            score += 0.15

        if len(text_low) > 250:
            score += 0.05

        return min(score, 1.0)

    def _truncate_chars(self, text: str, max_chars: int) -> str:
        safe = (text or "").strip()
        if len(safe) <= max_chars:
            return safe
        return safe[:max_chars]

    def _clean_llm_output(self, text: str) -> str:
        cleaned = (text or "").strip()
        cleaned = cleaned.replace("```json", "").replace("```text", "").replace("```markdown", "")
        cleaned = cleaned.replace("```", "")
        cleaned = re.sub(r"<think\b[^>]*>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"<think\b[^>]*>.*$", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
        cleaned = re.sub(r"</think>", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.strip()
        return cleaned

    def _looks_like_reasoning_text(self, text: str) -> bool:
        candidate = (text or "").strip().lower()
        if not candidate:
            return False

        lead_in_patterns = [
            r"^let'?s\b",
            r"^first(?:,|:|\s+we\b|\s+i\b|\s+let'?s\b)",
            r"^second(?:,|:|\s+we\b|\s+i\b|\s+let'?s\b)",
            r"^third(?:,|:|\s+we\b|\s+i\b|\s+let'?s\b)",
            r"^analysis(?:\s|:|$)",
            r"^reasoning(?:\s|:|$)",
            r"^step\s*by\s*step\b",
            r"^i\s+(will|can|should|need|am\s+going\s+to)\b",
            r"^we\s+(will|can|should|need|are\s+going\s+to)\b",
            r"^thought\s+process(?:\s|:|$)",
        ]
        if any(re.search(pattern, candidate, flags=re.IGNORECASE) for pattern in lead_in_patterns):
            return True

        reasoning_fragments = [
            "chain of thought",
            "internal reasoning",
            "reasoning process",
            "my reasoning",
            "this analysis",
        ]
        return any(fragment in candidate for fragment in reasoning_fragments)

    def _normalize_item_summary_output(self, output: str, source_text: str) -> str:
        cleaned = self._clean_llm_output(output)
        if not cleaned:
            return self._heuristic_item_summary(source_text)

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) != 3:
            return self._heuristic_item_summary(source_text)

        parsed = []
        for line in lines:
            normalized_line = re.sub(r"^[\-\*\d\.)\s]+", "", line).strip()
            if not normalized_line:
                return self._heuristic_item_summary(source_text)

            screen_line = re.sub(
                r"^(key facts|business/risk impact|tone)\s*:\s*",
                "",
                normalized_line,
                flags=re.IGNORECASE,
            ).strip()
            if self._looks_like_reasoning_text(normalized_line) or self._looks_like_reasoning_text(screen_line):
                return self._heuristic_item_summary(source_text)

            parsed.append(normalized_line)

        key_line = parsed[0] or "Limited provider detail."
        impact_line = parsed[1] or "Watch for effects on earnings, guidance, and near-term volatility."
        tone_line = parsed[2] or "neutral."

        if not key_line.lower().startswith("key facts:"):
            key_line = f"Key facts: {key_line}"
        if not impact_line.lower().startswith("business/risk impact:"):
            impact_line = f"Business/risk impact: {impact_line}"
        if not tone_line.lower().startswith("tone:"):
            tone_line = f"Tone: {tone_line}"

        return f"- {key_line}\n- {impact_line}\n- {tone_line}"

    def _normalize_ticker_summary_output(self, output: str, item_summaries: List[str], ticker: str) -> str:
        cleaned = self._clean_llm_output(output)
        if not cleaned:
            return self._heuristic_ticker_summary(item_summaries, ticker)

        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if not lines:
            return self._heuristic_ticker_summary(item_summaries, ticker)

        labels = ["Overview", "Risks", "Catalysts", "Final signal"]
        normalized = {}
        for line in lines:
            normalized_line = re.sub(r"^[\-\*\d\.)\s]+", "", line).strip()
            if not normalized_line:
                continue

            for label in labels:
                if normalized_line.lower().startswith(label.lower() + ":"):
                    screen_line = re.sub(
                        rf"^{re.escape(label)}\s*:\s*",
                        "",
                        normalized_line,
                        flags=re.IGNORECASE,
                    ).strip()
                    if self._looks_like_reasoning_text(normalized_line) or self._looks_like_reasoning_text(screen_line):
                        return self._heuristic_ticker_summary(item_summaries, ticker)
                    normalized[label] = normalized_line

        if len(normalized) < 4:
            return self._heuristic_ticker_summary(item_summaries, ticker)

        return "\n".join([normalized[label] for label in labels])

    def _news_item_id(self, ticker: str, item: Dict) -> str:
        payload = "|".join(
            [
                (ticker or "").upper().strip(),
                (item.get("title") or "").strip().lower(),
                (item.get("url_canonical") or item.get("url") or "").strip().lower(),
                str(item.get("published_at") or "").strip(),
            ]
        )
        return sha1(payload.encode("utf-8")).hexdigest()

    def _get_persisted_news_summary(self, item_id: str) -> str:
        if self._chroma_collection is None:
            return ""

        try:
            record = self._chroma_collection.get(ids=[item_id])
            documents = record.get("documents") or []
            if documents and documents[0]:
                return self._normalize_item_summary_output(str(documents[0]), "")
            return ""
        except Exception:
            return ""

    def _persist_news_summary(self, item_id: str, summary: str, metadata: Dict[str, str]) -> bool:
        if self._chroma_collection is None:
            return False

        try:
            safe_metadata = {
                "ticker": str(metadata.get("ticker", "")),
                "title": str(metadata.get("title", ""))[:500],
                "published_at": str(metadata.get("published_at", "")),
            }
            self._chroma_collection.upsert(
                ids=[item_id],
                documents=[summary],
                metadatas=[safe_metadata],
            )
            return True
        except Exception:
            return False

    def _groq_chat_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        max_user_chars: int = 2500,
        retry_user_chars: int = 1000,
    ) -> str:
        if self.client is None:
            return ""

        safe_user_prompt = self._truncate_chars(user_prompt, max_user_chars)

        try:
            response = self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": safe_user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=20,
            )
            answer = self._clean_llm_output(response.choices[0].message.content or "")
            if answer:
                return answer

            strict_prompt = self._truncate_chars(
                "Return only the final answer. Do not include analysis or <think> tags.\n\n" + safe_user_prompt,
                retry_user_chars,
            )
            response = self.client.chat.completions.create(
                model="qwen/qwen3-32b",
                messages=[
                    {"role": "system", "content": "Return concise final output only."},
                    {"role": "user", "content": strict_prompt},
                ],
                temperature=0.0,
                max_tokens=max(80, min(max_tokens, 120)),
                timeout=20,
            )
            return self._clean_llm_output(response.choices[0].message.content or "")
        except Exception as exc:
            logger.warning("Groq call failed, using fallback summaries: %s", exc)
            return ""

    def _summarize_news_item(self, text: str, bank_name: str, ticker: str) -> str:
        if not self.client:
            return self._heuristic_item_summary(text)

        system_prompt = "You are a financial analyst. Reply in English with final answer only, no reasoning."
        user_prompt = (
            f"Summarize this news item about {bank_name} ({ticker}) in English.\n"
            "Provide exactly 3 short bullets: (1) key facts, (2) business/risk impact, (3) tone (positive/neutral/negative).\n"
            "Maximum 110 words total.\n\n"
            f"Article text:\n{text[:2400]}"
        )

        output = self._groq_chat_completion(system_prompt, user_prompt, max_tokens=140)
        return self._normalize_item_summary_output(output, text)

    def _summarize_ticker(self, item_summaries: List[str], bank_name: str, ticker: str) -> str:
        if not item_summaries:
            return "No summarizable news."

        if not self.client:
            return self._heuristic_ticker_summary(item_summaries, ticker)

        joined = "\n\n".join([f"- {summary}" for summary in item_summaries])
        system_prompt = "You are a senior financial analyst. Reply in English with final answer only, no reasoning."
        user_prompt = (
            f"You are given news summaries for {bank_name} ({ticker}).\n"
            "Write an executive summary in English with exactly 4 labeled lines: Overview, Risks, Catalysts, Final signal.\n"
            "Maximum 150 words total.\n\n"
            f"Summaries:\n{joined[:3200]}"
        )

        output = self._groq_chat_completion(system_prompt, user_prompt, max_tokens=160)
        return self._normalize_ticker_summary_output(output, item_summaries, ticker)

    def _heuristic_item_summary(self, text: str) -> str:
        snippet = re.sub(r"\s+", " ", (text or "").strip())
        snippet = snippet[:280] if len(snippet) > 280 else snippet
        tone = "neutral"
        low = snippet.lower()
        pos_hits = sum(1 for word in POSITIVE_WORDS if word in low)
        neg_hits = sum(1 for word in NEGATIVE_WORDS if word in low)
        if pos_hits > neg_hits:
            tone = "positive"
        elif neg_hits > pos_hits:
            tone = "negative"

        return (
            f"- Key facts: {snippet or 'Limited provider detail.'}\n"
            f"- Business/risk impact: Watch for effects on earnings, guidance, and near-term volatility.\n"
            f"- Tone: {tone}."
        )

    def _heuristic_ticker_summary(self, item_summaries: List[str], ticker: str) -> str:
        combined = "\n".join(item_summaries)
        score = self._estimate_sentiment_score(combined)
        signal = "Neutral"
        if score > 0.15:
            signal = "Constructive/Bullish"
        elif score < -0.15:
            signal = "Defensive/Bearish"

        return (
            f"Overview: Recent headlines for {ticker} show mixed catalysts and risks.\n"
            "Risks: Macro uncertainty, guidance changes, and regulatory headlines can pressure positioning.\n"
            "Catalysts: Earnings prints, analyst revisions, and sector momentum remain key drivers.\n"
            f"Final signal: {signal}."
        )

    def _estimate_sentiment_score(self, text: str) -> float:
        low = (text or "").lower()
        pos_hits = sum(1 for word in POSITIVE_WORDS if word in low)
        neg_hits = sum(1 for word in NEGATIVE_WORDS if word in low)
        total = pos_hits + neg_hits
        if total == 0:
            return 0.0
        return float(max(-1.0, min(1.0, (pos_hits - neg_hits) / total)))

    def analyze_ticker_news(
        self,
        ticker: str,
        sector: Optional[str] = None,
        bank_name: Optional[str] = None,
        max_items: int = 10,
    ) -> Dict:
        ticker_upper = ticker.upper().strip()
        cache_key = self._ticker_news_cache_key(
            ticker=ticker_upper,
            sector=sector,
            max_items=max_items,
            bank_name=bank_name,
        )
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            fetched = self._fetch_news_from_yf(ticker_upper, limit=max(15, max_items * 2))

            filtered = []
            for item in fetched:
                if self._is_third_party_analyst_call(ticker_upper, item["title"], item["summary"]):
                    continue

                rel_score = self._relevance_score(
                    ticker=ticker_upper,
                    sector=(sector or ""),
                    title=item["title"],
                    summary=item["summary"],
                )
                if rel_score >= 0.35:
                    item["relevance_score"] = round(rel_score, 3)
                    filtered.append(item)

            retained = filtered[:max_items]
            if not retained and fetched:
                retained = fetched[: min(3, len(fetched))]
                for row in retained:
                    row["relevance_score"] = 0.2

            news_summaries = []
            detailed_rows = []
            persisted_hits = 0
            generated_hits = 0
            for item in retained:
                item_id = self._news_item_id(ticker_upper, item)
                full_text = f"{item.get('title', '')}\n\n{item.get('summary', '')}".strip()
                summary = self._get_persisted_news_summary(item_id)
                if summary:
                    persisted_hits += 1
                else:
                    summary = self._summarize_news_item(
                        text=full_text,
                        bank_name=bank_name or ticker_upper,
                        ticker=ticker_upper,
                    )
                    summary = self._normalize_item_summary_output(summary, full_text)
                    self._persist_news_summary(
                        item_id,
                        summary,
                        {
                            "ticker": ticker_upper,
                            "title": item.get("title", ""),
                            "published_at": item.get("published_at", ""),
                        },
                    )
                    generated_hits += 1

                news_summaries.append(summary)
                detailed_rows.append(
                    {
                        "title": item.get("title", ""),
                        "published_at": item.get("published_at"),
                        "source": item.get("publisher", ""),
                        "relevance_score": float(item.get("relevance_score", 0.0)),
                        "summary_en": summary,
                        "summary_es": summary,
                    }
                )

            bank_summary = self._summarize_ticker(
                item_summaries=news_summaries,
                bank_name=bank_name or ticker_upper,
                ticker=ticker_upper,
            )

            sentiment_score = self._estimate_sentiment_score("\n".join(news_summaries + [bank_summary]))
            sentiment = "neutral"
            if sentiment_score > 0.15:
                sentiment = "bullish"
            elif sentiment_score < -0.15:
                sentiment = "bearish"

            key_signals = self.extract_signals(sector or "unknown", bank_summary)

            result = {
                "ticker": ticker_upper,
                "sector": sector,
                "bank_name": bank_name or ticker_upper,
                "metrics": {
                    "fetched": len(fetched),
                    "filtered": len(filtered),
                    "retained": len(retained),
                    "persisted_reused": persisted_hits,
                    "summaries_generated": generated_hits,
                },
                "news_summaries": detailed_rows,
                "bank_summary_en": self._normalize_ticker_summary_output(bank_summary, news_summaries, ticker_upper),
                "bank_summary_es": self._normalize_ticker_summary_output(bank_summary, news_summaries, ticker_upper),
                "sentiment": sentiment,
                "sentiment_score": float(round(sentiment_score, 4)),
                "summary": self._normalize_ticker_summary_output(bank_summary, news_summaries, ticker_upper),
                "key_signals": key_signals,
                "status": "success" if self.client else "success_fallback",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            self._set_cache(cache_key, result)
            return result
        except Exception as exc:
            logger.error("Ticker news analysis failed for %s: %s", ticker_upper, exc)
            fallback = {
                "ticker": ticker_upper,
                "sector": sector,
                "bank_name": bank_name or ticker_upper,
                "metrics": {
                    "fetched": 0,
                    "filtered": 0,
                    "retained": 0,
                },
                "news_summaries": [],
                "bank_summary_en": "News analysis unavailable",
                "bank_summary_es": "News analysis unavailable",
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "summary": "News analysis unavailable",
                "key_signals": [],
                "status": "error",
                "error": str(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            return fallback

    def get_news_status(self) -> Dict:
        return {
            "groq_available": GROQ_AVAILABLE,
            "client_ready": self.client is not None,
            "chromadb_available": CHROMA_AVAILABLE,
            "chromadb_ready": self._chroma_collection is not None,
            "cache_size": len(self.cache),
            "timestamp": datetime.now().isoformat(),
        }



def create_news_service(api_key: Optional[str] = None, cache_ttl: int = 600, chroma_path: str = "chromadb") -> NewsService:
    """Factory function to create NewsService instance."""
    return NewsService(api_key=api_key, cache_ttl=cache_ttl, chroma_path=chroma_path)
