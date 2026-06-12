from __future__ import annotations

import gzip
import json
import re
import urllib.request

from trading_agent.data.http_utils import fetch_with_retry
from datetime import datetime, timezone
from typing import Any

from trading_agent.domain.evidence import EvidenceItem, SourceType

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
# EDGAR "current events": the newest 8-Ks across ALL companies, in one Atom
# request. count=400 is the endpoint's maximum window -- around earnings
# season the 8-K firehose can push hundreds of filings in a burst, and a
# too-small window would silently miss the very catalysts being hunted.
CURRENT_8K_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&company=&dateb=&owner=include&count=400&output=atom"
)

# SEC form -> our evidence source type. Forms not listed here are skipped so the
# committee only ever sees catalyst- or contradiction-relevant filings.
_FORM_MAP: dict[str, SourceType] = {
    "8-K": "sec_8k",
    "10-Q": "sec_10q",
    "10-K": "sec_10k",
    "S-3": "sec_s3",
    "S-3/A": "sec_s3",
    "SC 13D": "sec_13d",
    "SC 13D/A": "sec_13d",
    "SC 13G": "sec_13g",
    "SC 13G/A": "sec_13g",
    "4": "sec_form4",
}


class SecEdgarError(RuntimeError):
    """Raised when EDGAR is unreachable or returns an unexpected payload."""


def map_form_to_source_type(form: str) -> SourceType | None:
    form = (form or "").strip()
    if form in _FORM_MAP:
        return _FORM_MAP[form]
    # Prospectus forms are 424B1..424B5 etc; treat the whole family as 424B.
    if form.startswith("424B"):
        return "sec_424b"
    return None


# 8-K item code -> (human label, directional lean). The committee reads the
# label as observed_fact, and the leans let the deterministic layer reason about
# direction: an 8-K is not bullish by default -- item 4.02 (financials can't be
# trusted) or 3.01 (delisting notice) is exactly the kind of catalyst that wants
# a long PUT, while 1.01/2.01 (a signed deal / a closed acquisition) wants a
# call. "earnings" marks the IV-crush risk window (item 2.02 IS the earnings 8-K,
# so this is what finally fires the earnings_iv_crush red flag without a separate
# calendar feed). Lean is one of: bull, bear, earnings, neutral.
_EIGHT_K_ITEMS: dict[str, tuple[str, str]] = {
    "1.01": ("Entry into Material Definitive Agreement", "bull"),
    "1.02": ("Termination of Material Definitive Agreement", "bear"),
    "1.03": ("Bankruptcy or Receivership", "bear"),
    "1.05": ("Material Cybersecurity Incident", "bear"),
    "2.01": ("Completion of Acquisition or Disposition", "bull"),
    "2.02": ("Results of Operations (earnings)", "earnings"),
    "2.03": ("Creation of a Material Direct Financial Obligation", "bear"),
    "2.04": ("Triggering Events That Accelerate an Obligation", "bear"),
    "2.05": ("Costs Associated with Exit or Disposal", "bear"),
    "2.06": ("Material Impairments", "bear"),
    "3.01": ("Notice of Delisting or Failure to Satisfy Listing Rule", "bear"),
    "3.02": ("Unregistered Sales of Equity Securities (dilution)", "bear"),
    "3.03": ("Material Modification to Rights of Security Holders", "bear"),
    "4.01": ("Changes in Registrant's Certifying Accountant", "bear"),
    "4.02": ("Non-Reliance on Previously Issued Financials", "bear"),
    "5.02": ("Departure/Election of Directors or Officers", "neutral"),
    "5.07": ("Submission of Matters to a Vote of Security Holders", "neutral"),
    "7.01": ("Regulation FD Disclosure", "neutral"),
    "8.01": ("Other Events", "neutral"),
    "9.01": ("Financial Statements and Exhibits", "neutral"),
}


def label_8k_items(report_items: str | None) -> tuple[str, set[str]]:
    """Map an 8-K ``items`` string ('2.02,9.01') to a label and its leans.

    Returns ``(label, leans)`` where label is a human-readable join used in the
    evidence fact and leans is the set of directional tags ({'earnings'},
    {'bull'}, ...) the deterministic layer keys off. Unknown codes are surfaced
    verbatim with a 'neutral' lean so a new item type is never silently dropped.
    """

    codes = [c.strip() for c in (report_items or "").replace(";", ",").split(",")]
    codes = [c for c in codes if c]
    if not codes:
        return "", set()
    labels: list[str] = []
    leans: set[str] = set()
    for code in codes:
        label, lean = _EIGHT_K_ITEMS.get(code, (f"item {code}", "neutral"))
        labels.append(f"{code} {label}" if code not in label else label)
        leans.add(lean)
    return "; ".join(labels), leans


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_SCRIPT_STYLE_RE = re.compile(
    r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL
)


def strip_html(html: str) -> str:
    """Plain-text body from an SEC HTML filing (no parser dependency).

    Drops script/style blocks, removes tags, unescapes entities, and collapses
    whitespace. Good enough to feed catalyst keywords to the committee; not a
    faithful render. XBRL/inline tags become spaces so words stay separated.
    """

    import html as _html

    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    return _WS_RE.sub(" ", text).strip()


def parse_current_filing_ciks(atom_xml: str) -> set[int]:
    """CIK numbers from an EDGAR current-events Atom feed.

    Entry titles look like ``8-K - Acme Corp (0001234567) (Filer)``: the only
    parenthesized 10-digit runs in the document are CIKs (accession numbers are
    dash-separated), so a regex beats a full XML parse over a feed whose markup
    EDGAR does not guarantee to stay namespace-stable.
    """

    return {int(m) for m in re.findall(r"\((\d{10})\)", atom_xml)}


def build_filing_url(cik: int | str, accession_number: str, primary_document: str) -> str:
    accession = accession_number.replace("-", "")
    return ARCHIVES_URL.format(
        cik=int(cik), accession=accession, document=primary_document
    )


def _parse_published_at(acceptance: str | None, filing_date: str | None) -> datetime:
    if acceptance:
        return datetime.fromisoformat(acceptance.replace("Z", "+00:00"))
    if filing_date:
        return datetime.fromisoformat(f"{filing_date}T00:00:00+00:00")
    raise SecEdgarError("filing is missing both acceptanceDateTime and filingDate")


def parse_submissions(
    submissions: dict[str, Any],
    *,
    ticker: str,
    cik: int | str,
    forms: list[str] | None = None,
    limit: int = 20,
    retrieved_at: datetime | None = None,
) -> list[EvidenceItem]:
    """Turn an EDGAR submissions payload into traceable evidence rows.

    Pure: takes the already-fetched JSON so it can be tested against a fixture.
    """

    retrieved = retrieved_at or datetime.now(timezone.utc)
    recent = submissions.get("filings", {}).get("recent", {})
    form_list = recent.get("form", [])
    wanted = {f.strip() for f in forms} if forms else None

    items: list[EvidenceItem] = []
    for i in range(len(form_list)):
        form = form_list[i]
        if wanted is not None and form not in wanted:
            continue
        source_type = map_form_to_source_type(form)
        if source_type is None:
            continue

        accession = recent["accessionNumber"][i]
        primary_document = recent.get("primaryDocument", [""] * len(form_list))[i]
        description = recent.get("primaryDocDescription", [""] * len(form_list))[i]
        filing_date = recent.get("filingDate", [None] * len(form_list))[i]
        report_items = recent.get("items", [""] * len(form_list))[i]
        published_at = _parse_published_at(
            recent.get("acceptanceDateTime", [None] * len(form_list))[i],
            filing_date,
        )
        detail = f" ({description})" if description else ""
        date_note = f" filed {filing_date}" if filing_date else ""
        # 8-K items carry the actual signal (earnings / dilution / delisting /
        # signed deal); decode them so the fact is more than "an 8-K exists".
        if source_type == "sec_8k" and report_items:
            label, leans = label_8k_items(report_items)
            item_note = f"; items: {label}" if label else ""
            directional = sorted(leans - {"neutral"})
            if directional:
                item_note += f" [signals: {', '.join(directional)}]"
        elif report_items:
            item_note = f"; items {report_items}"
        else:
            item_note = ""
        items.append(
            EvidenceItem(
                evidence_id=f"sec-{accession}",
                ticker=ticker.upper(),
                source_type=source_type,
                source_url=build_filing_url(cik, accession, primary_document),
                published_at=published_at,
                observed_fact=f"SEC {form} filing{detail}{date_note}{item_note}.",
                retrieved_at=retrieved,
            )
        )
        if len(items) >= limit:
            break
    return items


class SecEdgarClient:
    """Minimal EDGAR client over stdlib urllib (no extra dependency).

    SEC requires a descriptive User-Agent that includes contact info; supply one
    via settings/env. See https://www.sec.gov/os/webmaster-faq#developers.
    """

    def __init__(self, user_agent: str, timeout: float = 25.0):
        if not user_agent or "@" not in user_agent:
            raise SecEdgarError(
                "SEC requires a User-Agent with contact info, e.g. "
                "'My Research name@example.com'. Set TRADING_AGENT_SEC_USER_AGENT."
            )
        self.user_agent = user_agent
        self.timeout = timeout
        # ticker -> (cik_padded, cik_int, company_name)
        self._ticker_cache: dict[str, tuple[str, int, str]] | None = None

    def _get_bytes(self, url: str) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = response.read()
                if response.headers.get("Content-Encoding") == "gzip":
                    payload = gzip.decompress(payload)
        except Exception as exc:  # noqa: BLE001 - normalize network errors
            raise SecEdgarError(f"EDGAR request failed for {url}: {exc}") from exc
        return payload

    def _get_json(self, url: str) -> Any:
        return json.loads(self._get_bytes(url))

    def ticker_to_cik(self, ticker: str) -> tuple[str, int]:
        if self._ticker_cache is None:
            self._ticker_cache = self._load_ticker_map()
        key = ticker.strip().upper()
        if key not in self._ticker_cache:
            raise SecEdgarError(f"ticker not found in EDGAR registry: {ticker}")
        entry = self._ticker_cache[key]
        return entry[0], entry[1]

    def company_name(self, ticker: str) -> str | None:
        """Registry company name for a ticker, or None if unknown.

        Best-effort (never raises): used to query news by company name instead
        of a bare ticker, so single-word symbols like TE or BULL stop pulling
        unrelated headlines.
        """

        try:
            if self._ticker_cache is None:
                self._ticker_cache = self._load_ticker_map()
        except SecEdgarError:
            return None
        entry = self._ticker_cache.get(ticker.strip().upper())
        return entry[2] if entry and len(entry) > 2 and entry[2] else None

    def _load_ticker_map(self) -> dict[str, tuple[str, int, str]]:
        raw = self._get_json(COMPANY_TICKERS_URL)
        mapping: dict[str, tuple[str, int, str]] = {}
        for row in raw.values():
            cik_int = int(row["cik_str"])
            name = str(row.get("title") or "")
            mapping[str(row["ticker"]).upper()] = (str(cik_int).zfill(10), cik_int, name)
        return mapping

    def recent_8k_tickers(self) -> set[str]:
        """Registry tickers that just filed an 8-K (current-events feed).

        Event seeds for universe selection: these names rank ahead of the
        volume-ratio order, but still only enter the universe if they pass the
        mandate screen. CIKs with no listed ticker (funds, private filers) drop
        out in the registry join.
        """

        atom = self._get_bytes(CURRENT_8K_URL).decode("utf-8", errors="replace")
        ciks = parse_current_filing_ciks(atom)
        if not ciks:
            return set()
        if self._ticker_cache is None:
            self._ticker_cache = self._load_ticker_map()
        return {
            ticker
            for ticker, entry in self._ticker_cache.items()
            if entry[1] in ciks
        }

    def filing_body_excerpt(self, url: str, max_chars: int = 1500) -> str:
        """Plain-text excerpt of a filing document; '' on any failure.

        Best-effort enrichment: the committee otherwise sees only that an 8-K
        exists, never what it says. A fetch/parse error returns '' so a slow or
        moved document never blocks the entry it was meant to inform.
        """

        try:
            raw = self._get_bytes(url).decode("utf-8", errors="replace")
        except SecEdgarError:
            return ""
        return strip_html(raw)[:max_chars].strip()

    def fetch_evidence(
        self,
        ticker: str,
        *,
        forms: list[str] | None = None,
        limit: int = 20,
        fetch_bodies: bool = False,
        body_limit: int = 2,
        body_chars: int = 1500,
    ) -> list[EvidenceItem]:
        cik_padded, cik_int = self.ticker_to_cik(ticker)
        submissions = self._get_json(SUBMISSIONS_URL.format(cik=cik_padded))
        items = parse_submissions(
            submissions, ticker=ticker, cik=cik_int, forms=forms, limit=limit
        )
        if fetch_bodies:
            items = self._enrich_with_bodies(items, body_limit, body_chars)
        return items

    def _enrich_with_bodies(
        self, items: list[EvidenceItem], body_limit: int, body_chars: int
    ) -> list[EvidenceItem]:
        """Append a body excerpt to the most recent few 8-K evidence items.

        Bounded to ``body_limit`` filings (most recent first) so a name with a
        long 8-K history costs a fixed, small number of extra requests. Only
        8-Ks are fetched: filings are returned newest-first by EDGAR, so the
        first matches are the freshest catalysts.
        """

        fetched = 0
        enriched: list[EvidenceItem] = []
        for item in items:
            if (
                fetched < body_limit
                and item.source_type == "sec_8k"
                and item.source_url
            ):
                excerpt = self.filing_body_excerpt(item.source_url, body_chars)
                if excerpt:
                    item = item.model_copy(
                        update={"observed_fact": f"{item.observed_fact} {excerpt}"}
                    )
                fetched += 1
            enriched.append(item)
        return enriched
