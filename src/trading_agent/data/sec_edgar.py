from __future__ import annotations

import gzip
import json
import urllib.request
from datetime import datetime, timezone
from typing import Any

from trading_agent.domain.evidence import EvidenceItem, SourceType

COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

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
        item_note = f"; items {report_items}" if report_items else ""
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
        self._ticker_cache: dict[str, tuple[str, int]] | None = None

    def _get_json(self, url: str) -> Any:
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
        return json.loads(payload)

    def ticker_to_cik(self, ticker: str) -> tuple[str, int]:
        if self._ticker_cache is None:
            self._ticker_cache = self._load_ticker_map()
        key = ticker.strip().upper()
        if key not in self._ticker_cache:
            raise SecEdgarError(f"ticker not found in EDGAR registry: {ticker}")
        return self._ticker_cache[key]

    def _load_ticker_map(self) -> dict[str, tuple[str, int]]:
        raw = self._get_json(COMPANY_TICKERS_URL)
        mapping: dict[str, tuple[str, int]] = {}
        for row in raw.values():
            cik_int = int(row["cik_str"])
            mapping[str(row["ticker"]).upper()] = (str(cik_int).zfill(10), cik_int)
        return mapping

    def fetch_evidence(
        self, ticker: str, *, forms: list[str] | None = None, limit: int = 20
    ) -> list[EvidenceItem]:
        cik_padded, cik_int = self.ticker_to_cik(ticker)
        submissions = self._get_json(SUBMISSIONS_URL.format(cik=cik_padded))
        return parse_submissions(
            submissions, ticker=ticker, cik=cik_int, forms=forms, limit=limit
        )
