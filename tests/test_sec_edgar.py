import json
from datetime import datetime, timezone

import pytest

from trading_agent.data.sec_edgar import (
    SecEdgarClient,
    SecEdgarError,
    build_filing_url,
    label_8k_items,
    map_form_to_source_type,
    parse_current_filing_ciks,
    parse_submissions,
    strip_html,
)

RETRIEVED = datetime(2026, 6, 2, 13, 0, tzinfo=timezone.utc)


def _submissions() -> dict:
    return {
        "filings": {
            "recent": {
                "accessionNumber": [
                    "0001045810-26-000060",
                    "0001045810-26-000059",
                    "0001045810-26-000058",
                    "0001045810-26-000057",
                    "0001045810-26-000056",
                ],
                "filingDate": ["2026-06-01", "2026-05-30", "2026-05-29", "2026-05-28", "2026-05-27"],
                "acceptanceDateTime": [
                    "2026-06-01T16:30:00.000Z",
                    "2026-05-30T16:30:00.000Z",
                    "2026-05-29T16:30:00.000Z",
                    "2026-05-28T16:30:00.000Z",
                    "2026-05-27T16:30:00.000Z",
                ],
                "form": ["8-K", "144", "S-3", "424B5", "4"],
                "primaryDocument": ["a.htm", "b.xml", "c.htm", "d.htm", "e.xml"],
                "primaryDocDescription": ["FORM 8-K", "", "FORM S-3", "424B5", "FORM 4"],
            }
        }
    }


def test_map_form_to_source_type() -> None:
    assert map_form_to_source_type("8-K") == "sec_8k"
    assert map_form_to_source_type("S-3/A") == "sec_s3"
    assert map_form_to_source_type("424B5") == "sec_424b"
    assert map_form_to_source_type("4") == "sec_form4"
    assert map_form_to_source_type("144") is None


def test_build_filing_url_strips_dashes() -> None:
    url = build_filing_url(1045810, "0001045810-26-000058", "c.htm")
    assert url == "https://www.sec.gov/Archives/edgar/data/1045810/000104581026000058/c.htm"


def test_parse_submissions_maps_and_skips_unmapped_forms() -> None:
    items = parse_submissions(
        _submissions(), ticker="nvda", cik=1045810, retrieved_at=RETRIEVED
    )
    # 144 is unmapped and dropped; the other four map.
    types = [i.source_type for i in items]
    assert types == ["sec_8k", "sec_s3", "sec_424b", "sec_form4"]
    first = items[0]
    assert first.ticker == "NVDA"
    assert first.evidence_id == "sec-0001045810-26-000060"
    assert first.published_at == datetime(2026, 6, 1, 16, 30, tzinfo=timezone.utc)
    assert first.source_url.startswith("https://www.sec.gov/Archives/edgar/data/1045810/")
    assert "8-K" in first.observed_fact


def test_parse_submissions_respects_form_filter_and_limit() -> None:
    items = parse_submissions(
        _submissions(), ticker="NVDA", cik=1045810, forms=["8-K", "S-3"], limit=1
    )
    assert len(items) == 1
    assert items[0].source_type == "sec_8k"


def test_client_requires_contactful_user_agent() -> None:
    with pytest.raises(SecEdgarError, match="User-Agent"):
        SecEdgarClient("no-contact-here")
    # A contactful UA is accepted.
    SecEdgarClient("Research Team research@example.com")


_CURRENT_8K_ATOM = """<?xml version="1.0" encoding="ISO-8859-1" ?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>Latest Filings - Thu, 11 Jun 2026 15:00:00 EDT</title>
  <entry>
    <title>8-K - Acme Therapeutics Inc. (0001234567) (Filer)</title>
    <summary> Filed: 2026-06-11 AccNo: 0001234567-26-000042 Size: 312 KB</summary>
  </entry>
  <entry>
    <title>8-K/A - Borealis Mining Corp (0000111222) (Filer)</title>
    <summary> Filed: 2026-06-11 AccNo: 0000111222-26-000007 Size: 88 KB</summary>
  </entry>
</feed>
"""


def test_parse_current_filing_ciks_reads_parenthesized_ciks() -> None:
    # The dash-separated accession numbers in the summaries must not match.
    assert parse_current_filing_ciks(_CURRENT_8K_ATOM) == {1234567, 111222}
    assert parse_current_filing_ciks("<feed></feed>") == set()


def test_recent_8k_tickers_joins_feed_against_registry() -> None:
    client = SecEdgarClient("Research Team research@example.com")
    client._get_bytes = lambda url: _CURRENT_8K_ATOM.encode("utf-8")  # type: ignore[method-assign]
    # Pre-seeded registry: ACME maps to a feed CIK, NVDA does not, and the
    # second feed CIK (a fund with no listed ticker) simply joins to nothing.
    client._ticker_cache = {
        "ACME": ("0001234567", 1234567, "Acme Therapeutics Inc."),
        "NVDA": ("0001045810", 1045810, "NVIDIA Corp"),
    }

    assert client.recent_8k_tickers() == {"ACME"}


def test_company_name_from_registry() -> None:
    client = SecEdgarClient("Research Team research@example.com")
    client._ticker_cache = {"ACME": ("0001234567", 1234567, "Acme Therapeutics Inc.")}
    assert client.company_name("acme") == "Acme Therapeutics Inc."
    assert client.company_name("UNKNOWN") is None


def test_label_8k_items_decodes_codes_and_leans() -> None:
    label, leans = label_8k_items("2.02,9.01")
    assert "Results of Operations" in label
    assert leans == {"earnings", "neutral"}
    # Signed-deal and bankruptcy carry opposite leans.
    assert label_8k_items("1.01")[1] == {"bull"}
    assert label_8k_items("1.03")[1] == {"bear"}
    # Unknown code is surfaced verbatim, never dropped.
    assert label_8k_items("9.99") == ("item 9.99", {"neutral"})
    assert label_8k_items("") == ("", set())


def test_strip_html_drops_tags_scripts_and_entities() -> None:
    html = "<html><style>p{color:red}</style><body>Acme <b>signs</b>&nbsp;deal</body></html>"
    assert strip_html(html) == "Acme signs deal"


def test_parse_submissions_labels_8k_items_with_signals() -> None:
    subs = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-1"],
                "filingDate": ["2026-06-01"],
                "acceptanceDateTime": ["2026-06-01T16:30:00.000Z"],
                "form": ["8-K"],
                "primaryDocument": ["a.htm"],
                "primaryDocDescription": ["FORM 8-K"],
                "items": ["2.02,9.01"],
            }
        }
    }
    fact = parse_submissions(subs, ticker="ABC", cik=1, retrieved_at=RETRIEVED)[0].observed_fact
    assert "Results of Operations" in fact
    assert "[signals: earnings]" in fact


def test_fetch_bodies_appends_excerpt_to_8k(monkeypatch) -> None:
    client = SecEdgarClient("Research Team research@example.com")
    client._ticker_cache = {"ABC": ("0000000001", 1, "ABC Corp")}
    subs = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-1"],
                "filingDate": ["2026-06-01"],
                "acceptanceDateTime": ["2026-06-01T16:30:00.000Z"],
                "form": ["8-K"],
                "primaryDocument": ["a.htm"],
                "primaryDocDescription": ["FORM 8-K"],
                "items": ["1.01"],
            }
        }
    }
    body = b"<html><body>Company entered a binding supply agreement worth $50M.</body></html>"

    def fake_get_bytes(url):
        return json.dumps(subs).encode() if url.endswith(".json") else body

    client._get_bytes = fake_get_bytes  # type: ignore[method-assign]
    items = client.fetch_evidence("ABC", fetch_bodies=True)
    assert "binding supply agreement worth $50M" in items[0].observed_fact


def test_fetch_bodies_failure_is_nonfatal(monkeypatch) -> None:
    client = SecEdgarClient("Research Team research@example.com")
    client._ticker_cache = {"ABC": ("0000000001", 1, "ABC Corp")}
    subs = {
        "filings": {
            "recent": {
                "accessionNumber": ["0001-26-1"],
                "filingDate": ["2026-06-01"],
                "acceptanceDateTime": ["2026-06-01T16:30:00.000Z"],
                "form": ["8-K"],
                "primaryDocument": ["a.htm"],
                "primaryDocDescription": ["FORM 8-K"],
                "items": ["1.01"],
            }
        }
    }

    def fake_get_bytes(url):
        if url.endswith(".json"):
            return json.dumps(subs).encode()
        raise SecEdgarError("document moved")

    client._get_bytes = fake_get_bytes  # type: ignore[method-assign]
    # Body fetch failed but the metadata evidence still comes back intact.
    items = client.fetch_evidence("ABC", fetch_bodies=True)
    assert len(items) == 1
    assert items[0].source_type == "sec_8k"
