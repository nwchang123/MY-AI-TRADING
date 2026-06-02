from datetime import datetime, timezone

import pytest

from trading_agent.data.sec_edgar import (
    SecEdgarClient,
    SecEdgarError,
    build_filing_url,
    map_form_to_source_type,
    parse_submissions,
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
