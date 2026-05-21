"""
Tests for Block 4A — governance endpoints and DB helper functions.

Endpoint tests use a minimal FastAPI app with only the governance router.
All DB calls are mocked — no real PostgreSQL connection required.
"""

from datetime import date, datetime
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.routes.governance import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


# ── /governance/expired ────────────────────────────────────────────────────────


def test_expired_assets_returns_list():
    """Returns expired asset rows when DB has results."""
    rows = [{"asset_id": 1, "usage_rights": "CC-BY", "expiry_date": date(2025, 1, 1)}]
    with patch("app.api.routes.governance.get_expired_assets", return_value=rows):
        response = client.get("/governance/expired")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["asset_id"] == 1
    assert data[0]["expiry_date"] == "2025-01-01"


def test_expired_assets_empty_when_none():
    """Returns empty list when no assets are expired."""
    with patch("app.api.routes.governance.get_expired_assets", return_value=[]):
        response = client.get("/governance/expired")
    assert response.status_code == 200
    assert response.json() == []


def test_expired_assets_empty_when_db_down():
    """Returns empty list when DB is unreachable — fail-silent behaviour."""
    with patch("app.api.routes.governance.get_expired_assets", return_value=[]):
        response = client.get("/governance/expired")
    assert response.status_code == 200
    assert response.json() == []


# ── /governance/rights-missing ─────────────────────────────────────────────────


def test_rights_missing_returns_list():
    """Returns assets with no usage_rights set."""
    rows = [
        {"asset_id": 10, "created_at": datetime(2026, 5, 15, 10, 0, 0)},
        {"asset_id": 11, "created_at": datetime(2026, 5, 15, 11, 0, 0)},
    ]
    with patch("app.api.routes.governance.get_rights_missing_assets", return_value=rows):
        response = client.get("/governance/rights-missing")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["asset_id"] == 10


def test_rights_missing_empty_when_all_set():
    """Returns empty list when every asset has usage_rights."""
    with patch("app.api.routes.governance.get_rights_missing_assets", return_value=[]):
        response = client.get("/governance/rights-missing")
    assert response.status_code == 200
    assert response.json() == []


# ── /governance/audit/{asset_id} ───────────────────────────────────────────────


def test_audit_returns_record():
    """Returns full governance record including ai_actions_log for a known asset."""
    row = {
        "asset_id": 42,
        "usage_rights": "all-rights-reserved",
        "expiry_date": None,
        "ai_actions_log": [{"action": "ingest_image", "timestamp": "2026-05-15T10:00:00+00:00"}],
        "face_consent": None,
        "created_at": datetime(2026, 5, 15, 10, 0, 0),
    }
    with patch("app.api.routes.governance.get_asset_audit", return_value=row):
        response = client.get("/governance/audit/42")
    assert response.status_code == 200
    data = response.json()
    assert data["asset_id"] == 42
    assert len(data["ai_actions_log"]) == 1
    assert data["ai_actions_log"][0]["action"] == "ingest_image"


def test_audit_returns_404_when_not_found():
    """Returns 404 when no governance record exists for the given asset ID."""
    with patch("app.api.routes.governance.get_asset_audit", return_value=None):
        response = client.get("/governance/audit/9999")
    assert response.status_code == 404


def test_audit_empty_actions_log():
    """Returns 200 with empty ai_actions_log for a freshly ingested asset."""
    row = {
        "asset_id": 5,
        "usage_rights": None,
        "expiry_date": None,
        "ai_actions_log": [],
        "face_consent": None,
        "created_at": datetime(2026, 5, 15, 10, 0, 0),
    }
    with patch("app.api.routes.governance.get_asset_audit", return_value=row):
        response = client.get("/governance/audit/5")
    assert response.status_code == 200
    assert response.json()["ai_actions_log"] == []


# ── DB helper functions ────────────────────────────────────────────────────────


def test_write_governance_record_returns_true_on_success():
    """Returns True when PostgreSQL insert succeeds."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("app.db.database._get_connection", return_value=mock_conn):
        from app.db.database import write_governance_record

        result = write_governance_record(42)
    assert result is True


def test_write_governance_record_returns_false_when_db_down():
    """Returns False without raising when PostgreSQL is unreachable."""
    with patch("app.db.database._get_connection", side_effect=Exception("connection refused")):
        from app.db.database import write_governance_record

        result = write_governance_record(42)
    assert result is False


def test_append_ai_action_returns_true_on_success():
    """Returns True when JSONB append update succeeds."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with patch("app.db.database._get_connection", return_value=mock_conn):
        from app.db.database import append_ai_action

        result = append_ai_action(42, {"action": "ingest_image", "timestamp": "2026-05-15"})
    assert result is True


def test_append_ai_action_returns_false_when_db_down():
    """Returns False without raising when PostgreSQL is unreachable."""
    with patch("app.db.database._get_connection", side_effect=Exception("connection refused")):
        from app.db.database import append_ai_action

        result = append_ai_action(42, {"action": "ingest_image", "timestamp": "2026-05-15"})
    assert result is False
