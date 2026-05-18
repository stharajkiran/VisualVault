"""PostgreSQL — corrections, governance, and person-cluster tables."""
import json
import os

import psycopg2
from psycopg2.extras import RealDictCursor

POSTGRES_HOST = os.environ.get("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT", "5432")
POSTGRES_USER = os.environ.get("POSTGRES_USER", "visualvault")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "visualvault")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "visualvault")

_CREATE_CORRECTIONS = """
CREATE TABLE IF NOT EXISTS corrections (
    id          SERIAL PRIMARY KEY,
    filename    TEXT NOT NULL,
    class_id    INTEGER NOT NULL,
    cx          REAL NOT NULL,
    cy          REAL NOT NULL,
    w           REAL NOT NULL,
    h           REAL NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# person_clusters stays empty in v4.0 — populated by MTCNN+ArcFace in v4.1.
# consent_status NULL means unknown; ArcFace will set it after cluster review.
_CREATE_PERSON_CLUSTERS = """
CREATE TABLE IF NOT EXISTS person_clusters (
    cluster_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    consent_status  TEXT,
    first_seen      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    review_status   TEXT NOT NULL DEFAULT 'pending'
);
"""

# face_consent is TEXT (nullable) in v4.0.
# In v4.1 it becomes a FK to person_clusters.cluster_id once ArcFace populates clusters.
_CREATE_ASSETS_GOVERNANCE = """
CREATE TABLE IF NOT EXISTS assets_governance (
    asset_id        INTEGER PRIMARY KEY,
    usage_rights    TEXT,
    expiry_date     DATE,
    ai_actions_log  JSONB NOT NULL DEFAULT '[]'::jsonb,
    face_consent    TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""


def _get_connection() -> psycopg2.extensions.connection:
    """
    Open a new PostgreSQL connection using env-var credentials.

    Args:
        None

    Returns:
        psycopg2.extensions.connection: Open database connection.
    """
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        dbname=POSTGRES_DB,
    )


def _ensure_tables(cur: psycopg2.extensions.cursor) -> None:
    """
    Create all tables if they do not already exist.

    Args:
        cur (psycopg2.extensions.cursor): Open cursor inside an active transaction.

    Returns:
        None
    """
    cur.execute(_CREATE_CORRECTIONS)
    cur.execute(_CREATE_PERSON_CLUSTERS)
    cur.execute(_CREATE_ASSETS_GOVERNANCE)


def write_correction(
    filename: str,
    class_id: int,
    cx: float,
    cy: float,
    w: float,
    h: float,
) -> bool:
    """
    Insert one correction row into the corrections table.

    Creates all tables on first call if they do not exist. Fails silently
    if PostgreSQL is unreachable so a missing database never crashes the webhook.

    Args:
        filename (str): Image filename the correction belongs to.
        class_id (int): COCO class index.
        cx (float): Bounding box center x, normalized 0–1.
        cy (float): Bounding box center y, normalized 0–1.
        w (float): Bounding box width, normalized 0–1.
        h (float): Bounding box height, normalized 0–1.

    Returns:
        bool: True if the row was written, False if the connection failed.
    """
    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                _ensure_tables(cur)
                cur.execute(
                    "INSERT INTO corrections (filename, class_id, cx, cy, w, h) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    (filename, class_id, cx, cy, w, h),
                )
        conn.close()
        return True
    except Exception:
        return False


def count_corrections() -> int:
    """
    Return the total number of correction rows in the database.

    Used by Block 3E Celery beat to decide when retraining should trigger.

    Args:
        None

    Returns:
        int: Row count, or 0 if the connection fails.
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT COUNT(*) AS total FROM corrections")
            result = cur.fetchone()
        conn.close()
        return int(result["total"])
    except Exception:
        return 0


def write_governance_record(asset_id: int) -> bool:
    """
    Insert a governance row for a newly uploaded asset.

    Called by ingest_image immediately after Qdrant upsert. usage_rights,
    expiry_date, and face_consent are all NULL until set explicitly.

    Args:
        asset_id (int): Qdrant point ID for the asset.

    Returns:
        bool: True if the row was written, False if the connection failed.
    """
    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                _ensure_tables(cur)
                cur.execute(
                    "INSERT INTO assets_governance (asset_id) VALUES (%s) "
                    "ON CONFLICT (asset_id) DO NOTHING",
                    (asset_id,),
                )
        conn.close()
        return True
    except Exception:
        return False


def append_ai_action(asset_id: int, action: dict) -> bool:
    """
    Append one AI action entry to the asset's ai_actions_log JSON array.

    Each action dict should include at minimum: action, timestamp, and any
    model-specific fields (e.g. tags, caption, drift_score).

    Args:
        asset_id (int): Qdrant point ID for the asset.
        action (dict): Action record to append.

    Returns:
        bool: True if the row was updated, False if the connection failed.
    """
    try:
        conn = _get_connection()
        with conn:
            with conn.cursor() as cur:
                _ensure_tables(cur)
                cur.execute(
                    "UPDATE assets_governance "
                    "SET ai_actions_log = ai_actions_log || %s::jsonb "
                    "WHERE asset_id = %s",
                    (json.dumps([action]), asset_id),
                )
        conn.close()
        return True
    except Exception:
        return False


def get_expired_assets() -> list[dict]:
    """
    Return all governance rows where expiry_date is in the past.

    Args:
        None

    Returns:
        list[dict]: Governance rows with asset_id, usage_rights, expiry_date.
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT asset_id, usage_rights, expiry_date "
                "FROM assets_governance "
                "WHERE expiry_date IS NOT NULL AND expiry_date < CURRENT_DATE "
                "ORDER BY expiry_date"
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_rights_missing_assets() -> list[dict]:
    """
    Return all governance rows where usage_rights has not been set.

    Args:
        None

    Returns:
        list[dict]: Governance rows with asset_id and created_at.
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT asset_id, created_at "
                "FROM assets_governance "
                "WHERE usage_rights IS NULL "
                "ORDER BY created_at DESC"
            )
            rows = cur.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


def get_asset_audit(asset_id: int) -> dict | None:
    """
    Return the full governance record for a single asset.

    Args:
        asset_id (int): Qdrant point ID for the asset.

    Returns:
        dict | None: Full governance row, or None if not found or DB unreachable.
    """
    try:
        conn = _get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM assets_governance WHERE asset_id = %s",
                (asset_id,),
            )
            row = cur.fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception:
        return None
