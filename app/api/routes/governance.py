"""Governance endpoints — expired assets, rights-missing assets, per-asset audit log."""
from datetime import date, datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.db.database import get_asset_audit, get_expired_assets, get_rights_missing_assets

router = APIRouter()

# Expired assets: expiry_date < today
class ExpiredAsset(BaseModel):
    asset_id: int
    usage_rights: str | None
    expiry_date: date

# Rights-missing assets: usage_rights IS NULL
class RightsMissingAsset(BaseModel):
    asset_id: int
    created_at: datetime

# Full governance record for audit endpoint
class GovernanceRecord(BaseModel):
    asset_id: int
    usage_rights: str | None
    expiry_date: date | None
    ai_actions_log: list[dict]
    face_consent: str | None
    created_at: datetime


@router.get("/governance/expired", response_model=list[ExpiredAsset])
async def expired_assets() -> list[ExpiredAsset]:
    """
    Return all assets whose expiry_date is in the past.

    Args:
        None

    Returns:
        list[ExpiredAsset]: Assets with asset_id, usage_rights, and expiry_date.
    """
    return [ExpiredAsset(**row) for row in get_expired_assets()]


@router.get("/governance/rights-missing", response_model=list[RightsMissingAsset])
async def rights_missing_assets() -> list[RightsMissingAsset]:
    """
    Return all assets that have no usage_rights set.

    Args:
        None

    Returns:
        list[RightsMissingAsset]: Assets with asset_id and upload timestamp.
    """
    return [RightsMissingAsset(**row) for row in get_rights_missing_assets()]


@router.get("/governance/audit/{asset_id}", response_model=GovernanceRecord)
async def asset_audit(asset_id: int) -> GovernanceRecord:
    """
    Return the full governance record for a single asset, including ai_actions_log.

    Args:
        asset_id (int): Qdrant point ID for the asset.

    Returns:
        GovernanceRecord: Full governance row with all fields and action history.

    Raises:
        HTTPException: 404 if the asset has no governance record.
    """
    row = get_asset_audit(asset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"No governance record for asset {asset_id}")
    return GovernanceRecord(**row)
