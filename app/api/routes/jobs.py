from celery.result import AsyncResult
from fastapi import APIRouter

from app.worker.celery_app import celery_app

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """
    Poll the status of an async ingest job by its ID.

    Looks up the Celery task result in Redis and returns the current state.
    On SUCCESS, includes the pipeline output (tags and caption). On FAILURE,
    includes the error message.

    Args:
        job_id (str): UUID returned by POST /upload.

    Returns:
        dict: Keys — job_id (str), status (str: PENDING | STARTED | SUCCESS | FAILURE),
              result (dict, only on SUCCESS), error (str, only on FAILURE).
    """
    result = AsyncResult(job_id, app=celery_app)
    return {
        "job_id": job_id,
        "status": result.state,
        "result": result.result if result.state == "SUCCESS" else None,
        "error": str(result.result) if result.state == "FAILURE" else None,
    }
