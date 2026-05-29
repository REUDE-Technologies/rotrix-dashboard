# type: ignore
"""
S3 storage helper for report PDFs and CSVs. Uses boto3 with credentials from
environment variables. Never hardcode credentials in code.

Required env vars (set on the server / in .env):
  AWS_ACCESS_KEY_ID or AWS_ACCESS_KEY
  AWS_SECRET_ACCESS_KEY or AWS_SECRET_KEY
  S3_BUCKET
  AWS_REGION or S3_REGION (e.g. ap-south-1)
"""
import os
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Config from environment (support both standard and custom names)
# ---------------------------------------------------------------------------
def _env(key: str, alt_key: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is not None and v.strip():
        return v.strip()
    if alt_key:
        v = os.environ.get(alt_key)
        if v is not None and v.strip():
            return v.strip()
    return None


def is_s3_configured() -> bool:
    """True if S3 bucket and credentials are set in environment."""
    access = _env("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY")
    secret = _env("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_KEY")
    bucket = _env("S3_BUCKET")
    return bool(access and secret and bucket)


def _get_s3_client():
    """Lazy boto3 S3 client using env credentials."""
    import boto3
    access = _env("AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY")
    secret = _env("AWS_SECRET_ACCESS_KEY", "AWS_SECRET_KEY")
    region = _env("AWS_REGION", "S3_REGION") or "ap-south-1"
    return boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
    )


def _bucket() -> str:
    b = _env("S3_BUCKET")
    if not b:
        raise ValueError("S3_BUCKET is not set")
    return b


def _report_key(prefix: str, filename_base: str, ext: str) -> str:
    """Generate a unique S3 key: prefix/YYYY/MM/DD/filename_base_timestamp.ext"""
    now = datetime.utcnow()
    safe_base = "".join(c if c.isalnum() or c in "-_" else "_" for c in filename_base)
    return f"{prefix}/{now.year}/{now.month:02d}/{now.day:02d}/{safe_base}_{now.strftime('%H%M%S')}.{ext}"


def upload_report_pdf(pdf_bytes: bytes, filename_base: str) -> Optional[str]:
    """
    Upload report PDF to S3. Returns S3 key on success, None on failure.
    Caller can then drop pdf_bytes from memory.
    """
    if not is_s3_configured() or not pdf_bytes:
        return None
    try:
        key = _report_key("reports", filename_base, "pdf")
        _get_s3_client().put_object(
            Bucket=_bucket(),
            Key=key,
            Body=pdf_bytes,
            ContentType="application/pdf",
        )
        return key
    except Exception:
        return None


def upload_report_csv(csv_bytes: bytes, filename_base: str) -> Optional[str]:
    """
    Upload report CSV to S3. Returns S3 key on success, None on failure.
    """
    if not is_s3_configured() or not csv_bytes:
        return None
    try:
        key = _report_key("reports", filename_base, "csv")
        _get_s3_client().put_object(
            Bucket=_bucket(),
            Key=key,
            Body=csv_bytes,
            ContentType="text/csv",
        )
        return key
    except Exception:
        return None


def get_object_bytes(key: str) -> Optional[bytes]:
    """Download object from S3 by key. Returns None on failure."""
    if not is_s3_configured() or not key:
        return None
    try:
        resp = _get_s3_client().get_object(Bucket=_bucket(), Key=key)
        return resp["Body"].read()
    except Exception:
        return None
