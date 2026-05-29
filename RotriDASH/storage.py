#type: ignore
"""
File storage module for Rotrix Dashboard.

Abstraction layer over storage backends (local filesystem, S3, or Supabase
Storage for files) and database backends (local Postgres or Supabase for
metadata).

Usage:
    from storage import upload_file, download_file, upload_report, get_download_url
"""

import os
import re
import time
import uuid
import logging
from datetime import datetime, timezone

import streamlit as st

logger = logging.getLogger(__name__)
from dotenv import load_dotenv

try:
    load_dotenv()
except OSError:
    pass  # e.g. Errno 24 too many open files; env may already be set

# ---------------------------------------------------------------------------
# Backend switches
# ---------------------------------------------------------------------------
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")  # "local", "s3", or "supabase"
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "rotrix-files")
LOCAL_STORAGE_ROOT = os.getenv("LOCAL_STORAGE_ROOT", os.path.join(os.path.dirname(__file__), "local_storage"))


def _use_local_db() -> bool:
    """Metadata is stored in local Postgres."""
    return True


def _get_db():
    """Return a SQLAlchemy session."""
    from models import SessionLocal
    return SessionLocal()


# ---------------------------------------------------------------------------
# Supabase Storage client (file blobs only — not used for authentication)
# ---------------------------------------------------------------------------
def _get_supabase():
    """Return Supabase client for storage API when STORAGE_BACKEND=supabase."""
    if STORAGE_BACKEND != "supabase":
        return None
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "") or os.getenv("SUPABASE_ANON_KEY", "")
    if not url or not key:
        return None
    from supabase import create_client
    return create_client(url, key)


# ---------------------------------------------------------------------------
# S3 client (lazy init — only created when backend is "s3")
# ---------------------------------------------------------------------------
_s3_client = None

def _get_s3_client():
    """Create and cache a boto3 S3 client."""
    global _s3_client
    if _s3_client is None:
        import boto3
        _s3_client = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "ap-south-1"),
        )
    return _s3_client


def _get_s3_bucket():
    """Get the S3 bucket name from env.

    Prefer AWS_S3_BUCKET, but fall back to S3_BUCKET for compatibility
    with existing environments that only define S3_BUCKET.
    """
    return (
        os.getenv("AWS_S3_BUCKET")
        or os.getenv("S3_BUCKET")
        or "rotrix-dashboard-files"
    )


# ---------------------------------------------------------------------------
# Local filesystem helpers (used when STORAGE_BACKEND == "local")
# ---------------------------------------------------------------------------
def _local_full_path(storage_path: str) -> str:
    return os.path.join(LOCAL_STORAGE_ROOT, storage_path)


def _local_upload(file_bytes: bytes, storage_path: str):
    full = _local_full_path(storage_path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "wb") as f:
        f.write(file_bytes)


def _local_download(storage_path: str) -> bytes | None:
    full = _local_full_path(storage_path)
    if not os.path.exists(full):
        return None
    with open(full, "rb") as f:
        return f.read()


def _local_delete(storage_path: str):
    full = _local_full_path(storage_path)
    if os.path.exists(full):
        os.remove(full)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _sanitize_filename(name: str) -> str:
    """Sanitize filename for storage (alphanumeric, hyphen, underscore, dot only)."""
    return re.sub(r"[^\w\-.]", "_", name)


def _build_file_path(org_id: str, user_id: str, filename: str) -> str:
    """Build a storage path: {org_id}/{user_id}/{timestamp_ms}_{short_uuid}_{filename}"""
    ts_ms = int(time.time() * 1000)
    short_uuid = uuid.uuid4().hex[:8]
    safe_name = _sanitize_filename(filename)
    return f"{org_id}/{user_id}/{ts_ms}_{short_uuid}_{safe_name}"


def _build_report_path(org_id: str, user_id: str, report_name: str, ext: str) -> str:
    """Build a report storage path: {org_id}/{user_id}/reports/{timestamp_ms}_{short_uuid}_{report_name}.{ext}"""
    ts_ms = int(time.time() * 1000)
    short_uuid = uuid.uuid4().hex[:8]
    safe_name = _sanitize_filename(report_name)
    return f"{org_id}/{user_id}/reports/{ts_ms}_{short_uuid}_{safe_name}.{ext}"


# ---------------------------------------------------------------------------
# Helper: convert a SQLAlchemy model row to dict (for list queries)
# ---------------------------------------------------------------------------
def _row_to_dict(row) -> dict:
    """Convert a SQLAlchemy model instance to a plain dict with string UUIDs."""
    d = {}
    for col in row.__table__.columns:
        val = getattr(row, col.key)
        # Convert UUID to string for JSON-compatibility
        if isinstance(val, uuid.UUID):
            val = str(val)
        # Convert datetime to ISO string
        elif isinstance(val, datetime):
            val = val.isoformat()
        d[col.key] = val
    return d


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
def upload_file(file_bytes: bytes, filename: str, user_id: str, org_id: str,
                file_type: str = "") -> str | None:
    """Upload a file to storage and record metadata in DB.

    Args:
        file_bytes: Raw file content
        filename: Original filename
        user_id: Current user's UUID
        org_id: Current organization's UUID
        file_type: File type (e.g. 'ulg', 'csv')

    Returns:
        storage_path on success, None on failure.
    """
    storage_path = _build_file_path(org_id, user_id, filename)

    try:
        # 1. Upload file to storage backend
        if STORAGE_BACKEND == "local":
            _local_upload(file_bytes, storage_path)
        elif STORAGE_BACKEND == "s3":
            from io import BytesIO
            s3 = _get_s3_client()
            s3.upload_fileobj(
                BytesIO(file_bytes),
                _get_s3_bucket(),
                storage_path,
                ExtraArgs={"ContentType": "application/octet-stream"},
            )
        else:
            supabase = _get_supabase()
            supabase.storage.from_(BUCKET_NAME).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": "application/octet-stream"},
            )

        # 2. Record metadata in database
        ft = file_type or os.path.splitext(filename)[-1].lstrip(".")
        if _use_local_db():
            from models import FileMetadata as FM
            db = _get_db()
            try:
                fm = FM(
                    user_id=user_id,
                    organization_id=org_id,
                    original_filename=filename,
                    storage_path=storage_path,
                    file_size=len(file_bytes),
                    file_type=ft,
                )
                db.add(fm)
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            supabase = _get_supabase()
            supabase.table("file_metadata").insert({
                "user_id": user_id,
                "organization_id": org_id,
                "original_filename": filename,
                "storage_path": storage_path,
                "file_size": len(file_bytes),
                "file_type": ft,
            }).execute()

        return storage_path

    except Exception as e:
        # If upload succeeded but metadata insert failed, try to remove orphaned file
        try:
            if STORAGE_BACKEND == "local":
                _local_delete(storage_path)
            elif STORAGE_BACKEND == "s3":
                _get_s3_client().delete_object(Bucket=_get_s3_bucket(), Key=storage_path)
            else:
                _get_supabase().storage.from_(BUCKET_NAME).remove([storage_path])
        except Exception:
            pass
        st.error(f"Upload failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
def download_file(storage_path: str, silent: bool = False) -> bytes | None:
    """Download a file from storage.

    Args:
        storage_path: Path/key inside the configured backend.
        silent: If True, suppress Streamlit error messages on failure.

    Returns:
        File bytes on success, None on failure.
    """
    try:
        if STORAGE_BACKEND == "local":
            data = _local_download(storage_path)
            if data is None:
                raise FileNotFoundError(f"File not found: {storage_path}")
            return data
        elif STORAGE_BACKEND == "s3":
            from io import BytesIO

            s3 = _get_s3_client()
            buf = BytesIO()
            s3.download_fileobj(_get_s3_bucket(), storage_path, buf)
            buf.seek(0)
            return buf.read()
        else:
            supabase = _get_supabase()
            data = supabase.storage.from_(BUCKET_NAME).download(storage_path)
            return data

    except Exception as e:
        logger.error("Download failed for %s: %s", storage_path, e)
        if not silent:
            st.error(f"Download failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Report upload (PDF + CSV)
# ---------------------------------------------------------------------------
def upload_report(pdf_bytes: bytes, csv_bytes: bytes | None,
                  report_name: str, user_id: str, org_id: str,
                  source_file_id: str | None = None) -> tuple[str | None, str | None]:
    """Upload a generated report (PDF + optional CSV) to storage.

    Returns:
        (pdf_storage_path, csv_storage_path) — either can be None on failure.
    """
    pdf_path = _build_report_path(org_id, user_id, report_name, "pdf")
    csv_path = _build_report_path(org_id, user_id, report_name, "csv") if csv_bytes else None

    try:
        # 1. Upload files to storage backend
        if STORAGE_BACKEND == "local":
            _local_upload(pdf_bytes, pdf_path)
            if csv_bytes and csv_path:
                _local_upload(csv_bytes, csv_path)
        elif STORAGE_BACKEND == "s3":
            from io import BytesIO
            s3 = _get_s3_client()
            s3.upload_fileobj(
                BytesIO(pdf_bytes), _get_s3_bucket(), pdf_path,
                ExtraArgs={"ContentType": "application/pdf"},
            )
            if csv_bytes and csv_path:
                s3.upload_fileobj(
                    BytesIO(csv_bytes), _get_s3_bucket(), csv_path,
                    ExtraArgs={"ContentType": "text/csv"},
                )
        else:
            supabase = _get_supabase()
            supabase.storage.from_(BUCKET_NAME).upload(
                path=pdf_path, file=pdf_bytes,
                file_options={"content-type": "application/pdf"},
            )
            if csv_bytes and csv_path:
                supabase.storage.from_(BUCKET_NAME).upload(
                    path=csv_path, file=csv_bytes,
                    file_options={"content-type": "text/csv"},
                )

        # 2. Record in report_metadata table
        generated_at = datetime.now(timezone.utc)

        if _use_local_db():
            from models import ReportMetadata as RM
            db = _get_db()
            try:
                rm = RM(
                    user_id=user_id,
                    organization_id=org_id,
                    report_name=report_name,
                    pdf_storage_path=pdf_path,
                    csv_storage_path=csv_path,
                    generated_at=generated_at,
                )
                if source_file_id:
                    rm.source_file_id = source_file_id
                db.add(rm)
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            supabase = _get_supabase()
            insert_data = {
                "user_id": user_id,
                "organization_id": org_id,
                "report_name": report_name,
                "pdf_storage_path": pdf_path,
                "csv_storage_path": csv_path,
                "generated_at": generated_at.isoformat(),
            }
            if source_file_id:
                insert_data["source_file_id"] = source_file_id
            supabase.table("report_metadata").insert(insert_data).execute()

        return pdf_path, csv_path

    except Exception as e:
        # Try to remove orphaned files if upload succeeded but metadata insert failed
        try:
            if STORAGE_BACKEND == "local":
                _local_delete(pdf_path)
                if csv_path:
                    _local_delete(csv_path)
            elif STORAGE_BACKEND == "s3":
                s3 = _get_s3_client()
                s3.delete_object(Bucket=_get_s3_bucket(), Key=pdf_path)
                if csv_path:
                    s3.delete_object(Bucket=_get_s3_bucket(), Key=csv_path)
            else:
                paths_to_remove = [pdf_path]
                if csv_path:
                    paths_to_remove.append(csv_path)
                _get_supabase().storage.from_(BUCKET_NAME).remove(paths_to_remove)
        except Exception:
            logger.warning("Failed to clean up orphaned report files after metadata error")
        st.error(f"Report upload failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Signed download URL
# ---------------------------------------------------------------------------
def get_download_url(storage_path: str, expires_in: int = 3600) -> str | None:
    """Generate a signed/temporary download URL.

    Args:
        storage_path: Path in storage
        expires_in: Seconds until URL expires (default 1 hour)

    Returns:
        URL string or None on failure.
    """
    try:
        if STORAGE_BACKEND == "local":
            # Local storage has no URL; callers should use download_file() instead
            return None
        elif STORAGE_BACKEND == "s3":
            s3 = _get_s3_client()
            url = s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": _get_s3_bucket(), "Key": storage_path},
                ExpiresIn=expires_in,
            )
            return url
        else:
            supabase = _get_supabase()
            result = supabase.storage.from_(BUCKET_NAME).create_signed_url(
                storage_path, expires_in
            )
            if isinstance(result, dict):
                return result.get("signedURL") or result.get("signedUrl")
            return getattr(result, "signed_url", None) or getattr(result, "signedURL", None)

    except Exception as e:
        st.error(f"Failed to generate download URL: {e}")
        return None


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------
def delete_file(storage_path: str) -> bool:
    """Delete a file from storage and remove its file_metadata row.

    Returns:
        True on success, False on failure.
    """
    try:
        # 1. Delete from storage backend
        if STORAGE_BACKEND == "local":
            _local_delete(storage_path)
        elif STORAGE_BACKEND == "s3":
            s3 = _get_s3_client()
            s3.delete_object(Bucket=_get_s3_bucket(), Key=storage_path)
        else:
            supabase = _get_supabase()
            supabase.storage.from_(BUCKET_NAME).remove([storage_path])

        # 2. Remove metadata row
        if _use_local_db():
            from models import FileMetadata as FM
            db = _get_db()
            try:
                db.query(FM).filter(FM.storage_path == storage_path).delete()
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            supabase = _get_supabase()
            supabase.table("file_metadata").delete().eq("storage_path", storage_path).execute()

        return True
    except Exception as e:
        st.error(f"Delete failed: {e}")
        return False


def delete_report_by_id(report_id: str) -> bool:
    """Delete report assets (PDF/CSV) and remove the report_metadata row.

    Args:
        report_id: Primary key of the report_metadata row.

    Returns:
        True on success, False on failure.
    """
    try:
        # 1. Fetch storage paths for this report
        pdf_path = None
        csv_path = None

        if _use_local_db():
            from models import ReportMetadata as RM
            db = _get_db()
            try:
                report = db.get(RM, report_id)
                if report is None:
                    return True  # Nothing to delete
                pdf_path = report.pdf_storage_path
                csv_path = report.csv_storage_path
            finally:
                db.close()
        else:
            supabase = _get_supabase()
            res = supabase.table("report_metadata") \
                .select("pdf_storage_path,csv_storage_path") \
                .eq("id", report_id) \
                .maybe_single() \
                .execute()
            row = getattr(res, "data", None) or getattr(res, "json", None) or None
            if row is None:
                return True
            pdf_path = row.get("pdf_storage_path")
            csv_path = row.get("csv_storage_path")

        # 2. Delete files from storage backend
        paths_to_remove: list[str] = []
        if pdf_path:
            paths_to_remove.append(pdf_path)
        if csv_path:
            paths_to_remove.append(csv_path)

        if paths_to_remove:
            if STORAGE_BACKEND == "local":
                for p in paths_to_remove:
                    _local_delete(p)
            elif STORAGE_BACKEND == "s3":
                s3 = _get_s3_client()
                for p in paths_to_remove:
                    s3.delete_object(Bucket=_get_s3_bucket(), Key=p)
            else:
                _get_supabase().storage.from_(BUCKET_NAME).remove(paths_to_remove)

        # 3. Remove metadata row
        if _use_local_db():
            db = _get_db()
            try:
                from models import ReportMetadata as RM
                db.query(RM).filter(RM.id == report_id).delete()
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
        else:
            supabase = _get_supabase()
            supabase.table("report_metadata").delete().eq("id", report_id).execute()

        return True
    except Exception as e:
        st.error(f"Failed to delete report: {e}")
        return False


# ---------------------------------------------------------------------------
# List files for an organization
# ---------------------------------------------------------------------------
def list_org_files(org_id: str) -> list[dict]:
    """Fetch all file_metadata records for an organization.

    Returns:
        List of file metadata dicts.
    """
    if _use_local_db():
        from models import FileMetadata as FM
        db = _get_db()
        try:
            rows = (
                db.query(FM)
                .filter(FM.organization_id == org_id)
                .order_by(FM.uploaded_at.desc())
                .all()
            )
            return [_row_to_dict(r) for r in rows]
        except Exception:
            return []
        finally:
            db.close()

    try:
        supabase = _get_supabase()
        result = supabase.table("file_metadata") \
            .select("*") \
            .eq("organization_id", org_id) \
            .order("uploaded_at", desc=True) \
            .execute()
        return result.data or []
    except Exception:
        return []


def list_org_reports(org_id: str) -> list[dict]:
    """Fetch all report_metadata records for an organization.

    Returns:
        List of report metadata dicts.
    """
    if _use_local_db():
        from models import ReportMetadata as RM
        db = _get_db()
        try:
            rows = (
                db.query(RM)
                .filter(RM.organization_id == org_id)
                .order_by(RM.generated_at.desc())
                .all()
            )
            return [_row_to_dict(r) for r in rows]
        except Exception:
            return []
        finally:
            db.close()

    try:
        supabase = _get_supabase()
        result = supabase.table("report_metadata") \
            .select("*") \
            .eq("organization_id", org_id) \
            .order("generated_at", desc=True) \
            .execute()
        return result.data or []
    except Exception:
        return []


def list_recent_files(org_id: str, user_id: str | None = None, limit: int = 10) -> list[dict]:
    """Fetch the last N file_metadata records for an org, optionally for one user.

    Returns:
        List of file metadata dicts (most recent first).
    """
    if _use_local_db():
        from models import FileMetadata as FM
        db = _get_db()
        try:
            q = (
                db.query(FM)
                .filter(FM.organization_id == org_id)
                .order_by(FM.uploaded_at.desc())
                .limit(limit)
            )
            if user_id:
                q = q.filter(FM.user_id == user_id)
            rows = q.all()
            return [_row_to_dict(r) for r in rows]
        except Exception:
            return []
        finally:
            db.close()

    try:
        supabase = _get_supabase()
        q = supabase.table("file_metadata") \
            .select("*") \
            .eq("organization_id", org_id) \
            .order("uploaded_at", desc=True) \
            .limit(limit)
        if user_id:
            q = q.eq("user_id", user_id)
        result = q.execute()
        return result.data or []
    except Exception:
        return []


def list_reports_for_user(org_id: str, user_id: str, limit: int = 50) -> list[dict]:
    """Fetch report_metadata records for an organization filtered by user (reports generated by this user).

    Returns:
        List of report metadata dicts (most recent first).
    """
    if _use_local_db():
        from models import ReportMetadata as RM
        db = _get_db()
        try:
            rows = (
                db.query(RM)
                .filter(RM.organization_id == org_id, RM.user_id == user_id)
                .order_by(RM.generated_at.desc())
                .limit(limit)
                .all()
            )
            return [_row_to_dict(r) for r in rows]
        except Exception:
            return []
        finally:
            db.close()

    try:
        supabase = _get_supabase()
        result = supabase.table("report_metadata") \
            .select("*") \
            .eq("organization_id", org_id) \
            .eq("user_id", user_id) \
            .order("generated_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data or []
    except Exception:
        return []


def list_reports_for_org(org_id: str, limit: int = 200) -> list[dict]:
    """Fetch ALL report_metadata records for an organization (no user filter).

    Used by org admins and viewers who should see all org reports.

    Returns:
        List of report metadata dicts (most recent first).
    """
    if _use_local_db():
        from models import ReportMetadata as RM
        db = _get_db()
        try:
            rows = (
                db.query(RM)
                .filter(RM.organization_id == org_id)
                .order_by(RM.generated_at.desc())
                .limit(limit)
                .all()
            )
            return [_row_to_dict(r) for r in rows]
        except Exception:
            return []
        finally:
            db.close()

    try:
        supabase = _get_supabase()
        result = supabase.table("report_metadata") \
            .select("*") \
            .eq("organization_id", org_id) \
            .order("generated_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Cached wrappers (reduce DB hits on Streamlit reruns)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False, max_entries=100)
def cached_list_org_files(org_id: str) -> list[dict]:
    """TTL-cached wrapper around list_org_files (refreshes every 30s)."""
    return list_org_files(org_id)


@st.cache_data(ttl=30, show_spinner=False, max_entries=100)
def cached_list_org_reports(org_id: str) -> list[dict]:
    """TTL-cached wrapper around list_org_reports (refreshes every 30s)."""
    return list_org_reports(org_id)


@st.cache_data(ttl=30, show_spinner=False, max_entries=100)
def cached_list_reports_for_org(org_id: str, limit: int = 200) -> list[dict]:
    """TTL-cached wrapper around list_reports_for_org."""
    return list_reports_for_org(org_id, limit)
