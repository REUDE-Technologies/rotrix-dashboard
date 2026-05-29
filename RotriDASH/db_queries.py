#type: ignore
"""
Shared database query helpers for admin dashboards.

Provides SQLAlchemy-based query functions that mirror the Supabase table queries
used in admin_dashboard.py and admin_org_dashboard.py. When USE_LOCAL_AUTH is true,
these functions are used instead of Supabase.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


def _get_db():
    """Return a SQLAlchemy session."""
    from models import SessionLocal
    return SessionLocal()


def _row_to_dict(row) -> dict:
    """Convert a SQLAlchemy model instance to a plain dict with string-ified special types."""
    import uuid as _uuid
    d = {}
    for col in row.__table__.columns:
        val = getattr(row, col.key)
        if isinstance(val, _uuid.UUID):
            val = str(val)
        elif isinstance(val, datetime):
            val = val.isoformat()
        d[col.key] = val
    return d


# ---------------------------------------------------------------------------
# Count helpers
# ---------------------------------------------------------------------------

def count_all_profiles() -> int:
    from models import Profile
    db = _get_db()
    try:
        return db.query(Profile).count()
    except Exception:
        return 0
    finally:
        db.close()


def count_all_files() -> int:
    from models import FileMetadata
    db = _get_db()
    try:
        return db.query(FileMetadata).count()
    except Exception:
        return 0
    finally:
        db.close()


def count_all_reports() -> int:
    from models import ReportMetadata
    db = _get_db()
    try:
        return db.query(ReportMetadata).count()
    except Exception:
        return 0
    finally:
        db.close()


def count_all_organizations() -> int:
    from models import Organization
    db = _get_db()
    try:
        return db.query(Organization).count()
    except Exception:
        return 0
    finally:
        db.close()


def count_profiles_where(org_id: str | None = None, profile_status: str | None = None,
                         created_after: str | None = None) -> int:
    from models import Profile
    db = _get_db()
    try:
        q = db.query(Profile)
        if org_id:
            q = q.filter(Profile.organization_id == org_id)
        if profile_status:
            q = q.filter(Profile.profile_status == profile_status)
        if created_after:
            q = q.filter(Profile.created_at >= created_after)
        return q.count()
    except Exception:
        return 0
    finally:
        db.close()


def count_files_where(org_id: str | None = None) -> int:
    from models import FileMetadata
    db = _get_db()
    try:
        q = db.query(FileMetadata)
        if org_id:
            q = q.filter(FileMetadata.organization_id == org_id)
        return q.count()
    except Exception:
        return 0
    finally:
        db.close()


def count_reports_where(org_id: str | None = None, generated_after: str | None = None) -> int:
    from models import ReportMetadata
    db = _get_db()
    try:
        q = db.query(ReportMetadata)
        if org_id:
            q = q.filter(ReportMetadata.organization_id == org_id)
        if generated_after:
            q = q.filter(ReportMetadata.generated_at >= generated_after)
        return q.count()
    except Exception:
        return 0
    finally:
        db.close()


def count_events_where(event_type: str | None = None, org_id: str | None = None,
                       created_after: str | None = None) -> int:
    from models import UsageEvent
    db = _get_db()
    try:
        q = db.query(UsageEvent)
        if event_type:
            q = q.filter(UsageEvent.event_type == event_type)
        if org_id:
            q = q.filter(UsageEvent.organization_id == org_id)
        if created_after:
            q = q.filter(UsageEvent.created_at >= created_after)
        return q.count()
    except Exception:
        return 0
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Active users (unique user_ids from login events)
# ---------------------------------------------------------------------------

def get_active_user_ids(since_iso: str, org_id: str | None = None) -> set:
    from models import UsageEvent
    db = _get_db()
    try:
        q = (
            db.query(UsageEvent.user_id)
            .filter(UsageEvent.event_type == "login")
            .filter(UsageEvent.created_at >= since_iso)
        )
        if org_id:
            q = q.filter(UsageEvent.organization_id == org_id)
        rows = q.all()
        return set(str(r[0]) for r in rows if r[0])
    except Exception:
        return set()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Fetch lists
# ---------------------------------------------------------------------------

def fetch_all_profiles(org_id: str | None = None, profile_status: str | None = None,
                       created_after: str | None = None, order_desc: bool = True,
                       limit: int | None = None) -> list[dict]:
    """Fetch profiles as list of dicts, optionally filtered."""
    from models import Profile, Organization
    db = _get_db()
    try:
        q = db.query(Profile)
        if org_id:
            q = q.filter(Profile.organization_id == org_id)
        if profile_status:
            q = q.filter(Profile.profile_status == profile_status)
        if created_after:
            q = q.filter(Profile.created_at >= created_after)
        if order_desc:
            q = q.order_by(Profile.created_at.desc())
        else:
            q = q.order_by(Profile.created_at.asc())
        if limit:
            q = q.limit(limit)
        rows = q.all()
        result = []
        org_cache = {}
        for p in rows:
            d = _row_to_dict(p)
            # Add organization name (join equivalent)
            org_name = ""
            if p.organization_id:
                if p.organization_id not in org_cache:
                    org = db.get(Organization, p.organization_id)
                    org_cache[p.organization_id] = org.name if org else ""
                org_name = org_cache[p.organization_id]
            d["organizations"] = {"name": org_name} if org_name else None
            result.append(d)
        return result
    except Exception:
        return []
    finally:
        db.close()


def fetch_profiles_by_ids(user_ids: list[str]) -> list[dict]:
    """Fetch profiles by a list of IDs."""
    from models import Profile, Organization
    if not user_ids:
        return []
    db = _get_db()
    try:
        rows = db.query(Profile).filter(Profile.id.in_(user_ids)).all()
        org_cache = {}
        result = []
        for p in rows:
            d = _row_to_dict(p)
            org_name = ""
            if p.organization_id:
                if p.organization_id not in org_cache:
                    org = db.get(Organization, p.organization_id)
                    org_cache[p.organization_id] = org.name if org else ""
                org_name = org_cache[p.organization_id]
            d["organizations"] = {"name": org_name} if org_name else None
            result.append(d)
        return result
    except Exception:
        return []
    finally:
        db.close()


def fetch_all_organizations() -> list[dict]:
    from models import Organization
    db = _get_db()
    try:
        rows = db.query(Organization).order_by(Organization.created_at.asc()).all()
        return [_row_to_dict(o) for o in rows]
    except Exception:
        return []
    finally:
        db.close()


def fetch_usage_events(start_iso: str, end_iso: str, org_id: str | None = None) -> list[dict]:
    from models import UsageEvent
    db = _get_db()
    try:
        q = (
            db.query(UsageEvent)
            .filter(UsageEvent.created_at >= start_iso)
            .filter(UsageEvent.created_at <= end_iso)
            .order_by(UsageEvent.created_at.asc())
        )
        if org_id:
            q = q.filter(UsageEvent.organization_id == org_id)
        rows = q.all()
        result = []
        for r in rows:
            d = _row_to_dict(r)
            # The column is stored as event_metadata in SQLAlchemy but called "metadata" in DB
            d["metadata"] = d.pop("event_metadata", {})
            result.append(d)
        return result
    except Exception:
        return []
    finally:
        db.close()


def fetch_file_metadata(start_iso: str, end_iso: str, org_id: str | None = None) -> list[dict]:
    from models import FileMetadata
    db = _get_db()
    try:
        q = (
            db.query(FileMetadata)
            .filter(FileMetadata.uploaded_at >= start_iso)
            .filter(FileMetadata.uploaded_at <= end_iso)
        )
        if org_id:
            q = q.filter(FileMetadata.organization_id == org_id)
        rows = q.all()
        return [_row_to_dict(r) for r in rows]
    except Exception:
        return []
    finally:
        db.close()


def fetch_all_file_metadata(org_id: str | None = None) -> list[dict]:
    """Fetch all file_metadata (no date filter), optionally org-scoped."""
    from models import FileMetadata, Organization
    db = _get_db()
    try:
        q = db.query(FileMetadata)
        if org_id:
            q = q.filter(FileMetadata.organization_id == org_id)
        rows = q.all()
        result = []
        org_cache = {}
        for r in rows:
            d = _row_to_dict(r)
            # Add org name for storage pie chart
            oid = r.organization_id
            if oid:
                if oid not in org_cache:
                    org = db.get(Organization, oid)
                    org_cache[oid] = org.name if org else "Unknown"
                d["organizations"] = {"name": org_cache[oid]}
            else:
                d["organizations"] = {"name": "Unknown"}
            result.append(d)
        return result
    except Exception:
        return []
    finally:
        db.close()


def fetch_report_metadata(start_iso: str, end_iso: str, org_id: str | None = None) -> list[dict]:
    from models import ReportMetadata
    db = _get_db()
    try:
        q = (
            db.query(ReportMetadata)
            .filter(ReportMetadata.generated_at >= start_iso)
            .filter(ReportMetadata.generated_at <= end_iso)
        )
        if org_id:
            q = q.filter(ReportMetadata.organization_id == org_id)
        rows = q.all()
        return [_row_to_dict(r) for r in rows]
    except Exception:
        return []
    finally:
        db.close()


def fetch_org_reports(org_id: str, limit: int = 200) -> list[dict]:
    """Fetch report_metadata for an org, with all columns needed by reports tab."""
    from models import ReportMetadata
    db = _get_db()
    try:
        rows = (
            db.query(ReportMetadata)
            .filter(ReportMetadata.organization_id == org_id)
            .order_by(ReportMetadata.generated_at.desc())
            .limit(limit)
            .all()
        )
        return [_row_to_dict(r) for r in rows]
    except Exception:
        return []
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------

def update_profile(user_id: str, data: dict) -> bool:
    from models import Profile
    db = _get_db()
    try:
        profile = db.get(Profile, user_id)
        if profile is None:
            return False
        for k, v in data.items():
            if hasattr(profile, k):
                setattr(profile, k, v)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def delete_profile(user_id: str) -> bool:
    from models import Profile
    db = _get_db()
    try:
        profile = db.get(Profile, user_id)
        if profile:
            db.delete(profile)
            db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def create_organization(name: str) -> dict | None:
    from models import Organization
    db = _get_db()
    try:
        org = Organization(name=name)
        db.add(org)
        db.commit()
        return _row_to_dict(org)
    except Exception:
        db.rollback()
        return None
    finally:
        db.close()


def update_organization(org_id: str, data: dict) -> bool:
    from models import Organization
    db = _get_db()
    try:
        org = db.get(Organization, org_id)
        if org is None:
            return False
        for k, v in data.items():
            if hasattr(org, k):
                setattr(org, k, v)
        db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def delete_organization(org_id: str) -> bool:
    from models import Organization
    db = _get_db()
    try:
        org = db.get(Organization, org_id)
        if org:
            db.delete(org)
            db.commit()
        return True
    except Exception:
        db.rollback()
        return False
    finally:
        db.close()


def create_local_user(email: str, password: str, full_name: str, role: str,
                      org_id: str) -> tuple[bool, str]:
    """Create a user in local PG auth (profile + hashed password)."""
    from models import Profile
    import bcrypt
    from sqlalchemy import select

    db = _get_db()
    try:
        existing = db.execute(
            select(Profile).where(Profile.email == email.strip().lower())
        ).scalar_one_or_none()
        if existing:
            return False, "A user with this email already exists."
        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        profile = Profile(
            email=email.strip().lower(),
            password_hash=hashed,
            full_name=full_name,
            role=role,
            organization_id=org_id,
            is_active=True,
            profile_status="approved",
            email_verified=True,
        )
        db.add(profile)
        db.commit()
        return True, f"User {full_name} ({email}) created successfully."
    except Exception as exc:
        db.rollback()
        return False, f"Failed to create user: {exc}"
    finally:
        db.close()
