#type: ignore
"""
Daily report-generation quota system for Rotrix Dashboard.

Supports both local Postgres (SQLAlchemy) and Supabase backends.

Quota resolution: per-user override (profiles.daily_report_quota) > system default (100).
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Tuple

logger = logging.getLogger(__name__)

try:
    DEFAULT_DAILY_REPORT_QUOTA = int(os.getenv("DEFAULT_DAILY_REPORT_QUOTA", "100"))
except Exception:
    DEFAULT_DAILY_REPORT_QUOTA = 100
REPORT_COOLDOWN_SECONDS = 10


def _use_local_auth() -> bool:
    """Check if local PG backend is active."""
    from auth import USE_LOCAL_AUTH
    return USE_LOCAL_AUTH


def _get_db():
    """Return a SQLAlchemy session."""
    from models import SessionLocal
    return SessionLocal()


def _get_supabase_admin():
    """Return a Supabase client suitable for admin-style queries."""
    try:
        from auth import get_supabase_service, get_supabase

        svc = get_supabase_service()
        if svc is not None:
            return svc
        return get_supabase()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to get Supabase client for quotas: %s", exc)
        return None


def get_daily_report_count(user_id: str) -> int:
    """Count reports generated today (UTC) by this user."""
    if not user_id:
        return 0

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    if _use_local_auth():
        from models import ReportMetadata
        from sqlalchemy import func as sa_func
        db = _get_db()
        try:
            count = (
                db.query(sa_func.count(ReportMetadata.id))
                .filter(
                    ReportMetadata.user_id == user_id,
                    ReportMetadata.generated_at >= today_start,
                )
                .scalar()
            )
            return int(count or 0)
        except Exception as exc:
            logger.warning("Failed to count daily reports from PG: %s", exc)
            return 0
        finally:
            db.close()

    # Supabase path
    client = _get_supabase_admin()
    if client is None:
        return 0
    try:
        result = (
            client.table("report_metadata")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .gte("generated_at", today_start.isoformat())
            .execute()
        )
        count = getattr(result, "count", None)
        return int(count or 0)
    except Exception as exc:  # pragma: no cover - best-effort
        logger.warning("Failed to count daily reports from Supabase: %s", exc)
        return 0


def _read_user_quota(user_id: str) -> int | None:
    """Read per-user quota override from profiles.daily_report_quota."""
    if not user_id:
        return None

    if _use_local_auth():
        from models import Profile
        db = _get_db()
        try:
            profile = db.get(Profile, user_id)
            if profile is None:
                return None
            return profile.daily_report_quota
        except Exception:
            return None
        finally:
            db.close()

    # Supabase path
    client = _get_supabase_admin()
    if client is None:
        return None
    try:
        res = (
            client.table("profiles")
            .select("daily_report_quota")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        row = getattr(res, "data", None) or {}
        value = row.get("daily_report_quota")
        if value is None:
            return None
        return int(value)
    except Exception:  # pragma: no cover - schema / RLS variations
        return None


def get_user_quota(user_id: str, org_id: str) -> int:
    """Get the effective daily report quota for a user."""
    user_q = _read_user_quota(user_id)
    if user_q is not None:
        return user_q
    import streamlit as st
    try:
        if st.session_state.get("user_role") == "super_admin":
            return 100
    except Exception:
        pass

    # For now we use a single system-wide default; org-level overrides can be added later.
    return DEFAULT_DAILY_REPORT_QUOTA


def get_remaining_quota(user_id: str, org_id: str) -> int:
    """Get remaining number of reports this user can generate today."""
    quota = get_user_quota(user_id, org_id)
    used = get_daily_report_count(user_id)
    return max(0, quota - used)


def check_quota(user_id: str, org_id: str, num_files: int = 1) -> Tuple[bool, int, int, int]:
    """Check if user can generate num_files reports.

    Applies both a daily quota and a short-term cooldown to avoid bursts
    of many heavy reports in quick succession.
    """
    import time
    import streamlit as st

    # Short-term cooldown (per-session, best-effort)
    last_gen = st.session_state.get("_last_report_generated_at", 0)
    if last_gen:
        elapsed = time.time() - float(last_gen)
        if elapsed < REPORT_COOLDOWN_SECONDS:
            quota = get_user_quota(user_id, org_id)
            used = get_daily_report_count(user_id)
            remaining = max(0, quota - used)
            return False, remaining, quota, used

    quota = get_user_quota(user_id, org_id)
    used = get_daily_report_count(user_id)
    remaining = max(0, quota - used)
    if remaining >= num_files:
        # Only stamp cooldown when a generation is actually allowed
        try:
            st.session_state["_last_report_generated_at"] = time.time()
        except Exception:
            pass
    return remaining >= num_files, remaining, quota, used


def set_user_quota(user_id: str, quota: int | None) -> Tuple[bool, str]:
    """Set daily report quota override for a specific user."""
    if _use_local_auth():
        from models import Profile
        db = _get_db()
        try:
            profile = db.get(Profile, user_id)
            if profile is None:
                return False, "User profile not found."
            profile.daily_report_quota = quota
            db.commit()
            if quota is None:
                return True, "User quota override removed (using system default)."
            return True, f"User quota set to {quota}/day."
        except Exception as exc:
            db.rollback()
            return False, f"Failed to set user quota: {exc}"
        finally:
            db.close()

    # Supabase path
    client = _get_supabase_admin()
    if client is None:
        return False, "Supabase service key not configured; cannot update quotas."
    try:
        update_val = quota
        client.table("profiles").update(
            {"daily_report_quota": update_val}
        ).eq("id", user_id).execute()
        if quota is None:
            return True, "User quota override removed (using system default)."
        return True, f"User quota set to {quota}/day."
    except Exception as exc:  # pragma: no cover - best-effort
        return False, f"Failed to set user quota: {exc}"
