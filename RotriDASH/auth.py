#type: ignore
"""
Authentication module for Rotrix Dashboard.

Handles Supabase Auth integration: login, logout, session management,
role-based access control, and profile setup / approval workflow.

Roles:
    - super_admin : Rotrix team — full platform access, cross-org visibility
    - admin       : Organization admin — manage users within their own org
    - editor      : Customer user — upload files, generate reports, view own org
    - viewer      : Customer user — read-only access to own org data

Profile statuses:
    - pending_setup    : Logged in but profile not yet filled
    - pending_approval : Profile submitted, awaiting super admin approval
    - approved         : Super admin approved — full dashboard access
    - rejected         : Super admin rejected — must re-submit
"""

import os
import time
import html as _html
import base64
from typing import Tuple, List

import bcrypt
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select

from models import SessionLocal, Profile, Organization
from supabase import create_client, Client

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
try:
    load_dotenv()
except OSError:
    pass  # e.g. Errno 24 too many open files; env may already be set

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Backend selector: "supabase" (default) or "local_pg"
AUTH_BACKEND = os.getenv("AUTH_BACKEND", "supabase").lower()
USE_LOCAL_AUTH = AUTH_BACKEND in ("local_pg", "local", "postgres")


def _get_db():
    """Return a SQLAlchemy session."""
    return SessionLocal()


def _hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Supabase client (per-session: no cache to avoid cross-session auth leakage)
# ---------------------------------------------------------------------------
def get_supabase() -> Client:
    """Return a Supabase client. Creates a new client each run and restores session from session_state if present."""
    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        st.error("⚠️ Supabase credentials not configured. Set SUPABASE_URL and SUPABASE_ANON_KEY in your .env file.")
        st.stop()
    client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    # Restore session for this user so table() calls use the correct auth
    access_token = st.session_state.get("supabase_session")
    refresh_token = st.session_state.get("supabase_refresh_token")
    if access_token:
        try:
            client.auth.set_session(access_token, refresh_token or "")
        except Exception:
            pass
    return client


def get_supabase_service() -> Client | None:
    """Return a Supabase client with service_role key for admin operations (bypasses RLS)."""
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        return None
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Login / Logout
# ---------------------------------------------------------------------------
def login(email: str, password: str) -> tuple[bool, str]:
    """Authenticate user with email + password.

    Returns:
        (success: bool, message: str)
    """
    # Local Postgres-backed auth path
    if USE_LOCAL_AUTH:
        email_norm = email.strip().lower()
        if not email_norm or not password:
            return False, "Please enter both email and password."
        db = _get_db()
        try:
            stmt = select(Profile).where(Profile.email == email_norm)
            profile = db.execute(stmt).scalar_one_or_none()
            if profile is None:
                return False, "Invalid credentials. Please check your email and password."
            if not profile.is_active:
                return False, "Your account has been deactivated. Contact your administrator."
            if not profile.password_hash or not _verify_password(password, profile.password_hash):
                return False, "Invalid credentials. Please check your email and password."

            # Session state mirrors the Supabase-based layout
            st.session_state.authenticated = True
            st.session_state.user_id = str(profile.id)
            st.session_state.user_email = profile.email
            st.session_state.user_name = profile.full_name or "New User"
            st.session_state.user_role = profile.role or "viewer"
            st.session_state.organization_id = str(profile.organization_id) if profile.organization_id else None
            st.session_state.profile_status = profile.profile_status or "pending_setup"

            org_name = ""
            if profile.organization_id:
                org = db.get(Organization, profile.organization_id)
                if org:
                    org_name = org.name or ""
            st.session_state.organization_name = org_name

            # Legacy author fields for report generation
            st.session_state.author_name = profile.full_name or ""
            st.session_state.author_email = profile.email
            st.session_state.author_company = org_name
            st.session_state.author_details_completed = True

            # Update last_login
            from datetime import datetime, timezone

            profile.last_login = datetime.now(timezone.utc)
            db.commit()

            # Persist local-session identifier in browser so reloads keep the user signed in.
            try:
                save_session_to_browser()
            except Exception:
                # Session persistence is a best-effort enhancement; never break login flow.
                pass

            # User just logged in successfully, allow future auto-restore again
            st.session_state["skip_auth_restore"] = False

            return True, "Login successful!"
        except Exception as exc:
            db.rollback()
            return False, f"Login failed: {exc}"
        finally:
            db.close()

    # Existing Supabase-based auth path (default)
    try:
        supabase = get_supabase()
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })

        user = response.user
        session = response.session

        if not user or not session:
            return False, "Invalid credentials. Please check your email and password."

        # Fetch profile (role, org, status, etc.)
        # Prefer service-role client so RLS does not block (e.g. super_admin profile already exists but RLS/join blocks user client)
        profile_data = None
        for client in [get_supabase_service(), get_supabase()]:
            if client is None:
                continue
            try:
                profile = client.table("profiles") \
                    .select("full_name, role, organization_id, is_active, profile_status, organizations(name)") \
                    .eq("id", user.id) \
                    .maybe_single() \
                    .execute()
                profile_data = getattr(profile, "data", None) if profile is not None else None
                if profile_data:
                    break
            except Exception:
                profile_data = None

        if not profile_data:
            return False, "User profile not found. Contact your administrator."

        if profile_data.get("is_active") is False:
            return False, "Your account has been deactivated. Contact your administrator."

        # Store in session state
        st.session_state.authenticated = True
        st.session_state.user_id = user.id
        st.session_state.user_email = user.email
        st.session_state.user_name = profile_data.get("full_name", "New User")
        st.session_state.user_role = profile_data.get("role", "viewer")
        st.session_state.organization_id = profile_data.get("organization_id")
        st.session_state.supabase_session = session.access_token
        st.session_state.supabase_refresh_token = getattr(session, "refresh_token", None) or ""
        # Store token expiry (Supabase default ~1h); used to auto-logout when expired
        expires_at = getattr(session, "expires_at", None)
        st.session_state.supabase_token_expires_at = expires_at if expires_at is not None else (int(time.time()) + 3600)
        st.session_state.profile_status = profile_data.get("profile_status", "pending_setup")

        # Organization name (from joined organizations table)
        org_data = profile_data.get("organizations")
        if org_data and isinstance(org_data, dict):
            st.session_state.organization_name = org_data.get("name", "")
        else:
            st.session_state.organization_name = ""

        # Also populate the legacy author fields for report generation
        st.session_state.author_name = profile_data.get("full_name", "")
        st.session_state.author_email = user.email
        st.session_state.author_company = st.session_state.organization_name
        st.session_state.author_details_completed = True

        return True, "Login successful!"

    except Exception as e:
        error_msg = str(e)
        # PostgREST 204 / missing profile response — show actionable message
        if "204" in error_msg or "Missing response" in error_msg or "Postgrest" in error_msg or "postgrest" in error_msg:
            return False, "User profile not found. Ask your administrator to create your profile (Supabase Dashboard → Table Editor → profiles) or check RLS policies."
        if "Invalid login credentials" in error_msg:
            return False, "Invalid email or password."
        elif "Email not confirmed" in error_msg:
            return False, "Please confirm your email address before logging in."
        else:
            return False, f"Login failed: {error_msg}"


def logout():
    """Sign out the current user: revoke Supabase session, clear browser localStorage, clear local state."""
    if not USE_LOCAL_AUTH:
        # Revoke the session server-side so stored tokens become invalid immediately
        try:
            supabase = get_supabase()
            supabase.auth.sign_out()
        except Exception:
            pass
    # Always clear browser localStorage tokens / identifiers for both backends
    clear_browser_session()
    # Mark that the user explicitly logged out so we don't auto-restore on the next rerun
    st.session_state["skip_auth_restore"] = True
    # Clear all other session keys so app re-initializes to home page
    for _key in list(st.session_state.keys()):
        if _key != "skip_auth_restore":
            try:
                del st.session_state[_key]
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Browser session persistence (localStorage via streamlit-js-eval)
# ---------------------------------------------------------------------------

def save_session_to_browser() -> None:
    """Persist auth session details to browser localStorage so reloads keep the user signed in.

    - For Supabase auth, we store access/refresh tokens and expiry.
    - For local Postgres auth, we store the user_id only and re-hydrate the profile from the DB.
    """
    # Local Postgres-backed auth: store user_id only
    if USE_LOCAL_AUTH:
        user_id = st.session_state.get("user_id")
        if not user_id:
            return
        try:
            from streamlit_js_eval import streamlit_js_eval  # type: ignore
            counter = st.session_state.get("_ls_save_ctr", 0) + 1
            st.session_state["_ls_save_ctr"] = counter
            js = (
                f"localStorage.setItem('r_tok','{user_id}');"
                "localStorage.removeItem('r_ref');"
                "localStorage.removeItem('r_exp');"
                "true;"
            )
            streamlit_js_eval(js_expressions=js, key=f"_sjs_save_local_{counter}")
        except ImportError:
            pass
        except Exception:
            pass
        return

    # Supabase auth: persist access/refresh tokens
    access_token = st.session_state.get("supabase_session", "")
    if not access_token:
        return
    try:
        from streamlit_js_eval import streamlit_js_eval  # type: ignore
        refresh_token = st.session_state.get("supabase_refresh_token", "") or ""
        expires_at = st.session_state.get("supabase_token_expires_at", int(time.time()) + 3600) or 0
        counter = st.session_state.get("_ls_save_ctr", 0) + 1
        st.session_state["_ls_save_ctr"] = counter
        js = (
            f"localStorage.setItem('r_tok','{access_token}');"
            f"localStorage.setItem('r_ref','{refresh_token}');"
            f"localStorage.setItem('r_exp','{int(expires_at)}');"
            "true;"
        )
        streamlit_js_eval(js_expressions=js, key=f"_sjs_save_{counter}")
    except ImportError:
        pass
    except Exception:
        pass


def clear_browser_session() -> None:
    """Remove auth tokens from browser localStorage (called on logout)."""
    try:
        from streamlit_js_eval import streamlit_js_eval  # type: ignore
        counter = st.session_state.get("_ls_clear_ctr", 0) + 1
        st.session_state["_ls_clear_ctr"] = counter
        js = (
            "localStorage.removeItem('r_tok');"
            "localStorage.removeItem('r_ref');"
            "localStorage.removeItem('r_exp');"
            "true;"
        )
        streamlit_js_eval(js_expressions=js, key=f"_sjs_clear_{counter}")
    except ImportError:
        pass
    except Exception:
        pass


def try_restore_session_from_browser() -> None:
    """On page load, try to restore session from browser localStorage.

    streamlit-js-eval is async: first call returns None and fires JS in browser;
    second call (after JS sends back the value) returns the stored string.
    This causes at most one sub-second flicker before restoring the session.
    """
    # Local Postgres-backed auth: restore from stored user_id if present,
    # unless the user explicitly logged out in this session.
    if USE_LOCAL_AUTH:
        if is_authenticated() or st.session_state.get("skip_auth_restore", False):
            return
        try:
            from streamlit_js_eval import streamlit_js_eval  # type: ignore
            stored_user_id = streamlit_js_eval(
                js_expressions="localStorage.getItem('r_tok')",
                key="_sjs_r_tok_local",
            )
            if not stored_user_id or stored_user_id in (None, "null", "undefined", ""):
                return

            db = _get_db()
            try:
                profile = db.get(Profile, stored_user_id)
                if profile is None or not profile.is_active:
                    return

                # Mirror local login session_state setup
                st.session_state.authenticated = True
                st.session_state.user_id = str(profile.id)
                st.session_state.user_email = profile.email
                st.session_state.user_name = profile.full_name or "New User"
                st.session_state.user_role = profile.role or "viewer"
                st.session_state.organization_id = (
                    str(profile.organization_id) if profile.organization_id else None
                )
                st.session_state.profile_status = profile.profile_status or "pending_setup"

                org_name = ""
                if profile.organization_id:
                    org = db.get(Organization, profile.organization_id)
                    if org:
                        org_name = org.name or ""
                st.session_state.organization_name = org_name
                st.session_state.author_name = profile.full_name or ""
                st.session_state.author_email = profile.email
                st.session_state.author_company = org_name
                st.session_state.author_details_completed = True

                st.rerun()
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        except ImportError:
            # streamlit-js-eval not installed; nothing to do
            return
        except Exception:
            # On any error, don't break the app; simply fall back to logged-out state
            return
        return

    if is_authenticated():
        return
    try:
        from streamlit_js_eval import streamlit_js_eval  # type: ignore
        stored_token = streamlit_js_eval(js_expressions="localStorage.getItem('r_tok')", key="_sjs_r_tok")
        stored_refresh = streamlit_js_eval(js_expressions="localStorage.getItem('r_ref')", key="_sjs_r_ref")
        stored_expires = streamlit_js_eval(js_expressions="localStorage.getItem('r_exp')", key="_sjs_r_exp")

        if not stored_token or stored_token in (None, "null", "undefined", ""):
            return

        # Populate session state so get_supabase() can set the session
        st.session_state.supabase_session = stored_token
        st.session_state.supabase_refresh_token = stored_refresh or ""
        try:
            st.session_state.supabase_token_expires_at = int(stored_expires)
        except (TypeError, ValueError):
            st.session_state.supabase_token_expires_at = int(time.time()) + 3600

        # Validate token via set_session (this also refreshes if needed)
        supabase = get_supabase()
        # set_session will attempt to use the refresh token if the access token is expired
        try:
            refreshed = supabase.auth.set_session(stored_token, stored_refresh or "")
            if refreshed and getattr(refreshed, "access_token", None):
                st.session_state.supabase_session = refreshed.access_token
                st.session_state.supabase_refresh_token = getattr(
                    refreshed, "refresh_token", stored_refresh
                ) or stored_refresh or ""
                exp = getattr(refreshed, "expires_at", None)
                st.session_state.supabase_token_expires_at = (
                    exp if exp else int(time.time()) + 3600
                )
        except Exception:
            pass  # Fall through to get_user validation below

        user_resp = supabase.auth.get_user()
        if not (user_resp and user_resp.user):
            raise ValueError("Invalid token")

        user = user_resp.user
        try:
            profile = (
                supabase.table("profiles")
                .select(
                    "full_name, role, organization_id, is_active, profile_status, organizations(name)"
                )
                .eq("id", user.id)
                .maybe_single()
                .execute()
            )
            profile_data = getattr(profile, "data", None) if profile is not None else None
        except Exception:
            raise ValueError("No profile found")

        if not profile_data:
            raise ValueError("No profile found")
        if profile_data.get("is_active") is False:
            raise ValueError("Account deactivated")

        # Restore full session
        st.session_state.authenticated = True
        st.session_state.user_id = user.id
        st.session_state.user_email = user.email
        st.session_state.user_name = profile_data.get("full_name", "New User")
        st.session_state.user_role = profile_data.get("role", "viewer")
        st.session_state.organization_id = profile_data.get("organization_id")
        st.session_state.profile_status = profile_data.get("profile_status", "pending_setup")
        org_data = profile_data.get("organizations")
        st.session_state.organization_name = (
            org_data.get("name", "") if isinstance(org_data, dict) else ""
        )
        st.session_state.author_name = profile_data.get("full_name", "")
        st.session_state.author_email = user.email
        st.session_state.author_company = st.session_state.organization_name
        st.session_state.author_details_completed = True

        # Persist any refreshed token to browser localStorage
        current_tok = st.session_state.get("supabase_session", "")
        if current_tok and current_tok != stored_token:
            save_session_to_browser()

        st.rerun()

    except ImportError:
        pass  # streamlit-js-eval not installed; silent no-op
    except Exception:
        # Token invalid or revoked — clear localStorage and any partial state
        for key in (
            "supabase_session",
            "supabase_refresh_token",
            "supabase_token_expires_at",
            "authenticated",
        ):
            st.session_state.pop(key, None)
        try:
            from streamlit_js_eval import streamlit_js_eval  # type: ignore
            counter = st.session_state.get("_ls_clear_ctr", 0) + 1
            st.session_state["_ls_clear_ctr"] = counter
            streamlit_js_eval(
                js_expressions=(
                    "localStorage.removeItem('r_tok');"
                    "localStorage.removeItem('r_ref');"
                    "localStorage.removeItem('r_exp');"
                    "true;"
                ),
                key=f"_sjs_clear_err_{counter}",
            )
        except Exception:
            pass


def signup(email: str, password: str) -> tuple[bool, str]:
    """Create a new account.

    For local auth, this creates a Profile row in Postgres.
    For Supabase backend, it delegates to Supabase Auth.
    """
    email_norm = email.strip().lower()
    if not email_norm:
        return False, "Please enter your email."
    if not password or len(password) < 6:
        return False, "Password must be at least 6 characters."

    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            existing = db.execute(
                select(Profile).where(Profile.email == email_norm)
            ).scalar_one_or_none()
            if existing is not None:
                return False, "An account with this email already exists. Try signing in instead."

            hashed = _hash_password(password)
            profile = Profile(
                email=email_norm,
                password_hash=hashed,
                full_name="",
                role="viewer",
                profile_status="pending_setup",
                is_active=True,
                email_verified=True,
            )
            db.add(profile)
            db.commit()
            return True, "Account created. You can now sign in with your email and password."
        except Exception as exc:
            db.rollback()
            return False, f"Sign up failed: {exc}"
        finally:
            db.close()

    # Supabase-based signup (legacy path)
    try:
        supabase = get_supabase()
        redirect_url = os.getenv("APP_URL", "http://localhost:8502")
        result = supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": {
                "email_redirect_to": redirect_url
            }
        })

        # Supabase returns a "fake" user for existing emails (no new account). Check identities.
        if result.user and getattr(result.user, "identities", None) and len(result.user.identities) > 0:
            return True, (
                "✅ Account created! Please check your email to verify your address, "
                "then come back and sign in."
            )
        elif result.user:
            return False, "An account with this email may already exist. Try signing in instead."
        else:
            return False, "Could not create account. Please try again."

    except Exception as e:
        error_msg = str(e)
        if "already registered" in error_msg.lower() or "already been registered" in error_msg.lower():
            return False, "An account with this email already exists. Try signing in instead."
        elif "valid email" in error_msg.lower():
            return False, "Please enter a valid email address."
        elif "at least" in error_msg.lower() or "password" in error_msg.lower():
            return False, "Password must be at least 6 characters."
        else:
            return False, f"Sign up failed: {error_msg}"


def reset_password(email: str) -> tuple[bool, str]:
    """Handle password reset request."""
    if USE_LOCAL_AUTH:
        # Minimal implementation for now: no email flow.
        return False, "Password reset via email is not configured for local auth. Please contact your administrator."

    try:
        supabase = get_supabase()
        redirect_url = os.getenv("APP_URL", "http://localhost:8502")
        supabase.auth.reset_password_email(email, options={"redirect_to": redirect_url})
        return True, (
            "📧 If an account with that email exists, you'll receive a password reset link shortly. "
            "Check your inbox (and spam folder)."
        )
    except Exception as e:
        return False, f"Could not send reset email: {e}"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

# Buffer in seconds: attempt a refresh if the access token expires within this window
_TOKEN_REFRESH_BUFFER_SECS = 300  # 5 minutes


def _try_refresh_token() -> bool:
    """Attempt to refresh the Supabase access token using the stored refresh token.

    Returns True if the refresh succeeded and session_state was updated with fresh tokens.
    Returns False if refresh failed or no refresh token exists.
    """
    if USE_LOCAL_AUTH:
        # Local auth does not use access/refresh tokens.
        return False
    refresh_token = st.session_state.get("supabase_refresh_token", "")
    if not refresh_token:
        return False
    try:
        client = get_supabase()
        # Use set_session with old access token + refresh token to trigger a refresh
        old_access = st.session_state.get("supabase_session", "")
        new_session = client.auth.set_session(old_access, refresh_token)
        if new_session and getattr(new_session, "access_token", None):
            st.session_state.supabase_session = new_session.access_token
            st.session_state.supabase_refresh_token = getattr(new_session, "refresh_token", refresh_token) or refresh_token
            exp = getattr(new_session, "expires_at", None)
            st.session_state.supabase_token_expires_at = exp if exp else int(time.time()) + 3600
            # Persist refreshed tokens to browser localStorage
            save_session_to_browser()
            return True
    except Exception:
        pass
    return False


def is_authenticated() -> bool:
    """Check if the user is currently logged in.

    If the access token is expired or close to expiring, attempts a proactive
    refresh using the Supabase refresh token. Only clears auth state if the
    refresh also fails.
    """
    if not st.session_state.get("authenticated", False):
        return False
    if USE_LOCAL_AUTH:
        # Local auth: rely solely on session_state flag.
        return True
    expires_at = st.session_state.get("supabase_token_expires_at")
    if expires_at is not None and time.time() > (expires_at - _TOKEN_REFRESH_BUFFER_SECS):
        # Token expired or about to expire — try refreshing
        if _try_refresh_token():
            return True
        # Refresh failed — clear auth state
        for key in ("authenticated", "user_id", "user_email", "user_name", "user_role",
                    "organization_id", "organization_name", "supabase_session", "supabase_refresh_token",
                    "supabase_token_expires_at", "profile_status"):
            st.session_state.pop(key, None)
        return False
    return True


def get_profile_status() -> str:
    """Return the current user's profile status."""
    return st.session_state.get("profile_status", "pending_setup")


def is_approved() -> bool:
    """Check if the current user's profile has been approved."""
    return is_authenticated() and get_profile_status() == "approved"


def sync_profile_status_from_db() -> None:
    """Refetch current user's profile from DB and update session state.

    Ensures that after a super_admin approves the user, the next run shows
    the correct status (approved) so the user lands on the dashboard instead
    of the pending-approval or setup screen.
    """
    if not is_authenticated():
        return
    user_id = st.session_state.get("user_id")
    if not user_id:
        return
    if USE_LOCAL_AUTH:
        try:
            db = _get_db()
            profile = db.get(Profile, user_id)
            if profile is None:
                return
            st.session_state.profile_status = profile.profile_status or "pending_setup"
            st.session_state.user_name = profile.full_name or st.session_state.get("user_name", "")
            st.session_state.user_role = profile.role or st.session_state.get("user_role", "viewer")
            st.session_state.organization_id = str(profile.organization_id) if profile.organization_id else None
            org_name = ""
            if profile.organization_id:
                org = db.get(Organization, profile.organization_id)
                if org:
                    org_name = org.name or ""
            st.session_state.organization_name = org_name
            st.session_state.author_name = profile.full_name or st.session_state.get("author_name", "")
            st.session_state.author_company = org_name
        except Exception:
            pass
        finally:
            try:
                db.close()
            except Exception:
                pass
        return

    try:
        supabase = get_supabase()
        profile = supabase.table("profiles") \
            .select("full_name, role, organization_id, profile_status, organizations(name)") \
            .eq("id", user_id) \
            .maybe_single() \
            .execute()
        profile_data = getattr(profile, "data", None) if profile is not None else None
        if not profile_data:
            return
        st.session_state.profile_status = profile_data.get("profile_status", "pending_setup")
        st.session_state.user_name = profile_data.get("full_name", st.session_state.get("user_name", ""))
        st.session_state.user_role = profile_data.get("role", st.session_state.get("user_role", "viewer"))
        st.session_state.organization_id = profile_data.get("organization_id")
        org_data = profile_data.get("organizations")
        st.session_state.organization_name = org_data.get("name", "") if isinstance(org_data, dict) else ""
        st.session_state.author_name = profile_data.get("full_name", st.session_state.get("author_name", ""))
        st.session_state.author_company = st.session_state.organization_name
    except Exception:
        pass


def get_current_user() -> dict | None:
    """Return the current user's profile as a dict, or None if not logged in."""
    if not is_authenticated():
        return None
    return {
        "user_id": st.session_state.get("user_id"),
        "email": st.session_state.get("user_email", ""),
        "name": st.session_state.get("user_name", ""),
        "role": st.session_state.get("user_role", ""),
        "organization_id": st.session_state.get("organization_id"),
        "organization_name": st.session_state.get("organization_name", ""),
        "profile_status": st.session_state.get("profile_status", "pending_setup"),
    }


def check_role(required_roles: list[str]) -> bool:
    """Check if the current user's role is in the required_roles list.

    Args:
        required_roles: e.g. ['super_admin', 'editor']

    Returns:
        True if user has one of the required roles.
    """
    if not is_authenticated():
        return False
    current_role = st.session_state.get("user_role", "")
    return current_role in required_roles


# ---------------------------------------------------------------------------
# Profile setup / update
# ---------------------------------------------------------------------------
def submit_profile(full_name: str, organization_name: str, role: str = "viewer") -> tuple[bool, str]:
    """Submit or update the user's profile.

    Creates the organization if it doesn't exist yet.

    Returns:
        (success: bool, message: str)
    """
    user_id = st.session_state.get("user_id")
    if not user_id:
        return False, "Not authenticated."

    # Server-side role validation
    if role not in ("viewer", "editor", "org_admin", "super_admin"):
        return False, "Invalid role."

    # Enforce REUDE Technologies restriction for super_admin
    if role == "super_admin" and organization_name.strip().lower() != "reude technologies":
        return False, "Super Admin role is restricted to REUDE Technologies organization."

    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            # 1. Find or create organization
            org_name_clean = organization_name.strip()
            org = db.execute(
                select(Organization).where(Organization.name == org_name_clean)
            ).scalar_one_or_none()
            if org is None:
                org = Organization(name=org_name_clean)
                db.add(org)
                db.flush()

            # 2. Update profile
            profile = db.get(Profile, user_id)
            if profile is None:
                return False, "Profile not found."
            current_role = profile.role or st.session_state.get("user_role", "viewer")
            current_org_id = str(profile.organization_id) if profile.organization_id else None
            requested_role = current_role if current_role in ("super_admin", "org_admin") else role

            profile.full_name = full_name.strip()
            profile.organization_id = org.id
            if current_role not in ("super_admin", "org_admin"):
                profile.role = requested_role

            # Name-only edits should not require approval again.
            # Re-approval is required only when role/org changes for non-admin users,
            # or when a user is still in initial setup.
            role_changed = requested_role != (current_role or "viewer")
            org_changed = str(org.id) != (current_org_id or "")
            if profile.profile_status == "pending_setup":
                next_status = "pending_approval"
            elif current_role in ("super_admin", "org_admin"):
                next_status = profile.profile_status or "approved"
            else:
                next_status = "pending_approval" if (role_changed or org_changed) else (profile.profile_status or "approved")
            profile.profile_status = next_status

            db.commit()

            # 3. Update session state
            st.session_state.user_name = full_name.strip()
            st.session_state.organization_id = str(org.id)
            if st.session_state.get("user_role") not in ("super_admin", "org_admin"):
                st.session_state.user_role = requested_role
            st.session_state.organization_name = org.name
            st.session_state.profile_status = next_status

            # Legacy fields
            st.session_state.author_name = full_name.strip()
            st.session_state.author_company = org.name

            if next_status == "pending_approval":
                return True, "Profile submitted for approval!"
            return True, "Profile updated successfully."
        except Exception as exc:
            db.rollback()
            return False, f"Failed to update profile: {exc}"
        finally:
            db.close()

    try:
        supabase = get_supabase()
        # 1. Find or create organization (use service client so we see all orgs and avoid duplicate key)
        service_client = get_supabase_service()
        if not service_client:
            return False, "Server configuration error (service key not set)."
        org_result = service_client.table("organizations") \
            .select("id, name") \
            .eq("name", organization_name.strip()) \
            .execute()

        if org_result.data:
            org_id = org_result.data[0]["id"]
        else:
            new_org = service_client.table("organizations") \
                .insert({"name": organization_name.strip()}) \
                .execute()
            if not new_org.data:
                return False, "Failed to create organization."
            org_id = new_org.data[0]["id"]

        # 2. Read current profile to decide whether re-approval is needed
        current_profile_res = (
            service_client.table("profiles")
            .select("role, organization_id, profile_status")
            .eq("id", user_id)
            .maybe_single()
            .execute()
        )
        current_profile = getattr(current_profile_res, "data", None) or {}
        current_role = current_profile.get("role") or st.session_state.get("user_role", "viewer")
        current_org_id = current_profile.get("organization_id")
        requested_role = current_role if current_role in ("super_admin", "org_admin") else role
        role_changed = requested_role != (current_role or "viewer")
        org_changed = str(org_id) != str(current_org_id) if current_org_id is not None else True
        current_status = current_profile.get("profile_status") or st.session_state.get("profile_status", "pending_setup")
        if current_status == "pending_setup":
            next_status = "pending_approval"
        elif current_role in ("super_admin", "org_admin"):
            next_status = current_status
        else:
            next_status = "pending_approval" if (role_changed or org_changed) else (current_status or "approved")

        # 3. Update profile
        update_data = {
            "full_name": full_name.strip(),
            "organization_id": org_id,
            "profile_status": next_status,
        }
        if st.session_state.get("user_role") not in ("super_admin", "org_admin"):
            update_data["role"] = requested_role
        supabase.table("profiles") \
            .update(update_data) \
            .eq("id", user_id) \
            .execute()

        # 4. Update session state (keep super_admin / org_admin role in session)
        st.session_state.user_name = full_name.strip()
        st.session_state.organization_id = org_id
        if st.session_state.get("user_role") not in ("super_admin", "org_admin"):
            st.session_state.user_role = requested_role
        st.session_state.organization_name = organization_name.strip()
        st.session_state.profile_status = next_status

        # Legacy fields
        st.session_state.author_name = full_name.strip()
        st.session_state.author_company = organization_name.strip()

        if next_status == "pending_approval":
            return True, "Profile submitted for approval!"
        return True, "Profile updated successfully."

    except Exception as e:
        return False, f"Failed to update profile: {e}"


def approve_profile(user_id: str) -> tuple[bool, str]:
    """Approve a user's profile. Allowed for super_admin (any user) or org_admin (own org only).
    Org admin approval also enforces the organization's max_users quota.
    Uses service role to bypass RLS."""
    caller_role = st.session_state.get("user_role", "")
    if caller_role not in ("super_admin", "org_admin"):
        return False, "Unauthorized."

    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            profile = db.get(Profile, user_id)
            if profile is None:
                return False, "Profile not found."
            # If caller is org_admin, verify target user belongs to same org and check quota
            if caller_role == "org_admin":
                caller_org = st.session_state.get("organization_id")
                if not caller_org or str(profile.organization_id) != str(caller_org):
                    return False, "You can only approve users in your own organization."
                quota_ok, quota_msg = _check_org_quota(caller_org)
                if not quota_ok:
                    return False, quota_msg
            profile.profile_status = "approved"
            db.commit()
            return True, "Profile approved."
        except Exception as exc:
            db.rollback()
            return False, f"Failed to approve: {exc}"
        finally:
            db.close()

    supabase = get_supabase_service()
    if not supabase:
        return False, "Server configuration error (service key not set)."
    try:
        # If caller is org_admin, verify target user belongs to same org and check quota
        if caller_role == "org_admin":
            caller_org = st.session_state.get("organization_id")
            target = supabase.table("profiles").select("organization_id").eq("id", user_id).maybe_single().execute()
            target_data = getattr(target, "data", None)
            if not target_data or target_data.get("organization_id") != caller_org:
                return False, "You can only approve users in your own organization."
            # Enforce max_users quota
            quota_ok, quota_msg = _check_org_quota(caller_org)
            if not quota_ok:
                return False, quota_msg
        supabase.table("profiles") \
            .update({"profile_status": "approved"}) \
            .eq("id", user_id) \
            .execute()
        return True, "Profile approved."
    except Exception as e:
        return False, f"Failed to approve: {e}"


def reject_profile(user_id: str) -> tuple[bool, str]:
    """Reject a user's profile. Allowed for super_admin (any user) or org_admin (own org only).
    Uses service role to bypass RLS."""
    caller_role = st.session_state.get("user_role", "")
    if caller_role not in ("super_admin", "org_admin"):
        return False, "Unauthorized."

    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            profile = db.get(Profile, user_id)
            if profile is None:
                return False, "Profile not found."
            if caller_role == "org_admin":
                caller_org = st.session_state.get("organization_id")
                if not caller_org or str(profile.organization_id) != str(caller_org):
                    return False, "You can only reject users in your own organization."
            profile.profile_status = "rejected"
            db.commit()
            return True, "Profile rejected."
        except Exception as exc:
            db.rollback()
            return False, f"Failed to reject: {exc}"
        finally:
            db.close()

    supabase = get_supabase_service()
    if not supabase:
        return False, "Server configuration error (service key not set)."
    try:
        # If caller is org_admin, verify target user belongs to same org
        if caller_role == "org_admin":
            caller_org = st.session_state.get("organization_id")
            target = supabase.table("profiles").select("organization_id").eq("id", user_id).maybe_single().execute()
            target_data = getattr(target, "data", None)
            if not target_data or target_data.get("organization_id") != caller_org:
                return False, "You can only reject users in your own organization."
        supabase.table("profiles") \
            .update({"profile_status": "rejected"}) \
            .eq("id", user_id) \
            .execute()
        return True, "Profile rejected."
    except Exception as e:
        return False, f"Failed to reject: {e}"


def set_own_profile_pending_setup() -> bool:
    """Set current user's profile_status to pending_setup (e.g. after rejection so they can re-submit)."""
    user_id = st.session_state.get("user_id")
    if not user_id:
        return False
    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            profile = db.get(Profile, user_id)
            if profile is None:
                return False
            profile.profile_status = "pending_setup"
            db.commit()
            return True
        except Exception:
            db.rollback()
            return False
        finally:
            db.close()
    try:
        supabase = get_supabase()
        supabase.table("profiles").update({"profile_status": "pending_setup"}).eq("id", user_id).execute()
        return True
    except Exception:
        return False


def get_pending_profiles(org_id: str | None = None) -> list[dict]:
    """Fetch profiles with status = pending_approval.

    Args:
        org_id: If given, only return profiles belonging to this organization.
                Used by org admins to see only their own org's pending users.
                If None, returns all pending profiles (for super_admin).

    Uses service role so admin sees pending users regardless of RLS.
    """
    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            stmt = select(Profile).where(Profile.profile_status == "pending_approval")
            if org_id:
                stmt = stmt.where(Profile.organization_id == org_id)
            stmt = stmt.order_by(Profile.created_at.desc())
            rows: List[Profile] = [r[0] for r in db.execute(stmt).all()]
            out: list[dict] = []
            # Preload organizations for name lookup
            org_cache: dict = {}
            for p in rows:
                org_name = ""
                if p.organization_id:
                    if p.organization_id in org_cache:
                        org_name = org_cache[p.organization_id]
                    else:
                        org = db.get(Organization, p.organization_id)
                        org_name = org.name if org else ""
                        org_cache[p.organization_id] = org_name
                out.append({
                    "id": str(p.id),
                    "email": p.email,
                    "full_name": p.full_name,
                    "role": p.role,
                    "profile_status": p.profile_status,
                    "organization_id": str(p.organization_id) if p.organization_id else None,
                    "organizations": {"name": org_name} if org_name else None,
                })
            return out
        except Exception:
            return []
        finally:
            db.close()

    try:
        supabase = get_supabase_service()
        if not supabase:
            return []
        query = supabase.table("profiles") \
            .select("id, email, full_name, role, profile_status, organization_id, organizations(name)") \
            .eq("profile_status", "pending_approval") \
            .order("created_at", desc=True)
        if org_id:
            query = query.eq("organization_id", org_id)
        result = query.execute()
        return result.data or []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Organization-scoped helpers (for admin role)
# ---------------------------------------------------------------------------

def is_org_admin() -> bool:
    """Return True if the current user is an org_admin of their organization."""
    if not is_authenticated():
        return False
    return st.session_state.get("user_role") == "org_admin"


def get_org_user_count(org_id: str) -> int:
    """Count approved users in an organization."""
    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            stmt = select(Profile).where(
                Profile.organization_id == org_id,
                Profile.profile_status == "approved",
            )
            count = db.execute(stmt).scalars().unique().count()
            return int(count)
        except Exception:
            return 0
        finally:
            db.close()
    try:
        supabase = get_supabase_service()
        if not supabase:
            return 0
        result = supabase.table("profiles") \
            .select("id", count="exact") \
            .eq("organization_id", org_id) \
            .eq("profile_status", "approved") \
            .execute()
        return result.count or 0
    except Exception:
        return 0


def get_org_max_users(org_id: str) -> int:
    """Fetch the max_users quota for an organization. Defaults to 50 if not set."""
    if USE_LOCAL_AUTH:
        db = _get_db()
        try:
            org = db.get(Organization, org_id)
            if not org:
                return 50
            return org.max_users or 50
        except Exception:
            return 50
        finally:
            db.close()
    try:
        supabase = get_supabase_service()
        if not supabase:
            return 50
        result = supabase.table("organizations") \
            .select("max_users") \
            .eq("id", org_id) \
            .maybe_single() \
            .execute()
        data = getattr(result, "data", None)
        if data:
            return data.get("max_users", 50) or 50
        return 50
    except Exception:
        return 50


def _check_org_quota(org_id: str) -> tuple[bool, str]:
    """Check if the organization has room for another approved user.
    Returns (ok, message)."""
    current = get_org_user_count(org_id)
    max_users = get_org_max_users(org_id)
    if current >= max_users:
        return False, f"Organization has reached its user limit ({current}/{max_users}). Contact your Rotrix administrator to increase the quota."
    return True, ""


# ---------------------------------------------------------------------------
# Auth gate — no longer blocks the full app, just checks status
# ---------------------------------------------------------------------------
def require_auth():
    """Legacy function — kept for backwards compatibility but does nothing.

    The home page is now public. Login is triggered by the Sign In button.
    """
    pass


# ---------------------------------------------------------------------------
# Login panel (right-side, slideshow left + form right)
# ---------------------------------------------------------------------------
def render_login_panel():
    """Render the login/signup/forgot-password panel.

    Switches between 3 modes based on st.session_state.login_mode:
      - 'signin':          Email + Password → Sign In
      - 'signup':           Email + Password + Confirm → Create Account
      - 'forgot_password':  Email → Send Reset Link
    """
    mode = st.session_state.get("login_mode", "signin")

    # Load slide images
    slide_data_urls = _load_slide_images()

    st.markdown("""
    <style>
    /* ---- Login panel professional styling ---- */
    .login-panel-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.2rem 0;
        text-align: left;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }
    .login-panel-sub {
        font-size: 0.9rem;
        color: #64748b;
        margin: 0 0 1.25rem 0;
        font-weight: 400;
    }
    /* Primary CTA button: gradient blue */
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button[kind="primary"],
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button[data-testid*="primary"] {
        background: linear-gradient(135deg, #0d3d56 0%, #154360 35%, #2E86C1 100%) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(13, 61, 86, 0.25) !important;
        border-radius: 8px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        margin-top: 0.5rem !important;
    }
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button[kind="primary"]:hover,
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button[data-testid*="primary"]:hover {
        box-shadow: 0 6px 20px rgba(13, 61, 86, 0.35) !important;
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
    /* Secondary text-link buttons — look like links, not buttons */
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button[kind="secondary"],
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button:not([data-testid*="primary"]):not([kind="primary"]) {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        color: #2E86C1 !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.3rem 0 !important;
        text-decoration: none !important;
        cursor: pointer !important;
        margin: 0 !important;
        min-height: 0 !important;
        height: auto !important;
    }
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button[kind="secondary"]:hover,
    div[data-testid="column"]:has(.login-form-cta-wrap) .stButton > button:not([data-testid*="primary"]):not([kind="primary"]):hover {
        color: #154360 !important;
        text-decoration: underline !important;
        background: transparent !important;
    }
    /* Divider line between primary and secondary links */
    .login-links-divider {
        border-top: 1px solid #e2e8f0;
        margin: 1rem 0 0.75rem 0;
    }
    .login-links-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 0.5rem;
    }
    .login-footer-text {
        text-align: center;
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    /* Slideshow */
    .author-slideshow-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 4/3;
        min-height: 320px;
        max-height: 580px;
        margin: 0;
        padding: 0;
        overflow: hidden;
        border-radius: 12px;
        box-sizing: border-box;
        display: block;
    }
    .author-slideshow-wrap img {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        margin: 0; padding: 0; border: none;
        display: block;
        object-fit: cover;
        object-position: center;
        opacity: 0;
        will-change: transform, opacity;
        backface-visibility: hidden;
        transform: translateZ(0);
    }
    .author-slideshow-wrap img.slide-img-1 { animation: author-show1 15s infinite cubic-bezier(0.45, 0, 0.55, 1); }
    .author-slideshow-wrap img.slide-img-2 { animation: author-show2 15s infinite cubic-bezier(0.45, 0, 0.55, 1); }
    .author-slideshow-wrap img.slide-img-3 { animation: author-show3 15s infinite cubic-bezier(0.45, 0, 0.55, 1); }
    @keyframes author-show1 {
        0%, 28% { transform: translateX(0) translateZ(0); opacity: 1; }
        29.5% { transform: translateX(-50%) translateZ(0); opacity: 0.92; }
        31% { transform: translateX(-100%) translateZ(0); opacity: 0.6; }
        32%, 93% { transform: translateX(100%) translateZ(0); opacity: 0; }
        94% { transform: translateX(100%) translateZ(0); opacity: 0.6; }
        95.5% { transform: translateX(50%) translateZ(0); opacity: 0.92; }
        97%, 100% { transform: translateX(0) translateZ(0); opacity: 1; }
    }
    @keyframes author-show2 {
        0%, 27% { transform: translateX(100%) translateZ(0); opacity: 0; }
        28% { transform: translateX(100%) translateZ(0); opacity: 0.6; }
        29.5% { transform: translateX(50%) translateZ(0); opacity: 0.92; }
        31%, 64% { transform: translateX(0) translateZ(0); opacity: 1; }
        65.5% { transform: translateX(-50%) translateZ(0); opacity: 0.92; }
        67% { transform: translateX(-100%) translateZ(0); opacity: 0.6; }
        68%, 100% { transform: translateX(100%) translateZ(0); opacity: 0; }
    }
    @keyframes author-show3 {
        0%, 63% { transform: translateX(100%) translateZ(0); opacity: 0; }
        64% { transform: translateX(100%) translateZ(0); opacity: 0.6; }
        65.5% { transform: translateX(50%) translateZ(0); opacity: 0.92; }
        67%, 94% { transform: translateX(0) translateZ(0); opacity: 1; }
        95.5% { transform: translateX(-50%) translateZ(0); opacity: 0.92; }
        97%, 100% { transform: translateX(-100%) translateZ(0); opacity: 0.6; }
    }
    /* ===== Login panel responsive breakpoints ===== */
    @media (max-width: 992px) {
        .author-slideshow-wrap { min-height: 260px !important; max-height: 420px !important; }
        .login-panel-title { font-size: 1.2rem !important; }
    }
    @media (max-width: 768px) {
        .author-slideshow-wrap { min-height: 200px !important; max-height: 320px !important; aspect-ratio: 16/9 !important; }
        .login-panel-title { font-size: 1.1rem !important; }
        .login-panel-sub { font-size: 0.82rem !important; }
    }
    </style>
    """, unsafe_allow_html=True)

    # Two-column layout: slideshow left + form right
    left_half, right_half = st.columns([1, 1])

    with left_half:
        if slide_data_urls:
            slides_html = "".join(
                f'<img class="slide-img-{i+1}" src="{_html.escape(url)}" alt="Slide {i+1}"/>'
                for i, url in enumerate(slide_data_urls)
            )
            st.markdown(
                f'<div class="author-slideshow-wrap">{slides_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write("")

    with right_half:
        st.markdown('<div style="height: 5rem;"></div>', unsafe_allow_html=True)
        _left, form_col, _right = st.columns([1, 2, 1])
        with form_col:
            # Shared CTA wrapper for CSS targeting
            st.markdown('<div class="login-form-cta-wrap" style="display:none;"></div>',
                        unsafe_allow_html=True)

            # ──────────── SIGN IN MODE ────────────
            if mode == "signin":
                st.markdown(
                    '<p class="login-panel-title">WELCOME TO RotriDASH</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="login-panel-sub">Sign in to continue to your dashboard</p>',
                    unsafe_allow_html=True,
                )

                email = st.text_input(
                    "E-MAIL",
                    key="login_panel_email",
                    placeholder="name@company.com",
                )
                password = st.text_input(
                    "PASSWORD",
                    type="password",
                    key="login_panel_password",
                    placeholder="Enter your password",
                )

                login_clicked = st.button(
                    "Sign In →", type="primary",
                    use_container_width=True, key="login_panel_submit"
                )

                # Subtle links row
                st.markdown('<div class="login-links-divider"></div>', unsafe_allow_html=True)
                if st.button("Forgot password?", key="goto_forgot", type="secondary", use_container_width=True):
                    st.session_state.login_mode = "forgot_password"
                    st.rerun()
                st.markdown(
                    '<p class="login-footer-text">Don\'t have an account?</p>',
                    unsafe_allow_html=True,
                )
                if st.button("Create a free account →", key="goto_signup", type="secondary", use_container_width=True):
                    st.session_state.login_mode = "signup"
                    st.rerun()

                if login_clicked:
                    if not email.strip() or not password.strip():
                        st.warning("Please enter both email and password.")
                    else:
                        with st.spinner("Signing in..."):
                            success, message = login(email.strip(), password.strip())
                        if success:
                            st.session_state.show_login_form = False
                            st.session_state.show_front_page = False
                            st.session_state.login_mode = "signin"
                            try:
                                from usage_tracking import auto_track_login
                                auto_track_login()
                            except Exception:
                                pass
                            save_session_to_browser()
                            status = get_profile_status()
                            if status == "approved":
                                st.session_state.author_details_completed = True
                                st.session_state.show_upload_area = True
                            st.rerun()
                        else:
                            st.error(message)

            # ──────────── SIGN UP MODE ────────────
            elif mode == "signup":
                st.markdown(
                    '<p class="login-panel-title">CREATE ACCOUNT</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="login-panel-sub">Get started with a free account</p>',
                    unsafe_allow_html=True,
                )

                su_email = st.text_input(
                    "E-MAIL",
                    key="signup_panel_email",
                    placeholder="name@company.com",
                )
                su_password = st.text_input(
                    "PASSWORD",
                    type="password",
                    key="signup_panel_password",
                    placeholder="Create a password (min 6 chars)",
                )
                su_confirm = st.text_input(
                    "CONFIRM PASSWORD",
                    type="password",
                    key="signup_panel_confirm",
                    placeholder="Re-enter your password",
                )

                signup_clicked = st.button(
                    "Create Account →", type="primary",
                    use_container_width=True, key="signup_panel_submit"
                )

                st.markdown('<div class="login-links-divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="login-footer-text">Already have an account?</p>',
                    unsafe_allow_html=True,
                )
                if st.button("← Sign in instead", key="goto_signin_from_signup", type="secondary", use_container_width=True):
                    st.session_state.login_mode = "signin"
                    st.rerun()

                if signup_clicked:
                    if not su_email.strip():
                        st.warning("Please enter your email.")
                    elif not su_password or len(su_password) < 6:
                        st.warning("Password must be at least 6 characters.")
                    elif su_password != su_confirm:
                        st.warning("Passwords do not match.")
                    else:
                        with st.spinner("Creating your account..."):
                            success, message = signup(su_email.strip(), su_password)
                        if success:
                            st.markdown(f'''
                            <div style="
                                background: linear-gradient(135deg, #ecfdf5 0%, #f0fdf4 100%);
                                border: 1px solid #86efac;
                                border-left: 4px solid #22c55e;
                                border-radius: 10px;
                                padding: 1.2rem 1.4rem;
                                margin: 0.8rem 0;
                                display: flex;
                                align-items: flex-start;
                                gap: 0.75rem;
                            ">
                                <span style="font-size: 1.6rem; line-height: 1;">🎉</span>
                                <div>
                                    <p style="font-weight: 600; color: #166534; margin: 0 0 0.3rem 0; font-size: 0.95rem;">Account Created Successfully!</p>
                                    <p style="color: #15803d; margin: 0; font-size: 0.85rem; line-height: 1.5;">Check your email for a verification link. Once verified, come back and sign in.</p>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div style="
                                background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
                                border: 1px solid #fca5a5;
                                border-left: 4px solid #ef4444;
                                border-radius: 10px;
                                padding: 1.2rem 1.4rem;
                                margin: 0.8rem 0;
                                display: flex;
                                align-items: flex-start;
                                gap: 0.75rem;
                            ">
                                <span style="font-size: 1.4rem; line-height: 1;">⚠️</span>
                                <div>
                                    <p style="font-weight: 600; color: #991b1b; margin: 0 0 0.2rem 0; font-size: 0.95rem;">Sign Up Failed</p>
                                    <p style="color: #b91c1c; margin: 0; font-size: 0.85rem;">{_html.escape(message)}</p>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)

            # ──────────── FORGOT PASSWORD MODE ────────────
            elif mode == "forgot_password":
                st.markdown(
                    '<p class="login-panel-title">RESET PASSWORD</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<p class="login-panel-sub">Enter your email and we\'ll send a reset link</p>',
                    unsafe_allow_html=True,
                )

                fp_email = st.text_input(
                    "E-MAIL",
                    key="forgot_panel_email",
                    placeholder="name@company.com",
                )

                reset_clicked = st.button(
                    "Send Reset Link →", type="primary",
                    use_container_width=True, key="forgot_panel_submit"
                )

                st.markdown('<div class="login-links-divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    '<p class="login-footer-text">Remembered your password?</p>',
                    unsafe_allow_html=True,
                )
                if st.button("← Back to Sign In", key="goto_signin_from_forgot", type="secondary", use_container_width=True):
                    st.session_state.login_mode = "signin"
                    st.rerun()

                if reset_clicked:
                    if not fp_email.strip():
                        st.warning("Please enter your email address.")
                    else:
                        with st.spinner("Sending reset link..."):
                            success, message = reset_password(fp_email.strip())
                        if success:
                            st.markdown('''
                            <div style="
                                background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
                                border: 1px solid #93c5fd;
                                border-left: 4px solid #3b82f6;
                                border-radius: 10px;
                                padding: 1.2rem 1.4rem;
                                margin: 0.8rem 0;
                                display: flex;
                                align-items: flex-start;
                                gap: 0.75rem;
                            ">
                                <span style="font-size: 1.6rem; line-height: 1;">📧</span>
                                <div>
                                    <p style="font-weight: 600; color: #1e40af; margin: 0 0 0.3rem 0; font-size: 0.95rem;">Reset Link Sent!</p>
                                    <p style="color: #1d4ed8; margin: 0; font-size: 0.85rem; line-height: 1.5;">If an account with that email exists, you\'ll receive a password reset link shortly. Check your inbox and spam folder.</p>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown(f'''
                            <div style="
                                background: linear-gradient(135deg, #fef2f2 0%, #fff1f2 100%);
                                border: 1px solid #fca5a5;
                                border-left: 4px solid #ef4444;
                                border-radius: 10px;
                                padding: 1.2rem 1.4rem;
                                margin: 0.8rem 0;
                                display: flex;
                                align-items: flex-start;
                                gap: 0.75rem;
                            ">
                                <span style="font-size: 1.4rem; line-height: 1;">⚠️</span>
                                <div>
                                    <p style="font-weight: 600; color: #991b1b; margin: 0 0 0.2rem 0; font-size: 0.95rem;">Could Not Send Reset Email</p>
                                    <p style="color: #b91c1c; margin: 0; font-size: 0.85rem;">{_html.escape(message)}</p>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Profile setup / edit form (right-side panel)
# ---------------------------------------------------------------------------
def render_profile_setup(is_edit: bool = False):
    """Render the profile setup or edit form.

    Args:
        is_edit: If True, pre-fill fields and show as "Edit Profile".
    """
    slide_data_urls = _load_slide_images()
    # Determine current role once (used for layout and fields)
    current_role = st.session_state.get("user_role") or "viewer"
    is_admin_like = current_role in ("org_admin", "super_admin")

    st.markdown("""
    <style>
    .profile-setup-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 0.35rem 0;
        text-align: left;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    div[data-testid="column"]:has(.profile-setup-cta-wrap) .stButton > button {
        background: linear-gradient(135deg, #0d3d56 0%, #154360 35%, #2E86C1 100%) !important;
        color: #fff !important;
        border: none !important;
        box-shadow: 0 4px 14px rgba(13, 61, 86, 0.25) !important;
    }
    div[data-testid="column"]:has(.profile-setup-cta-wrap) .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(13, 61, 86, 0.35) !important;
    }
    
    /* Slideshow CSS (Copied from login panel) */
    .author-slideshow-wrap {
        position: relative;
        width: 100%;
        aspect-ratio: 4/3;
        min-height: 320px;
        max-height: 580px;
        margin: 0;
        padding: 0;
        overflow: hidden;
        border-radius: 12px;
        box-sizing: border-box;
        display: block;
    }
    .author-slideshow-wrap img {
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 100%;
        margin: 0; padding: 0; border: none;
        display: block;
        object-fit: cover;
        object-position: center;
        opacity: 0;
        will-change: transform, opacity;
        backface-visibility: hidden;
        transform: translateZ(0);
    }
    .author-slideshow-wrap img.slide-img-1 { animation: author-show1 15s infinite cubic-bezier(0.45, 0, 0.55, 1); }
    .author-slideshow-wrap img.slide-img-2 { animation: author-show2 15s infinite cubic-bezier(0.45, 0, 0.55, 1); }
    .author-slideshow-wrap img.slide-img-3 { animation: author-show3 15s infinite cubic-bezier(0.45, 0, 0.55, 1); }
    @keyframes author-show1 {
        0%, 28% { transform: translateX(0) translateZ(0); opacity: 1; }
        29.5% { transform: translateX(-50%) translateZ(0); opacity: 0.92; }
        31% { transform: translateX(-100%) translateZ(0); opacity: 0.6; }
        32%, 93% { transform: translateX(100%) translateZ(0); opacity: 0; }
        94% { transform: translateX(100%) translateZ(0); opacity: 0.6; }
        95.5% { transform: translateX(50%) translateZ(0); opacity: 0.92; }
        97%, 100% { transform: translateX(0) translateZ(0); opacity: 1; }
    }
    @keyframes author-show2 {
        0%, 27% { transform: translateX(100%) translateZ(0); opacity: 0; }
        28% { transform: translateX(100%) translateZ(0); opacity: 0.6; }
        29.5% { transform: translateX(50%) translateZ(0); opacity: 0.92; }
        31%, 64% { transform: translateX(0) translateZ(0); opacity: 1; }
        65.5% { transform: translateX(-50%) translateZ(0); opacity: 0.92; }
        67% { transform: translateX(-100%) translateZ(0); opacity: 0.6; }
        68%, 100% { transform: translateX(100%) translateZ(0); opacity: 0; }
    }
    @keyframes author-show3 {
        0%, 63% { transform: translateX(100%) translateZ(0); opacity: 0; }
        64% { transform: translateX(100%) translateZ(0); opacity: 0.6; }
        65.5% { transform: translateX(50%) translateZ(0); opacity: 0.92; }
        67%, 94% { transform: translateX(0) translateZ(0); opacity: 1; }
        95.5% { transform: translateX(-50%) translateZ(0); opacity: 0.92; }
        97%, 100% { transform: translateX(-100%) translateZ(0); opacity: 0.6; }
    }
    </style>
    """, unsafe_allow_html=True)

    left_half, right_half = st.columns([1, 1])

    with left_half:
        # For admin/super_admin edit profile view, add a small top spacer
        # so the slideshow aligns more cleanly with the form.
        if is_edit and is_admin_like:
            st.markdown('<div style="height: 3rem;"></div>', unsafe_allow_html=True)

        if slide_data_urls:
            slides_html = "".join(
                f'<img class="slide-img-{i+1}" src="{_html.escape(url)}" alt="Slide {i+1}"/>'
                for i, url in enumerate(slide_data_urls)
            )
            st.markdown(
                f'<div class="author-slideshow-wrap">{slides_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.write("")

    with right_half:
        st.markdown('<div style="height: 6rem;"></div>', unsafe_allow_html=True)
        _left, form_col, _right = st.columns([1, 2, 1])
        with form_col:
            title = "EDIT YOUR PROFILE" if is_edit else "SET UP YOUR PROFILE"
            st.markdown(f'<p class="profile-setup-title">{title}</p>', unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:0.85rem; color:#64748b; margin-bottom:1rem;">'
                'This information will be reviewed by your administrator.</p>',
                unsafe_allow_html=True,
            )

            default_name = st.session_state.get("user_name", "") or ""
            if default_name == "New User":
                default_name = ""
            default_org = st.session_state.get("organization_name", "") or ""

            full_name = st.text_input(
                "END USER NAME",
                value=default_name,
                key="profile_setup_name",
                placeholder="e.g. Kandan V",
            )

            # In edit mode (for all roles), hide the organization selector and
            # keep the existing organization from session. For initial setup,
            # show the organization/company controls.
            if is_edit:
                org_name = default_org
            else:
                # Organization: dropdown of existing + "Other (add my company)"
                org_list = []
                try:
                    if USE_LOCAL_AUTH:
                        from db_queries import fetch_all_organizations
                        org_list = [o["name"] for o in fetch_all_organizations()]
                    else:
                        svc = get_supabase_service()
                        if svc:
                            r = svc.table("organizations").select("id, name").order("name").execute()
                            if r.data:
                                org_list = [x["name"] for x in r.data]
                except Exception:
                    pass
                other_label = "Other (add my company)"
                org_options = [other_label] + org_list
                default_idx = 0
                if default_org and default_org in org_list:
                    default_idx = org_options.index(default_org)
                selected_org_choice = st.selectbox(
                    "ORGANIZATION / COMPANY",
                    options=org_options,
                    index=default_idx,
                    key="profile_setup_org_select",
                    help="Select your company or choose 'Other' to add a new one.",
                )
                org_name = None
                if selected_org_choice == other_label:
                    org_name = st.text_input(
                        "Enter your company name",
                        value=default_org if default_org and default_org not in org_list else "",
                        key="profile_setup_org_other",
                        placeholder="e.g. REUDE Technologies",
                    )
                else:
                    org_name = selected_org_choice

            # Organization logo upload (brand logo used on PDF cover page).
            # This is optional and only shown when editing an existing profile,
            # so first-time setup stays focused on basic details.
            try:
                from org_logo import save_org_logo, get_org_logo_path
            except Exception:
                save_org_logo = None  # type: ignore[assignment]
                get_org_logo_path = None  # type: ignore[assignment]

            _org_id = st.session_state.get("organization_id")
            if is_edit and _org_id and get_org_logo_path is not None and save_org_logo is not None:
                _current_logo_path = get_org_logo_path(_org_id)
                _has_logo = bool(_current_logo_path or st.session_state.get("org_logo_path"))
                _expander_title = "Branding logo already set (click to update)" if _has_logo else "Branding (company logo for reports)"
                with st.expander(_expander_title, expanded=False):
                    if _has_logo:
                        st.info("A company logo is already set and will appear on report cover pages. You can upload a new one below to update it.")
                    if _current_logo_path:
                        try:
                            st.image(
                                _current_logo_path,
                                caption="Current organization logo (used on the report cover page)",
                                use_container_width=False,
                            )
                        except Exception:
                            st.caption("")

                    _logo_label = (
                        "Update company logo (PNG/JPG, used on the report cover page)"
                        if _has_logo
                        else "Upload company logo (PNG/JPG, used on the report cover page)"
                    )
                    _logo_file = st.file_uploader(
                        _logo_label,
                        type=["png", "jpg", "jpeg"],
                        accept_multiple_files=False,
                        key="profile_org_logo_uploader",
                    )
                    if _logo_file is not None:
                        _logo_bytes = _logo_file.getvalue()
                        _saved_path = save_org_logo(_org_id, _logo_bytes)
                        if _saved_path:
                            st.session_state["org_logo_path"] = _saved_path
                            st.success("Organization logo updated. Future reports will use this logo on the cover page.")

            # Role preference
            role_options = ["viewer", "editor", "org_admin"]
            role_labels = {"viewer": "Viewer", "editor": "Editor", "org_admin": "Org Admin"}
            is_super_admin = current_role == "super_admin"
            is_org_admin = current_role == "org_admin"
            if is_edit and is_super_admin:
                # On super admin edit page, hide the role preference control entirely
                # but keep the role fixed internally.
                selected_role = "super_admin"
            elif is_super_admin:
                # For other super_admin contexts, show a disabled role selector.
                st.selectbox(
                    "ROLE PREFERENCE",
                    ["super_admin"],
                    index=0,
                    key="profile_setup_role",
                    disabled=True,
                    help="Your role is managed by an administrator.",
                )
                selected_role = "super_admin"
            elif is_edit and is_org_admin:
                # Org admins can't change their own role via profile edit
                st.selectbox(
                    "ROLE PREFERENCE",
                    ["Org Admin"],
                    index=0,
                    key="profile_setup_role",
                    disabled=True,
                    help="Your role is managed by an administrator.",
                )
                selected_role = "org_admin"
            else:
                default_index = role_options.index(current_role) if current_role in role_options else 0
                selected_role = st.selectbox(
                    "ROLE PREFERENCE",
                    role_options,
                    index=default_index,
                    key="profile_setup_role",
                    format_func=lambda r: role_labels.get(r, r.replace('_', ' ').title()),
                    help="Viewer: Read-only access to org reports. Editor: Upload files and generate reports. Org Admin: Manage org users and view all org reports."
                )

            st.markdown('<div class="profile-setup-cta-wrap" style="display:none;"></div>',
                        unsafe_allow_html=True)
            btn_label = "Update Profile →" if is_edit else "Submit Profile →"
            submit_clicked = st.button(
                btn_label, type="primary",
                use_container_width=True, key="profile_setup_submit"
            )

            if is_edit:
                cancel_clicked = st.button("← Cancel", key="profile_edit_cancel", use_container_width=True)
                if cancel_clicked:
                    st.session_state.show_author_form = False
                    st.session_state.show_profile_editor = False
                    if st.session_state.get("user_role") == "viewer":
                        st.session_state.show_report_history = True
                        st.session_state.show_upload_area = False
                        st.session_state.files_submitted = False
                        st.session_state.show_front_page = False
                    else:
                        st.session_state.show_upload_area = st.session_state.get("prev_page_show_upload_area", True)
                        st.session_state.files_submitted = st.session_state.get("prev_page_files_submitted", False)
                    st.rerun()

    if submit_clicked:
        if is_edit:
            # For edits, keep the existing organization value from session.
            org_value = (default_org or "").strip()
        else:
            org_value = (org_name or "").strip()
        if not full_name.strip() or (not org_value and not is_edit):
            st.warning("Please fill in both **End User Name** and **Organization** (or select a company from the list).")
        else:
            with st.spinner("Submitting profile..."):
                success, msg = submit_profile(full_name.strip(), org_value, selected_role)
            if success:
                if "approval" in msg.lower():
                    st.toast("📨 Notification sent to Super Admin for approval", icon="🚀")
                else:
                    st.toast("✅ Profile updated", icon="✅")
                st.success(msg)
                st.session_state.show_profile_editor = False
                st.session_state.show_author_form = False
                if st.session_state.get("user_role") == "viewer":
                    st.session_state.show_report_history = True
                    st.session_state.show_upload_area = False
                    st.session_state.show_front_page = False
                st.rerun()
            else:
                st.error(msg)


# ---------------------------------------------------------------------------
# Status screens
# ---------------------------------------------------------------------------
def render_pending_approval_screen():
    """Show a 'waiting for approval' message when profile is pending."""
    st.markdown("""
    <div style="
        max-width: 500px; margin: 4rem auto; text-align: center;
        padding: 3rem 2rem;
        background: #f8fafc; border-radius: 16px;
        border: 1px solid #e2e8f0;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⏳</div>
        <h2 style="color: #0f172a; margin: 0 0 0.75rem 0;">Awaiting Approval</h2>
        <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6; margin: 0;">
            Your profile has been submitted and is waiting for<br>
            administrator approval. You'll have full access once approved.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("🚪 Logout", use_container_width=True, key="pending_logout"):
            logout()
            st.rerun()


def render_rejected_screen():
    """Show a 'profile rejected' message with option to re-submit."""
    st.markdown("""
    <div style="
        max-width: 500px; margin: 4rem auto; text-align: center;
        padding: 3rem 2rem;
        background: #fef2f2; border-radius: 16px;
        border: 1px solid #fecaca;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">❌</div>
        <h2 style="color: #991b1b; margin: 0 0 0.75rem 0;">Profile Not Approved</h2>
        <p style="color: #64748b; font-size: 0.95rem; line-height: 1.6; margin: 0;">
            Your profile was not approved. Please update your details<br>
            and re-submit for review.
        </p>
    </div>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1, 1])
    with center:
        if st.button("✏️ Edit & Re-submit", use_container_width=True, key="rejected_edit"):
            if set_own_profile_pending_setup():
                st.session_state.profile_status = "pending_setup"
            st.rerun()
        if st.button("🚪 Logout", use_container_width=True, key="rejected_logout"):
            logout()
            st.rerun()





# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_slide_images() -> list[str]:
    """Load slideshow images from Dashboard_detail_page_image/ or assets/."""
    out = []
    base_dir = os.path.dirname(os.path.abspath(__file__))
    for name in ("Slide_1", "Slide_2", "Slide_3"):
        found = False
        for ext in (".jpeg", ".jpg"):
            path = os.path.join(base_dir, "Dashboard_detail_page_image", name + ext)
            if os.path.isfile(path):
                try:
                    with open(path, "rb") as f:
                        out.append("data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8"))
                except Exception:
                    pass
                found = True
                break
        if not found and os.path.isdir(os.path.join(base_dir, "assets")):
            for f in os.listdir(os.path.join(base_dir, "assets")):
                if name.replace("_", "") in f.replace("_", "") and f.lower().endswith((".jpg", ".jpeg")):
                    path = os.path.join(base_dir, "assets", f)
                    try:
                        with open(path, "rb") as fp:
                            out.append("data:image/jpeg;base64," + base64.b64encode(fp.read()).decode("utf-8"))
                    except Exception:
                        pass
                    break
    return out

