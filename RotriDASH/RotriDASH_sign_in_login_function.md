# RotriDASH — sign-in (`login`) and database models (`models.py`)

**`login` source file:** `multi_file/auth.py`  
**Function:** `login(email: str, password: str) -> tuple[bool, str]`

**`models.py` source file:** `multi_file/models.py` — SQLAlchemy engine, session factory, and ORM tables used by local Postgres auth (`Profile`, `Organization`, etc.).

**Backend selection:** `USE_LOCAL_AUTH` is `True` when environment variable `AUTH_BACKEND` is one of `local_pg`, `local`, or `postgres` (see top of `auth.py`). Otherwise the Supabase path below runs.

---

## Python code (`login`)

```python
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
```

---

## Python code (`models.py` — full file)

Local auth uses `SessionLocal` from this module; the first DB operation in `login()` opens a connection to `DATABASE_URL`.

```python
from __future__ import annotations

import os
import uuid
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean,
    BigInteger,
    Column,
    DateTime,
    Integer,
    String,
    Text,
    ForeignKey,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, relationship, scoped_session, sessionmaker


try:
    load_dotenv()
except OSError:
    # Env may already be loaded; ignore filesystem errors.
    pass

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Configure it in your .env (e.g. postgresql://...)"
    )


engine = create_engine(
    DATABASE_URL,
    future=True,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=1800,  # recycle connections every 30 minutes
    pool_timeout=30,    # wait up to 30s for a connection from the pool
    connect_args={
        "options": "-c statement_timeout=30000",  # 30s per-statement timeout
    },
)
SessionLocal = scoped_session(
    sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
)

Base = declarative_base()


class Organization(Base):
    __tablename__ = "organizations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(Text, nullable=False, unique=True)
    max_users = Column(Integer, nullable=False, default=50)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    profiles = relationship("Profile", back_populates="organization")


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, nullable=False, unique=True)
    # Local auth credentials (for non-Supabase backend)
    password_hash = Column(Text, nullable=False)
    email_verified = Column(Boolean, nullable=False, default=True)
    full_name = Column(Text, nullable=False, default="")
    role = Column(Text, nullable=False, default="viewer")
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_active = Column(Boolean, nullable=False, default=True)
    profile_status = Column(Text, nullable=False, default="pending_setup")
    daily_report_quota = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    organization = relationship("Organization", back_populates="profiles")

    usage_events = relationship("UsageEvent", back_populates="user")
    files = relationship("FileMetadata", back_populates="user")
    reports = relationship("ReportMetadata", back_populates="user")


class UsageEvent(Base):
    __tablename__ = "usage_events"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )
    event_type = Column(Text, nullable=False)
    # Column name remains "metadata" in the database; attribute name avoids
    # the reserved "metadata" identifier in SQLAlchemy's Declarative API.
    event_metadata = Column("metadata", JSONB, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("Profile", back_populates="usage_events")


class FileMetadata(Base):
    __tablename__ = "file_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )
    original_filename = Column(Text, nullable=False)
    storage_path = Column(Text, nullable=False, unique=True)
    file_size = Column(BigInteger, nullable=False)
    file_type = Column(Text, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    user = relationship("Profile", back_populates="files")


class ReportMetadata(Base):
    __tablename__ = "report_metadata"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="SET NULL"),
        nullable=True,
    )
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="SET NULL"),
        nullable=True,
    )
    report_name = Column(Text, nullable=False)
    pdf_storage_path = Column(Text, nullable=False, unique=True)
    csv_storage_path = Column(Text, nullable=True)
    source_file_id = Column(
        UUID(as_uuid=True),
        ForeignKey("file_metadata.id", ondelete="SET NULL"),
        nullable=True,
    )
    generated_at = Column(DateTime(timezone=True), nullable=False)

    user = relationship("Profile", back_populates="reports")
    source_file = relationship("FileMetadata")


class ReportTemplate(Base):
    __tablename__ = "report_templates"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    organization_id = Column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
    )
    name = Column(Text, nullable=False)
    description = Column(Text, nullable=False, default="")
    plot_data_source = Column(Text, nullable=False, default="Sorted performance table")
    throttle_aggregation = Column(JSONB, nullable=False, default=dict)
    saved_graphs = Column(JSONB, nullable=False, default=list)
    created_by = Column(Text, nullable=False, default="—")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


def get_db() -> Generator:
    """Yield a SQLAlchemy session. Use as: `with next(get_db()) as db:` in simple scripts."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables in the target database if they do not exist."""
    Base.metadata.create_all(bind=engine)
```

---

## Related helpers (not the full sign-in UI)

| Name | Role |
|------|------|
| `_get_db()` | Returns `SessionLocal()` — SQLAlchemy session; connects using `DATABASE_URL` from `models.py`. |
| `USE_LOCAL_AUTH` | Set from `AUTH_BACKEND` env in `auth.py`. |
| `render_login_panel()` | Same file; builds the form and calls `login(email, password)` when the user clicks Sign In. |

Database engine and table definitions: inlined above in **`models.py`** (`DATABASE_URL`, `create_engine`, `SessionLocal`, ORM models).
