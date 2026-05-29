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

