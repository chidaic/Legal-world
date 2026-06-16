from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DATABASE_POOL_SIZE = 5
DEFAULT_DATABASE_MAX_OVERFLOW = 10
DEFAULT_DATABASE_POOL_RECYCLE_SECONDS = 1800


class Base(DeclarativeBase):
    """Shared declarative base for control-plane models."""

    pass


class DatabaseConfigError(RuntimeError):
    """Raised when database configuration is missing or invalid."""


def get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise DatabaseConfigError("DATABASE_URL environment variable is required")
    return database_url


def create_database_engine(database_url: str | None = None) -> Engine:
    url = database_url or get_database_url()
    return create_engine(
        url,
        future=True,
        pool_pre_ping=True,
        pool_size=_read_positive_int_env("DATABASE_POOL_SIZE", DEFAULT_DATABASE_POOL_SIZE),
        max_overflow=_read_non_negative_int_env("DATABASE_MAX_OVERFLOW", DEFAULT_DATABASE_MAX_OVERFLOW),
        pool_recycle=_read_positive_int_env(
            "DATABASE_POOL_RECYCLE_SECONDS",
            DEFAULT_DATABASE_POOL_RECYCLE_SECONDS,
        ),
    )


def _read_positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value > 0 else default


def _read_non_negative_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip())
    except ValueError:
        return default
    return value if value >= 0 else default


def create_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    runtime_engine = engine or create_database_engine()
    return sessionmaker(bind=runtime_engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def get_db_session(
    session_factory: sessionmaker[Session] | None = None,
) -> Iterator[Session]:
    factory = session_factory or create_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
