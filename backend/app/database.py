from pathlib import Path
 
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
 
# database.py lives in backend/app/, so parent.parent = backend/
BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "db" / "fantasyedge.db"
 
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI's threadpool
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
 
 
def get_db():
    """FastAPI dependency yielding a SQLAlchemy Session (not a raw
    sqlite3.Connection) — the routers use SQLAlchemy's select()/Session.get(),
    which only a Session understands."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()