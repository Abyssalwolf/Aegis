from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    pool_pre_ping=True,   # test connections before use — discards stale ones from Neon.tech
    pool_recycle=300,     # recycle connections every 5 min (Neon idle timeout is ~5 min)
    pool_size=5,
    max_overflow=10,
)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)
