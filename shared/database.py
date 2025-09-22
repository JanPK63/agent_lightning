"""
Database connection and session management for Agent Lightning
Supports multiple database backends: SQLite and PostgreSQL
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool, QueuePool
from .models import Base

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./agentlightning.db"  # Use SQLite for development
)

# Read/Write database splitting configuration
READ_DATABASE_URL = os.getenv("READ_DATABASE_URL", DATABASE_URL)  # Read replica URL
WRITE_DATABASE_URL = os.getenv("WRITE_DATABASE_URL", DATABASE_URL)  # Write primary URL
ENABLE_DB_SPLITTING = os.getenv("ENABLE_DB_SPLITTING", "false").lower() == "true"

# Connection pool configuration
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))  # Number of connections to keep in pool
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))  # Max additional connections beyond pool_size
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))  # Seconds to wait for connection
POOL_RECYCLE = int(os.getenv("DB_POOL_RECYCLE", "3600"))  # Seconds after which to recycle connections
POOL_PRE_PING = os.getenv("DB_POOL_PRE_PING", "true").lower() == "true"  # Test connections before use


def get_database_config():
    """Get database configuration based on URL scheme with configurable pooling"""
    # Read DATABASE_URL dynamically to support runtime changes
    current_url = os.getenv(
        "DATABASE_URL",
        "sqlite:///./agentlightning.db"  # Use SQLite for development
    )
    return get_database_config_for_url(current_url)


def get_database_config_for_url(url):
    """Get database configuration for a specific URL"""
    if url.startswith("sqlite"):
        return {
            "poolclass": StaticPool,
            "pool_pre_ping": POOL_PRE_PING,
            "connect_args": {"check_same_thread": False},  # SQLite specific
        }
    elif url.startswith("postgresql"):
        return {
            "poolclass": QueuePool,
            "pool_pre_ping": POOL_PRE_PING,
            "pool_size": POOL_SIZE,
            "max_overflow": MAX_OVERFLOW,
            "pool_timeout": POOL_TIMEOUT,
            "pool_recycle": POOL_RECYCLE,
        }
    elif url.startswith(("mongodb", "mongodb+srv")):
        # MongoDB is handled separately by MongoDBManager
        # Return None to indicate non-SQL database
        return None
    else:
        # Default configuration for other SQL databases
        return {
            "poolclass": QueuePool,
            "pool_pre_ping": POOL_PRE_PING,
            "pool_size": POOL_SIZE,
            "max_overflow": MAX_OVERFLOW,
            "pool_timeout": POOL_TIMEOUT,
            "pool_recycle": POOL_RECYCLE,
        }


# Get database-specific configuration
db_config = get_database_config()

# Create engines with appropriate configuration (only for SQL databases)
if db_config is not None:
    if ENABLE_DB_SPLITTING:
        # Read/write database splitting enabled
        read_config = get_database_config_for_url(READ_DATABASE_URL)
        write_config = get_database_config_for_url(WRITE_DATABASE_URL)

        if read_config is not None:
            read_engine = create_engine(
                READ_DATABASE_URL,
                **read_config,
                echo=False
            )
        else:
            read_engine = None

        if write_config is not None:
            write_engine = create_engine(
                WRITE_DATABASE_URL,
                **write_config,
                echo=False
            )
        else:
            write_engine = None

        # For backward compatibility, set engine to write_engine
        engine = write_engine
    else:
        # Single database configuration (existing behavior)
        engine = create_engine(
            DATABASE_URL,
            **db_config,
            echo=False
        )
        read_engine = engine
        write_engine = engine
else:
    # MongoDB or other non-SQL database
    engine = None
    read_engine = None
    write_engine = None


# Create session factories (only for SQL databases)
if ENABLE_DB_SPLITTING:
    if write_engine is not None:
        WriteSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=write_engine)
    else:
        WriteSessionLocal = None

    if read_engine is not None:
        ReadSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=read_engine)
    else:
        ReadSessionLocal = None

    # For backward compatibility
    SessionLocal = WriteSessionLocal
else:
    # Single database mode - all sessions use the same engine
    if engine is not None:
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        WriteSessionLocal = SessionLocal
        ReadSessionLocal = SessionLocal
    else:
        SessionLocal = None
        WriteSessionLocal = None
        ReadSessionLocal = None


def get_db_session():
    """Get database session (defaults to write session for backward compatibility)"""
    return get_write_session()


def get_read_session():
    """Get read-only database session"""
    if ReadSessionLocal is None:
        raise RuntimeError(
            "Read database session not available. This may be because "
            "MongoDB is configured or read/write splitting is not enabled."
        )
    return ReadSessionLocal()


def get_write_session():
    """Get write database session"""
    if WriteSessionLocal is None:
        raise RuntimeError(
            "Write database session not available. This may be because "
            "MongoDB is configured instead of SQL database."
        )
    return WriteSessionLocal()


def init_database():
    """Initialize database tables"""
    if engine is None:
        print("ℹ️  MongoDB configured - skipping SQL table initialization")
        return True

    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False


def test_connection():
    """Test database connection"""
    if engine is None:
        # For MongoDB, test using MongoDB manager
        try:
            from .mongodb import test_mongodb_connection
            return test_mongodb_connection()
        except ImportError:
            print("❌ MongoDB dependencies not available")
            return False

    try:
        from sqlalchemy import text
        db = get_db_session()
        db.execute(text("SELECT 1"))
        db.close()
        print("✅ Database connection successful")
        return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


class DatabaseManager:
    """Database manager providing session management and health checks"""

    def get_db(self):
        """Get database session context manager (backward compatibility)"""
        return get_db_session()

    def get_read_db(self):
        """Get read database session context manager"""
        return get_read_session()

    def get_write_db(self):
        """Get write database session context manager"""
        return get_write_session()

    def health_check(self) -> bool:
        """Check database health"""
        return test_connection()

    def is_mongodb_configured(self) -> bool:
        """Check if MongoDB is configured"""
        current_url = os.getenv(
            "DATABASE_URL",
            "sqlite:///./agentlightning.db"
        )
        return current_url.startswith(("mongodb", "mongodb+srv"))

    def is_db_splitting_enabled(self) -> bool:
        """Check if read/write database splitting is enabled"""
        return ENABLE_DB_SPLITTING

    def get_pool_stats(self, db_type: str = None) -> dict:
        """Get connection pool statistics"""
        # For backward compatibility, if no db_type specified, use legacy logic
        if db_type is None:
            if engine is None:
                return {"type": "mongodb", "message": "MongoDB does not use SQLAlchemy pooling"}

            try:
                pool = engine.pool
                stats = {
                    "pool_size": getattr(pool, 'size', 0),
                    "checkedin": getattr(pool, 'checkedin', 0),
                    "checkedout": getattr(pool, 'checkedout', 0),
                    "invalid": getattr(pool, 'invalid', 0),
                    "overflow": getattr(pool, 'overflow', 0),
                    "timeout": getattr(pool, 'timeout', 0),
                    "pool_recycle": getattr(pool, 'recycle', 0),
                    "connections_in_pool": getattr(pool, 'size', 0) - getattr(pool, 'checkedout', 0),
                    "connections_checked_out": getattr(pool, 'checkedout', 0),
                    "connections_overflow": getattr(pool, 'overflow', 0),
                    "connections_invalid": getattr(pool, 'invalid', 0),
                }
                return stats
            except Exception as e:
                return {"error": f"Failed to get pool stats: {str(e)}"}

        # New logic for read/write splitting
        if db_type == "read" and ENABLE_DB_SPLITTING:
            target_engine = read_engine
            db_name = "read"
            db_url = READ_DATABASE_URL
        else:
            target_engine = write_engine
            db_name = "write"
            db_url = WRITE_DATABASE_URL

        if target_engine is None:
            # Check if this specific database URL is MongoDB
            if db_url.startswith(("mongodb", "mongodb+srv")):
                return {"type": "mongodb", "message": "MongoDB does not use SQLAlchemy pooling"}
            else:
                return {"error": f"No {db_name} database engine available"}

        try:
            pool = target_engine.pool
            stats = {
                "database": db_name,
                "pool_size": getattr(pool, 'size', 0),
                "checkedin": getattr(pool, 'checkedin', 0),
                "checkedout": getattr(pool, 'checkedout', 0),
                "invalid": getattr(pool, 'invalid', 0),
                "overflow": getattr(pool, 'overflow', 0),
                "timeout": getattr(pool, 'timeout', 0),
                "pool_recycle": getattr(pool, 'recycle', 0),
                "connections_in_pool": getattr(pool, 'size', 0) - getattr(pool, 'checkedout', 0),
                "connections_checked_out": getattr(pool, 'checkedout', 0),
                "connections_overflow": getattr(pool, 'overflow', 0),
                "connections_invalid": getattr(pool, 'invalid', 0),
            }
            return stats
        except Exception as e:
            return {"error": f"Failed to get {db_name} pool stats: {str(e)}"}

    def get_connection_info(self) -> dict:
        """Get connection configuration information"""
        current_url = os.getenv("DATABASE_URL", "sqlite:///./agentlightning.db")

        if current_url.startswith(("mongodb", "mongodb+srv")):
            return {
                "type": "mongodb",
                "url": current_url.replace(current_url.split('@')[-1], "***") if '@' in current_url else current_url,
                "pooling": "MongoDB native connection pooling"
            }

        info = {
            "type": "sql",
            "db_splitting_enabled": ENABLE_DB_SPLITTING,
            "url": current_url.replace(current_url.split('@')[-1], "***") if '@' in current_url else current_url,
        }

        if ENABLE_DB_SPLITTING:
            # Show both read and write database info
            read_config = get_database_config_for_url(READ_DATABASE_URL)
            write_config = get_database_config_for_url(WRITE_DATABASE_URL)

            info["read_database"] = {
                "url": READ_DATABASE_URL.replace(READ_DATABASE_URL.split('@')[-1], "***") if '@' in READ_DATABASE_URL else READ_DATABASE_URL,
                "pool_class": read_config.get('poolclass', '').__name__ if read_config and hasattr(read_config.get('poolclass', ''), '__name__') else str(read_config.get('poolclass', '')) if read_config else 'N/A',
                "pool_size": read_config.get('pool_size', 'N/A') if read_config else 'N/A',
                "max_overflow": read_config.get('max_overflow', 'N/A') if read_config else 'N/A',
            }

            info["write_database"] = {
                "url": WRITE_DATABASE_URL.replace(WRITE_DATABASE_URL.split('@')[-1], "***") if '@' in WRITE_DATABASE_URL else WRITE_DATABASE_URL,
                "pool_class": write_config.get('poolclass', '').__name__ if write_config and hasattr(write_config.get('poolclass', ''), '__name__') else str(write_config.get('poolclass', '')) if write_config else 'N/A',
                "pool_size": write_config.get('pool_size', 'N/A') if write_config else 'N/A',
                "max_overflow": write_config.get('max_overflow', 'N/A') if write_config else 'N/A',
            }
        else:
            # Single database configuration
            config = get_database_config()
            if config:
                info.update({
                    "pool_class": config.get('poolclass', '').__name__ if hasattr(config.get('poolclass', ''), '__name__') else str(config.get('poolclass', '')),
                    "pool_size": config.get('pool_size', 'N/A'),
                    "max_overflow": config.get('max_overflow', 'N/A'),
                    "pool_timeout": config.get('pool_timeout', 'N/A'),
                    "pool_recycle": config.get('pool_recycle', 'N/A'),
                    "pool_pre_ping": config.get('pool_pre_ping', 'N/A'),
                })

        return info


# Global database manager instance
db_manager = DatabaseManager()