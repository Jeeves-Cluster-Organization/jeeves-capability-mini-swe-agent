"""Database migration system for mini-swe-agent v2.0."""

import asyncio
import logging
from pathlib import Path
from typing import List

import asyncpg

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Manage database schema migrations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.migrations_dir = Path(__file__).parent / "migrations"

    async def _get_connection(self) -> asyncpg.Connection:
        """Get database connection."""
        return await asyncpg.connect(self.database_url)

    async def _create_migrations_table(self, conn: asyncpg.Connection):
        """Create migrations tracking table."""
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_migrations (
                migration_id VARCHAR(255) PRIMARY KEY,
                applied_at TIMESTAMP DEFAULT NOW()
            )
        """)

    async def _get_applied_migrations(self, conn: asyncpg.Connection) -> List[str]:
        """Get list of applied migrations."""
        rows = await conn.fetch("SELECT migration_id FROM schema_migrations ORDER BY migration_id")
        return [row["migration_id"] for row in rows]

    async def _get_pending_migrations(self, conn: asyncpg.Connection) -> List[Path]:
        """Get list of pending migrations."""
        applied = await self._get_applied_migrations(conn)
        all_migrations = sorted(self.migrations_dir.glob("*.sql"))

        pending = []
        for migration_file in all_migrations:
            migration_id = migration_file.stem
            if migration_id not in applied:
                pending.append(migration_file)

        return pending

    async def _apply_migration(self, conn: asyncpg.Connection, migration_file: Path):
        """Apply a single migration."""
        migration_id = migration_file.stem
        logger.info(f"Applying migration: {migration_id}")

        # Read SQL file
        sql = migration_file.read_text()

        # Execute in transaction
        async with conn.transaction():
            # Execute migration SQL
            await conn.execute(sql)

            # Record migration
            await conn.execute(
                "INSERT INTO schema_migrations (migration_id) VALUES ($1)",
                migration_id
            )

        logger.info(f"✓ Applied migration: {migration_id}")

    async def migrate(self, dry_run: bool = False):
        """Run pending migrations."""
        conn = await self._get_connection()

        try:
            # Create migrations table
            await self._create_migrations_table(conn)

            # Get pending migrations
            pending = await self._get_pending_migrations(conn)

            if not pending:
                logger.info("No pending migrations")
                return

            logger.info(f"Found {len(pending)} pending migrations")

            if dry_run:
                for migration_file in pending:
                    logger.info(f"  - {migration_file.stem} (dry run)")
                return

            # Apply migrations
            for migration_file in pending:
                await self._apply_migration(conn, migration_file)

            logger.info(f"✓ Applied {len(pending)} migrations successfully")

        finally:
            await conn.close()

    async def status(self):
        """Show migration status."""
        conn = await self._get_connection()

        try:
            await self._create_migrations_table(conn)

            applied = await self._get_applied_migrations(conn)
            all_migrations = sorted(self.migrations_dir.glob("*.sql"))

            logger.info("Migration Status:")
            logger.info(f"  Total migrations: {len(all_migrations)}")
            logger.info(f"  Applied: {len(applied)}")
            logger.info(f"  Pending: {len(all_migrations) - len(applied)}")

            logger.info("\nApplied migrations:")
            for migration_id in applied:
                logger.info(f"  ✓ {migration_id}")

            pending = await self._get_pending_migrations(conn)
            if pending:
                logger.info("\nPending migrations:")
                for migration_file in pending:
                    logger.info(f"  ○ {migration_file.stem}")

        finally:
            await conn.close()


async def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m minisweagent.capability.db.migrator <database_url> <command>")
        print("Commands: migrate, status, dry-run")
        sys.exit(1)

    database_url = sys.argv[1]
    command = sys.argv[2]

    migrator = DatabaseMigrator(database_url)

    if command == "migrate":
        await migrator.migrate()
    elif command == "dry-run":
        await migrator.migrate(dry_run=True)
    elif command == "status":
        await migrator.status()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
