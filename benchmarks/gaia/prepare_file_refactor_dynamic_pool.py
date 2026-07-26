"""Prepare GAIA file-refactor dynamic-pool memory schemas.

Copies the existing rich GAIA training memory schemas into working schemas so
the file-refactor experiment can continue training on the test split without
mutating the original training memories.
"""

from __future__ import annotations

import argparse
import os

from psycopg import sql
from psycopg_pool import ConnectionPool

from orchestrator.memory.setup.schema_manager import MEMORY_TABLES, ensure_schema

DEFAULT_CONNINFO = "postgresql://orchestrator:orchestrator@localhost:5433/orchestrator"
SOURCE_TEMPLATE = "full_rich_20260626_gaia_adaptive_training_rich_fold_{fold}"
TARGET_TEMPLATE = "dynamic_pool_20260706_gaia_file_refactor_rich_fold_{fold}_working"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone rich GAIA training memory schemas for file-refactor runs.",
    )
    parser.add_argument(
        "--conninfo",
        default=os.getenv("DATABASE_URL", DEFAULT_CONNINFO),
        help="Postgres connection string.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Fold numbers to copy.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop and recreate existing target schemas before copying.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for fold in args.folds:
        source = SOURCE_TEMPLATE.format(fold=fold)
        target = TARGET_TEMPLATE.format(fold=fold)
        clone_schema(args.conninfo, source, target, replace=args.replace)
        print(f"Copied {source} -> {target}")


def clone_schema(conninfo: str, source: str, target: str, *, replace: bool) -> None:
    with ConnectionPool(conninfo, min_size=1, max_size=1, open=True) as pool:
        with pool.connection() as conn:
            ensure_source_exists(conn, source)
            prepare_target(conn, target, replace=replace)

    ensure_schema(conninfo, target)

    with ConnectionPool(conninfo, min_size=1, max_size=1, open=True) as pool:
        with pool.connection() as conn, conn.cursor() as cur:
            for table in MEMORY_TABLES:
                cur.execute(
                    sql.SQL("INSERT INTO {}.{} SELECT * FROM {}.{}").format(
                        sql.Identifier(target),
                        sql.Identifier(table),
                        sql.Identifier(source),
                        sql.Identifier(table),
                    ),
                )
            conn.commit()


def ensure_source_exists(conn, source: str) -> None:
    if schema_exists(conn, source):
        return
    raise RuntimeError(
        f"Source schema '{source}' does not exist. "
        "Run the original rich GAIA training fold first or check the schema name.",
    )


def prepare_target(conn, target: str, *, replace: bool) -> None:
    if not schema_exists(conn, target):
        return
    if not replace:
        raise RuntimeError(
            f"Target schema '{target}' already exists. "
            "Use --replace only if you intentionally want to rebuild it.",
        )
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(target)),
        )
    conn.commit()


def schema_exists(conn, schema_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = %s
            )
            """,
            (schema_name,),
        )
        return bool(cur.fetchone()[0])


if __name__ == "__main__":
    main()
