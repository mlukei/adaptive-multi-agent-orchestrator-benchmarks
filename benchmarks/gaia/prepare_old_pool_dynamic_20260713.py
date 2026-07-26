#!/usr/bin/env python3
"""Prepare GAIA old-pool adaptive continuation configs.

This is the control condition for the GAIA file-refactor dynamic-pool
experiment: same phase-2 setup as ``dynamic_pool_20260706`` (test split,
memory_mode=train, judge enabled), but with the original rich GAIA agent pool
only. No extra file-refactor agent cards and no disabled File Surfer tools are
applied.

Use ``--copy-memory`` before running to clone the original full-rich training
memories into safe working schemas.
"""

from __future__ import annotations

import argparse
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql
import yaml


ROOT = Path(__file__).resolve().parents[2]
SETUP = "dynamic_pool_old_pool_20260713"
CONFIG_ROOT = ROOT / "configs" / SETUP / "gaia"
BASE_DYNAMIC_ROOT = ROOT / "configs" / "dynamic_pool_20260706" / "gaia"
FOLDS = (1, 2, 3)
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class FoldMemory:
    fold: int
    source_variant: str
    target_memory_schema: str

    @property
    def source_schema(self) -> str:
        return _schema_name(self.source_variant)

    @property
    def target_schema(self) -> str:
        return _schema_name(self.target_memory_schema)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--copy-memory",
        action="store_true",
        help="Clone full-rich training memories into old-pool working schemas.",
    )
    parser.add_argument(
        "--force-copy-memory",
        action="store_true",
        help="Drop existing target schemas before cloning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated/copied without writing anything.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    memories = _fold_memories()
    generated = _generate_configs(memories, dry_run=args.dry_run)

    if args.copy_memory:
        _clone_memories(memories, force=args.force_copy_memory, dry_run=args.dry_run)
    else:
        print("Memory copy not executed. Run with --copy-memory before the old-pool runs.")

    print()
    print("Memory mapping:")
    for memory in memories:
        print(f"  fold {memory.fold}: {memory.source_schema} -> {memory.target_schema}")

    if generated:
        print()
        print(f"Generated {len(generated)} config files under {CONFIG_ROOT.relative_to(ROOT)}:")
        for path in generated:
            print(f"  {path.relative_to(ROOT)}")
    return 0


def _fold_memories() -> tuple[FoldMemory, ...]:
    return tuple(
        FoldMemory(
            fold=fold,
            source_variant=f"full_rich_20260626/gaia_adaptive_training_rich_fold_{fold}",
            target_memory_schema=f"{SETUP}/gaia_rich_fold_{fold}_working",
        )
        for fold in FOLDS
    )


def _generate_configs(
    memories: tuple[FoldMemory, ...],
    *,
    dry_run: bool,
) -> list[Path]:
    generated: list[Path] = []
    for memory in memories:
        base_path = BASE_DYNAMIC_ROOT / f"train_on_test_fold_{memory.fold}.yaml"
        data = _load_yaml(base_path)
        orch = data["orchestrator"]
        orch["variant"] = f"{SETUP}/gaia_old_pool_train_on_test_fold_{memory.fold}"
        orch["memory_schema"] = memory.target_memory_schema
        orch["memory_mode"] = "train"
        orch["enable_playbooks"] = True
        orch["enable_blueprints"] = True
        orch["enable_subagent_memory"] = True
        orch["enable_agent_filtering"] = True
        orch["enable_judge"] = True
        orch["profile_agents"] = True
        orch["agent_cards_path"] = "agents/gaia/rich/cards.json"
        orch["noise_agents_path"] = "agents/gaia/rich/noise.json"
        orch["extra_agent_cards_path"] = None
        orch["profiled_bullets_path"] = None
        orch.pop("disabled_tools", None)

        data["execution"]["mode"] = "default"
        data["execution"]["task_split"]["name"] = "test"

        path = CONFIG_ROOT / f"old_pool_fold_{memory.fold}.yaml"
        if dry_run:
            print(f"Would write {path.relative_to(ROOT)}")
        else:
            _write_yaml(path, data)
        generated.append(path)
    return generated


def _clone_memories(
    memories: tuple[FoldMemory, ...],
    *,
    force: bool,
    dry_run: bool,
) -> None:
    conninfo = _conninfo_from_base_config()
    if dry_run:
        for memory in memories:
            print(f"Would clone {memory.source_schema} -> {memory.target_schema}")
        return

    with psycopg.connect(conninfo) as conn:
        for memory in memories:
            _clone_schema(conn, memory.source_schema, memory.target_schema, force=force)
        conn.commit()


def _clone_schema(
    conn: psycopg.Connection[Any],
    source_schema: str,
    target_schema: str,
    *,
    force: bool,
) -> None:
    _validate_identifier(source_schema)
    _validate_identifier(target_schema)

    if not _schema_exists(conn, source_schema):
        raise SystemExit(f"Source memory schema does not exist: {source_schema}")

    target_exists = _schema_exists(conn, target_schema)
    if target_exists and not force:
        print(f"SKIP existing target schema: {target_schema}")
        return
    if target_exists and force:
        conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(target_schema)))

    conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(target_schema)))
    tables = _base_tables(conn, source_schema)
    for table in tables:
        conn.execute(
            sql.SQL("CREATE TABLE {}.{} (LIKE {}.{} INCLUDING ALL)").format(
                sql.Identifier(target_schema),
                sql.Identifier(table),
                sql.Identifier(source_schema),
                sql.Identifier(table),
            )
        )
        conn.execute(
            sql.SQL("INSERT INTO {}.{} SELECT * FROM {}.{}").format(
                sql.Identifier(target_schema),
                sql.Identifier(table),
                sql.Identifier(source_schema),
                sql.Identifier(table),
            )
        )
    print(f"CLONED {source_schema} -> {target_schema} ({len(tables)} tables)")


def _schema_exists(conn: psycopg.Connection[Any], schema_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = %s",
        (schema_name,),
    ).fetchone()
    return row is not None


def _base_tables(conn: psycopg.Connection[Any], schema_name: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = %s
          AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """,
        (schema_name,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _conninfo_from_base_config() -> str:
    data = _load_yaml(BASE_DYNAMIC_ROOT / "train_on_test_fold_1.yaml")
    return str(data["orchestrator"]["db"]["conninfo"])


def _schema_name(value: str) -> str:
    return value.replace("/", "_")


def _validate_identifier(value: str) -> None:
    if not IDENTIFIER_RE.match(value):
        raise SystemExit(f"Unsafe PostgreSQL identifier: {value!r}")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return deepcopy(yaml.safe_load(handle))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=False)


if __name__ == "__main__":
    raise SystemExit(main())
