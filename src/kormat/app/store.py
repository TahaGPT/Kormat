"""SQLite history of screening reports (app_data/kormat.db)."""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from kormat.common.paths import ROOT

DEFAULT_DB = ROOT / "app_data" / "kormat.db"


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS reports (
                id TEXT PRIMARY KEY, created TEXT, outcome TEXT, mode TEXT, n_criteria INTEGER,
                note_preview TEXT, payload TEXT)""")

    def _conn(self):
        return sqlite3.connect(self.path)

    def save(self, report: dict) -> str:
        rid = uuid.uuid4().hex[:12]
        report["id"] = rid
        with self._lock, self._conn() as c:
            c.execute("INSERT INTO reports VALUES (?,?,?,?,?,?,?)", (
                rid, datetime.now(timezone.utc).isoformat(timespec="seconds"), report["aggregate"]["outcome"],
                report["provenance"].get("decoding_mode"), len(report["criteria"]),
                report["note_text"][:120], json.dumps(report)))
        return rid

    def get(self, rid: str) -> dict | None:
        with self._lock, self._conn() as c:
            row = c.execute("SELECT payload FROM reports WHERE id=?", (rid,)).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT id,created,outcome,mode,n_criteria,note_preview FROM reports "
                             "ORDER BY created DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
        return [dict(zip(("id", "created", "outcome", "mode", "n_criteria", "note_preview"), r)) for r in rows]

    def delete(self, rid: str) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM reports WHERE id=?", (rid,)).rowcount > 0
