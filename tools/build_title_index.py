#!/usr/bin/env python3
"""
Build a compact title index for MPD playlist names.

Default paths (relative to repo root):
  - source DB: ./data/mpd.sqlite
  - output    : ./data/mpd_titles_fts.sqlite

Usage:
  python tools/build_title_index.py
  python tools/build_title_index.py --src ./data/mpd.sqlite --out ./data/mpd_titles_fts.sqlite --force
"""

from __future__ import annotations
import argparse, os, sqlite3, sys, time
from typing import Optional, Tuple

def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _default_src() -> str:
    return os.path.join(_repo_root(), "data", "mpd.sqlite")

def _default_out() -> str:
    return os.path.join(_repo_root(), "data", "mpd_titles_fts.sqlite")

def detect_playlists_table(conn: sqlite3.Connection) -> Tuple[str, str, Optional[str]]:
    """
    Return (table_name, id_col, ntracks_col_or_None).
    Requires a TEXT 'name' column and an integer id (pid/id/playlist_id or rowid).
    """
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    preferred = ["playlists", "playlist", "mpd_playlists", "pl_playlists"]
    ordered = [t for t in preferred if t in tables] + [t for t in tables if t not in preferred]

    for t in ordered:
        cols = conn.execute(f"PRAGMA table_info({t})").fetchall()
        if not cols:
            continue
        coltypes = {str(c[1]).lower(): str(c[2]).upper() for c in cols}
        if "name" not in coltypes:
            continue

        id_col = None
        for cand in ("pid", "id", "playlist_id"):
            if cand in coltypes and "INT" in coltypes[cand]:
                id_col = cand
                break
        if not id_col:
            try:
                conn.execute(f"SELECT rowid FROM {t} LIMIT 1")
                id_col = "rowid"
            except Exception:
                continue

        ntracks_col = None
        for cand in ("ntracks", "num_tracks", "n_songs", "length", "size"):
            if cand in coltypes and "INT" in coltypes[cand]:
                ntracks_col = cand
                break

        return (t, id_col, ntracks_col)

    sys.exit("[build_title_index] Could not detect a playlists table with (id, name[, ntracks]).")

def fts5_available(conn: sqlite3.Connection) -> bool:
    try:
        rows = conn.execute("PRAGMA compile_options").fetchall()
        return any("FTS5" in (row[0] if row and len(row) else "") for row in rows)
    except Exception:
        try:
            conn.execute("CREATE VIRTUAL TABLE temp.__fts5_test__ USING fts5(x)")
            conn.execute("DROP TABLE temp.__fts5_test__")
            return True
        except Exception:
            return False

def build_index(src_db: str, out_db: str, force: bool) -> None:
    t0 = time.time()
    src_db = os.path.abspath(src_db)
    out_db = os.path.abspath(out_db)

    if not os.path.exists(src_db):
        sys.exit(f"[build_title_index] Source DB not found: {src_db}")

    os.makedirs(os.path.dirname(out_db), exist_ok=True)

    if os.path.exists(out_db):
        if force:
            os.remove(out_db)
        else:
            print(f"[build_title_index] Index exists at {out_db}. Use --force to rebuild.")
            return

    # Inspect source schema
    probe = sqlite3.connect(src_db)
    try:
        table, id_col, ntracks_col = detect_playlists_table(probe)
    finally:
        probe.close()

    # Create output DB and bulk copy via ATTACH
    dst = sqlite3.connect(out_db)
    dst.row_factory = sqlite3.Row
    try:
        dst.execute("PRAGMA journal_mode=OFF")
        dst.execute("PRAGMA synchronous=OFF")
        dst.execute("PRAGMA temp_store=MEMORY")

        dst.execute("CREATE TABLE titles (pid INTEGER PRIMARY KEY, name TEXT NOT NULL, ntracks INTEGER)")
        dst.execute("CREATE TABLE meta   (k TEXT PRIMARY KEY, v TEXT)")

        esc = src_db.replace("'", "''")
        dst.execute(f"ATTACH DATABASE '{esc}' AS src")

        select_cols = f"{id_col} AS pid, name"
        if ntracks_col:
            select_cols += f", {ntracks_col} AS ntracks"
        else:
            select_cols += ", NULL AS ntracks"

        copy_sql = f"""
            INSERT INTO titles(pid, name, ntracks)
            SELECT {select_cols}
            FROM src.{table}
            WHERE name IS NOT NULL AND TRIM(name) != ''
        """
        dst.execute(copy_sql)
        dst.commit()              # commit before detach (Windows locking)
        dst.execute("DETACH DATABASE src")

        if fts5_available(dst):
            dst.execute("""
                CREATE VIRTUAL TABLE titles_fts USING fts5(
                    name, content='titles', content_rowid='pid',
                    tokenize='unicode61 remove_diacritics 2'
                )
            """)
            dst.execute("INSERT INTO titles_fts(rowid, name) SELECT pid, name FROM titles")
            fts_msg = "FTS5 index created"
        else:
            fts_msg = "FTS5 not available — LIKE fallback will be used"

        dst.execute("INSERT INTO meta(k, v) VALUES ('built', datetime('now'))")
        dst.commit()
    finally:
        dst.close()

    sz = os.path.getsize(out_db) / 1_000_000
    dt = time.time() - t0
    print(f"[build_title_index] Done in {dt:.1f}s — wrote {sz:.1f} MB to {out_db} ({fts_msg}).")

def main(argv=None):
    ap = argparse.ArgumentParser(description="Build the MPD playlist-title index (one-time).")
    ap.add_argument("--src", help="Path to main MPD SQLite DB", default=_default_src())
    ap.add_argument("--out", help="Path to output index DB", default=_default_out())
    ap.add_argument("--force", action="store_true", help="Overwrite if exists")
    args = ap.parse_args(argv)

    print(f"[build_title_index] Source: {args.src}")
    print(f"[build_title_index] Output: {args.out}")
    print(f"[build_title_index] Force:  {args.force}")
    build_index(args.src, args.out, force=args.force)

if __name__ == "__main__":
    main()
