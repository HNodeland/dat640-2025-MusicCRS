# musiccrs/title_index.py
# Lightweight title index lookup for MPD playlist names.
# Uses ./data/mpd_titles_fts.sqlite (no env vars), with BM25 if FTS5 exists,
# otherwise a fast LIKE fallback. Also exposes ensure_index() for safety.

from __future__ import annotations
import os, re, sqlite3, contextlib
from typing import List, Dict, Any, Optional, Tuple

# ----- paths (relative to repo root) -----
def _repo_root() -> str:
    # this file is in musiccrs/, repo root is its parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def _data_dir() -> str:
    return os.path.join(_repo_root(), "data")

def _index_path() -> str:
    # sidecar titles index built by tools/build_title_index.py
    return os.path.join(_data_dir(), "mpd_titles_fts.sqlite")

def _main_db_path() -> str:
    # the big MPD DB used for fetching tracks by playlist id
    return os.path.join(_data_dir(), "mpd.sqlite")

# ----- sqlite helpers -----
def _open_db(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

def _open_index_db() -> sqlite3.Connection:
    return _open_db(_index_path())

def _fts5_available(conn: sqlite3.Connection) -> bool:
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

# ----- optional build (only used if you call ensure_index) -----
def _index_exists(conn: Optional[sqlite3.Connection] = None) -> bool:
    close = False
    if conn is None:
        conn = _open_index_db()
        close = True
    try:
        r = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='titles'").fetchone()
        return r is not None
    finally:
        if close:
            conn.close()

def ensure_index() -> None:
    """
    Only verifies the index exists at ./data/mpd_titles_fts.sqlite.
    (If you need to build it, run tools/build_title_index.py.)
    """
    p = _index_path()
    if not os.path.exists(p):
        raise RuntimeError(f"title_index: missing index at {p}. Run tools/build_title_index.py first.")
    conn = _open_index_db()
    try:
        if not _index_exists(conn):
            raise RuntimeError(f"title_index: {p} exists but is missing 'titles' table. Rebuild the index.")
    finally:
        conn.close()

# ----- search API -----
def _fts_search(conn: sqlite3.Connection, query: str, limit: int) -> List[Dict[str, Any]]:
    """
    FTS5 search (BM25 when available). No aliasing to avoid ambiguity.
    """
    sql_bm25 = """
        SELECT
            titles_fts.rowid  AS pid,
            titles.name       AS name,
            titles.ntracks    AS ntracks
        FROM titles_fts
        JOIN titles
          ON titles.pid = titles_fts.rowid
        WHERE titles_fts MATCH ?
        ORDER BY bm25(titles_fts) ASC
        LIMIT ?
    """
    sql_fallback = """
        SELECT
            titles_fts.rowid  AS pid,
            titles.name       AS name,
            titles.ntracks    AS ntracks
        FROM titles_fts
        JOIN titles
          ON titles.pid = titles_fts.rowid
        WHERE titles_fts MATCH ?
        LIMIT ?
    """
    try:
        cur = conn.execute(sql_bm25, (query.strip(), int(limit)))
    except sqlite3.OperationalError:
        cur = conn.execute(sql_fallback, (query.strip(), int(limit)))

    out: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        out.append({
            "id": int(r["pid"]),
            "name": str(r["name"]),
            "ntracks": int(r["ntracks"]) if r["ntracks"] is not None else None
        })
    return out

def _like_search(conn: sqlite3.Connection, query: str, limit: int) -> List[Dict[str, Any]]:
    """
    Fallback when FTS5 or titles_fts isn't available. All tokens must appear.
    Simple ranking by shorter title first.
    """
    toks = [t for t in re.findall(r"[A-Za-z0-9']+", query or "") if t]
    if not toks:
        return []
    where = " AND ".join(["name LIKE ?"] * len(toks))
    params = [f"%{t}%" for t in toks] + [int(limit)]
    sql = f"""
        SELECT pid AS id, name, ntracks
        FROM titles
        WHERE {where}
        ORDER BY LENGTH(name) ASC
        LIMIT ?
    """
    cur = conn.execute(sql, params)
    out: List[Dict[str, Any]] = []
    for r in cur.fetchall():
        out.append({
            "id": int(r["id"]),
            "name": str(r["name"]),
            "ntracks": int(r["ntracks"]) if r["ntracks"] is not None else None
        })
    return out

def search_titles(query: str, limit: int = 12) -> List[Dict[str, Any]]:
    """
    Returns [{'id': pid, 'name': name, 'ntracks': int|None}, ...] from ./data/mpd_titles_fts.sqlite.
    """
    if not query or not query.strip():
        return []
    conn = _open_index_db()
    try:
        # Detect whether the FTS table is present
        try:
            conn.execute("SELECT 1 FROM titles_fts LIMIT 1")
            has_fts = True
        except Exception:
            has_fts = False

        if has_fts:
            return _fts_search(conn, query, limit)
        else:
            return _like_search(conn, query, limit)
    finally:
        conn.close()

# Expose the main DB path for callers that need it (e.g., fetching tracks by pid)
def discover_main_db_path() -> str:
    return _main_db_path()
