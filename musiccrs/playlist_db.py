"""SQLite helpers for the MPD-based database.

This version removes any Spotify-dependent ranking.
"""

from __future__ import annotations

import os
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

DEFAULT_DB_PATH = Path(os.getenv("MUSICCRS_DB", "data/mpd.sqlite"))

# ---------- connection ----------
def ensure_db(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con

# ---------- simple getters ----------
def get_track(artist: str, title: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Tuple]:
    close = False
    if conn is None:
        conn = ensure_db()
        close = True
    row = conn.execute(
        "SELECT track_uri, artist, title, album FROM tracks WHERE LOWER(artist)=? AND LOWER(title)=? LIMIT 1",
        (artist.lower(), title.lower()),
    ).fetchone()
    if close:
        conn.close()
    return tuple(row) if row else None

def get_track_by_uri(uri: str, conn: Optional[sqlite3.Connection] = None) -> Optional[Tuple]:
    close = False
    if conn is None:
        conn = ensure_db()
        close = True
    row = conn.execute(
        "SELECT track_uri, artist, title, album FROM tracks WHERE track_uri=? LIMIT 1",
        (uri,),
    ).fetchone()
    if close:
        conn.close()
    return tuple(row) if row else None

# ---------- searches ----------
def search_by_artist_title(artist: str, title: str, limit: int = 20, conn: Optional[sqlite3.Connection] = None) -> List[Tuple]:
    """If title is empty, return top tracks for artist; otherwise match both."""
    close = False
    if conn is None:
        conn = ensure_db()
        close = True
    if title:
        rows = conn.execute(
            """
            SELECT track_uri, artist, title, album
            FROM tracks
            WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
            LIMIT ?
            """,
            (f"%{artist.lower()}%", f"%{title.lower()}%", limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT track_uri, artist, title, album
            FROM tracks
            WHERE LOWER(artist) LIKE ?
            LIMIT ?
            """,
            (f"%{artist.lower()}%", limit),
        ).fetchall()
    if close:
        conn.close()
    return [tuple(r) for r in rows]

def search_by_artist_title_fuzzy(artist: str, title: str, limit: int = 20, conn: Optional[sqlite3.Connection] = None) -> List[Tuple]:
    """Basic fuzzy by LIKE on both fields."""
    return search_by_artist_title(artist, title, limit=limit, conn=conn)

def search_by_title(title: str, limit: int = 50, conn: Optional[sqlite3.Connection] = None) -> List[Tuple]:
    close = False
    if conn is None:
        conn = ensure_db()
        close = True
    rows = conn.execute(
        """
        SELECT track_uri, artist, title, album, COALESCE(popularity,0) AS popularity
        FROM tracks
        WHERE LOWER(title) LIKE ?
        LIMIT ?
        """,
        (f"%{title.lower()}%", limit),
    ).fetchall()
    if close:
        conn.close()
    return [tuple(r) for r in rows]

def search_by_title_ranked(title: str, limit: int = 20, conn: Optional[sqlite3.Connection] = None) -> List[Tuple[str,str,str,Optional[str]]]:
    """Thin wrapper preserved for compatibility: relies on DB, not Spotify."""
    rows = search_by_title(title, limit=limit, conn=conn)
    return [(r[0], r[1], r[2], r[3]) for r in rows[:limit]]

# ---------- artist & stats helpers ----------
def count_tracks_by_artist(artist: str, conn: Optional[sqlite3.Connection] = None) -> int:
    close = False
    if conn is None:
        conn = ensure_db()
        close = True
    row = conn.execute(
        "SELECT COUNT(*) FROM tracks WHERE LOWER(artist)=?",
        (artist.lower(),),
    ).fetchone()
    if close:
        conn.close()
    return int(row[0] if row else 0)
