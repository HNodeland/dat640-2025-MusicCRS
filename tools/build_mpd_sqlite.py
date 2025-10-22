#!/usr/bin/env python3
"""
Build (or augment) the MPD SQLite database with:
- tracks(track_uri PRIMARY KEY, artist, title, album, popularity INTEGER)
- playlists(pid PRIMARY KEY, name, num_tracks, num_artists)
- playlist_tracks(pid, track_uri)  <-- NEW: for co-occurrence

Usage:
  python tools/build_mpd_sqlite.py --mpd ./data/mpd --db ./data/mpd.sqlite

Idempotent:
- Re-running is safe; inserts are upserts, playlist_tracks uses INSERT OR IGNORE.
- Popularity is counted as “number of appearances across playlists”.

Requires MPD JSON slices (e.g., mpd.slice.0000-0009.json).
"""

from __future__ import annotations
import argparse
import glob
import json
import os
import sqlite3
from collections import Counter
from typing import Iterable


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS tracks (
            track_uri   TEXT PRIMARY KEY,
            artist      TEXT,
            title       TEXT,
            album       TEXT,
            popularity  INTEGER DEFAULT 0
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS playlists (
            pid         INTEGER PRIMARY KEY,
            name        TEXT,
            num_tracks  INTEGER,
            num_artists INTEGER
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS playlist_tracks (
            pid         INTEGER NOT NULL,
            track_uri   TEXT NOT NULL,
            PRIMARY KEY (pid, track_uri)
        )
    """)

    # Helpful indexes
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_artist_title ON tracks(artist COLLATE NOCASE, title COLLATE NOCASE)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_tracks_pop ON tracks(popularity)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pt_pid ON playlist_tracks(pid)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_pt_track ON playlist_tracks(track_uri)")
    con.commit()


def _iter_mpd_files(mpd_dir: str) -> Iterable[str]:
    # Typical MPD filenames: mpd.slice.0000-0009.json
    patterns = [
        os.path.join(mpd_dir, "*.json"),
        os.path.join(mpd_dir, "mpd.slice.*.json"),
    ]
    seen = set()
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            if path not in seen:
                seen.add(path)
                yield path


def _process_slice(con: sqlite3.Connection, path: str) -> None:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    playlists = blob.get("playlists", []) or []
    cur = con.cursor()

    # prepared statements
    upsert_track = """
        INSERT INTO tracks (track_uri, artist, title, album, popularity)
        VALUES (?, ?, ?, ?, 1)
        ON CONFLICT(track_uri) DO UPDATE SET
            artist=excluded.artist,
            title=excluded.title,
            album=excluded.album,
            popularity=COALESCE(tracks.popularity,0) + 1
    """

    insert_playlist = """
        INSERT INTO playlists (pid, name, num_tracks, num_artists)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(pid) DO UPDATE SET
            name=excluded.name,
            num_tracks=excluded.num_tracks,
            num_artists=excluded.num_artists
    """

    insert_pt = "INSERT OR IGNORE INTO playlist_tracks (pid, track_uri) VALUES (?, ?)"

    for pl in playlists:
        pid = pl.get("pid")
        name = pl.get("name") or ""
        tracks = pl.get("tracks") or []

        # Insert/update playlist row
        artist_set = set()
        for t in tracks:
            artist_name = (t.get("artist_name") or "").strip()
            if artist_name:
                artist_set.add(artist_name.lower())
        cur.execute(insert_playlist, (pid, name, len(tracks), len(artist_set)))

        # Upsert each track and wire playlist_tracks
        for t in tracks:
            uri = t.get("track_uri") or ""
            artist = t.get("artist_name") or ""
            title = t.get("track_name") or ""
            album = t.get("album_name") or ""

            if not uri:
                continue  # skip malformed rows

            cur.execute(upsert_track, (uri, artist, title, album))
            cur.execute(insert_pt, (pid, uri))

    con.commit()


def build(mpd_dir: str, db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        _ensure_schema(con)
        files = list(_iter_mpd_files(mpd_dir))
        if not files:
            raise SystemExit(f"No MPD JSON files found in {mpd_dir}")

        for i, path in enumerate(files, 1):
            _process_slice(con, path)
            if i % 10 == 0:
                print(f"[build_mpd_sqlite] processed {i}/{len(files)} slices")

        # Vacuum/optimize lightly
        con.execute("ANALYZE")
        con.commit()
        print("[build_mpd_sqlite] done.")
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description="Build/augment MPD SQLite DB with tracks, playlists, playlist_tracks.")
    ap.add_argument("--mpd", required=True, help="Directory with mpd.slice.*.json files")
    ap.add_argument("--db", required=True, help="Output SQLite path (existing or new)")
    args = ap.parse_args()
    build(args.mpd, args.db)


if __name__ == "__main__":
    main()
