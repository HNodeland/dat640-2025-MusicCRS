# musiccrs/recommender.py
from __future__ import annotations
import os
import sqlite3
from typing import Iterable, List, Dict, Any, Optional
from dataclasses import dataclass
from time import time

DB_PATH = os.getenv("MUSICCRS_DB", "data/mpd.sqlite")

def _connect_readonly(db_path: str) -> sqlite3.Connection:
    uri = f"file:{os.path.abspath(db_path)}?mode=ro&immutable=1"
    con = sqlite3.connect(uri, uri=True)
    con.row_factory = sqlite3.Row
    return con

def _apply_read_optimized_pragmas(con: sqlite3.Connection) -> None:
    cur = con.cursor()
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("PRAGMA mmap_size=268435456")   # 256 MB
    cur.execute("PRAGMA cache_size=-200000")    # ~200 MB cache
    cur.execute("PRAGMA read_uncommitted=ON")

def _best_single_col_index(con: sqlite3.Connection, table: str, col: str) -> Optional[str]:
    try:
        for _, idx_name, *_ in con.execute(f"PRAGMA index_list({table})"):
            cols = [r[2] for r in con.execute(f"PRAGMA index_info({idx_name})")]
            if len(cols) == 1 and cols[0].lower() == col.lower():
                return idx_name
    except Exception:
        pass
    return None

@dataclass(frozen=True)
class RecCandidate:
    track_uri: str
    artist: str
    title: str
    album: Optional[str]
    popularity: Optional[int]
    w_co: int                 # weighted co-occurrence (sum of hits)
    co_count: int             # # of contributing playlists
    seed_pid_count: int       # # of matched playlists considered
    seed_weight_total: int    # sum(hits) over matched playlists
    track_pid_count: int
    score: float              # normalized weighted score

def _normalize_seeds(seed_uris: Iterable[str], max_seeds: int) -> List[str]:
    out, seen = [], set()
    for u in seed_uris or []:
        u = (u or "").strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= max_seeds:
            break
    return out

def recommend_by_cooccurrence(
    seed_uris: Iterable[str],
    *,
    limit: int = 5,
    # ===== aggressive speed knobs =====
    max_seeds: int = 100,                  # consider at most this many seeds from the user's playlist
    rare_seed_sample: int = 48,            # use the K rarest seeds to find matching playlists
    min_seed_hits_per_playlist: int = 2,   # playlists must match at least k of those seeds
    pid_cap: int = 1000,                   # *** FIRST PASS HARD CAP: keep top 1,000 playlists by hits ***
    min_co: int = 2,                       # candidate must appear in at least this many matched playlists
    cand_cap: int = 3000,                  # rank only the top-N candidates by weighted co-occurrence
    timeout_seconds: float = 9.0,
    con: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """
    Ultra-fast, read-only, set-weighted playlist co-occurrence recommender.

    1) Choose the K rarest seeds (by global playlist frequency) from the user's playlist.
    2) Find playlists that contain >= min_seed_hits_per_playlist of those seeds; keep **top pid_cap (1,000)** by hits.
    3) Score candidates by weighted co-occurrence (sum of hits across matched playlists).
    4) Trim to top cand_cap candidates, then enrich (track meta + normalization) and rank.
    5) Return 3–5 diversified picks.

    All operations are read-only on the main DB; TEMP tables are in memory.
    """
    t0 = time()
    limit = max(3, min(5, int(limit or 5)))
    seeds_all = _normalize_seeds(seed_uris, max_seeds=max_seeds)
    if not seeds_all:
        return []

    owned_con = False
    if con is None:
        con = _connect_readonly(DB_PATH)
        owned_con = True

    try:
        _apply_read_optimized_pragmas(con)
        cur = con.cursor()

        # Planner hints if indexes exist (no writes needed)
        idx_pt_track = _best_single_col_index(con, "playlist_tracks", "track_uri")
        idx_pt_pid   = _best_single_col_index(con, "playlist_tracks", "pid")
        idx_hint_track = f"INDEXED BY {idx_pt_track}" if idx_pt_track else ""
        idx_hint_pid   = f"INDEXED BY {idx_pt_pid}" if idx_pt_pid else ""

        # TEMP: all seeds
        cur.execute("CREATE TEMP TABLE IF NOT EXISTS seeds (track_uri TEXT PRIMARY KEY) WITHOUT ROWID")
        cur.execute("DELETE FROM seeds")
        cur.executemany("INSERT INTO seeds(track_uri) VALUES (?)", [(s,) for s in seeds_all])

        # Global playlist frequency for our seeds -> pick rarest K
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS seed_counts (
                track_uri TEXT PRIMARY KEY,
                cnt       INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM seed_counts")
        cur.execute(f"""
            INSERT INTO seed_counts(track_uri, cnt)
            SELECT s.track_uri, COUNT(*) AS cnt
            FROM seeds s
            JOIN playlist_tracks pt {idx_hint_track} ON pt.track_uri = s.track_uri
            GROUP BY s.track_uri
        """)

        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS seed_work (
                track_uri TEXT PRIMARY KEY
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM seed_work")
        cur.execute("""
            INSERT INTO seed_work(track_uri)
            SELECT track_uri
            FROM seed_counts
            ORDER BY cnt ASC, track_uri ASC
            LIMIT ?
        """, (rare_seed_sample,))

        # 1) Overlapping playlists, count how many rare seeds they contain
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS pid_seed_hits (
                pid   INTEGER PRIMARY KEY,
                hits  INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM pid_seed_hits")
        cur.execute(f"""
            INSERT INTO pid_seed_hits(pid, hits)
            SELECT pt.pid, COUNT(*) AS hits
            FROM playlist_tracks pt {idx_hint_track}
            JOIN seed_work sw ON sw.track_uri = pt.track_uri
            GROUP BY pt.pid
            HAVING hits >= ?
            ORDER BY hits DESC
            LIMIT ?
        """, (min_seed_hits_per_playlist, pid_cap))

        row = cur.execute("SELECT COUNT(*) AS n, COALESCE(SUM(hits),0) AS w FROM pid_seed_hits").fetchone()
        seed_pid_count = int(row["n"])
        seed_weight_total = int(row["w"])
        if seed_pid_count == 0 or seed_weight_total == 0:
            return []

        # 2) Candidates weighted by hits
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS candidates_raw (
                track_uri TEXT PRIMARY KEY,
                w_co      INTEGER NOT NULL,
                co_count  INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM candidates_raw")
        cur.execute(f"""
            INSERT INTO candidates_raw(track_uri, w_co, co_count)
            SELECT
                pt.track_uri,
                SUM(psh.hits) AS w_co,
                COUNT(*)       AS co_count
            FROM pid_seed_hits psh
            JOIN playlist_tracks pt {idx_hint_pid} ON pt.pid = psh.pid
            WHERE pt.track_uri NOT IN (SELECT track_uri FROM seeds)
            GROUP BY pt.track_uri
            HAVING co_count >= ?
        """, (min_co,))

        # 3) Early trim to reduce downstream joins
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS candidates (
                track_uri TEXT PRIMARY KEY,
                w_co      INTEGER NOT NULL,
                co_count  INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM candidates")
        cur.execute("""
            INSERT INTO candidates(track_uri, w_co, co_count)
            SELECT track_uri, w_co, co_count
            FROM candidates_raw
            ORDER BY w_co DESC, co_count DESC, track_uri ASC
            LIMIT ?
        """, (cand_cap,))

        # 4) Playlist counts only for surviving candidates
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS cand_pl_counts (
                track_uri   TEXT PRIMARY KEY,
                plist_count INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM cand_pl_counts")
        cur.execute(f"""
            INSERT INTO cand_pl_counts(track_uri, plist_count)
            SELECT pt.track_uri, COUNT(*) AS cnt
            FROM playlist_tracks pt {idx_hint_track}
            JOIN candidates c ON c.track_uri = pt.track_uri
            GROUP BY pt.track_uri
        """)

        # 5) Rank & fetch a bit extra, then diversify by artist
        to_fetch = max(16, limit * 12)
        rows = cur.execute("""
            SELECT
                c.track_uri,
                t.artist, t.title, t.album, t.popularity,
                c.w_co,
                c.co_count,
                ? AS seed_pid_count,
                ? AS seed_weight_total,
                plc.plist_count AS track_pid_count,
                CAST(c.w_co AS FLOAT) / ? AS norm_w_score
            FROM candidates c
            JOIN cand_pl_counts plc ON plc.track_uri = c.track_uri
            JOIN tracks t            ON t.track_uri   = c.track_uri
            ORDER BY norm_w_score DESC, c.w_co DESC, c.co_count DESC, COALESCE(t.popularity,0) DESC
            LIMIT ?
        """, (seed_pid_count, seed_weight_total, seed_weight_total, to_fetch)).fetchall()

        seen_artists = set()
        picks: List[RecCandidate] = []
        for r in rows:
            a = (r["artist"] or "").strip().lower()
            if a in seen_artists:
                continue
            seen_artists.add(a)
            picks.append(
                RecCandidate(
                    track_uri=r["track_uri"],
                    artist=r["artist"],
                    title=r["title"],
                    album=r["album"],
                    popularity=r["popularity"],
                    w_co=r["w_co"],
                    co_count=r["co_count"],
                    seed_pid_count=seed_pid_count,
                    seed_weight_total=seed_weight_total,
                    track_pid_count=r["track_pid_count"],
                    score=float(r["norm_w_score"]),
                )
            )
            if len(picks) >= limit:
                break

        if time() - t0 > timeout_seconds:
            picks = picks[:max(3, limit)]

        out: List[Dict[str, Any]] = []
        for r in picks:
            out.append(
                {
                    "track_uri": r.track_uri,
                    "artist": r.artist,
                    "title": r.title,
                    "album": r.album,
                    "popularity": r.popularity,
                    "weighted_overlap": r.w_co,
                    "contributing_playlists": r.co_count,
                    "num_matched_playlists": r.seed_pid_count,
                    "total_seed_hits": r.seed_weight_total,
                    "num_track_playlists": r.track_pid_count,
                    "score": round(r.score, 6),
                    "reason": (
                        f"Strong set overlap: weighted co-occurrence {r.w_co} "
                        f"across {r.co_count} highly-matching playlists"
                    ),
                }
            )
        return out

    finally:
        if owned_con:
            con.close()
