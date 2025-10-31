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
    max_seeds: int = 100,
    rare_seed_sample: Optional[int] = None,
    min_seed_hits_per_playlist: Optional[int] = None,
    pid_cap: Optional[int] = None,
    min_co: int = 1,
    cand_cap: int = 1500,
    timeout_seconds: float = 5.0,
    con: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    """
    Playlist co-occurrence recommender with adaptive parameters.
    
    Parameters scale based on seed playlist size:
    - Small playlists (2-5 tracks) use tighter filtering
    - Larger playlists cast a wider net
    
    All operations are read-only; TEMP tables are in memory.
    """
    t0 = time()
    limit = max(1, min(10, int(limit or 5)))
    seeds_all = _normalize_seeds(seed_uris, max_seeds=max_seeds)
    seed_count = len(seeds_all)
    
    if not seeds_all:
        return []

    # Adaptive parameters based on playlist size
    if rare_seed_sample is None:
        if seed_count <= 5:
            rare_seed_sample = seed_count
        elif seed_count <= 15:
            rare_seed_sample = min(seed_count, 12)
        else:
            rare_seed_sample = min(seed_count, 24)
    
    if min_seed_hits_per_playlist is None:
        if seed_count <= 3:
            min_seed_hits_per_playlist = 1
        elif seed_count <= 10:
            min_seed_hits_per_playlist = max(1, seed_count // 4)
        else:
            min_seed_hits_per_playlist = max(2, seed_count // 5)
    
    if pid_cap is None:
        if seed_count <= 5:
            pid_cap = 150
        elif seed_count <= 15:
            pid_cap = 300
        elif seed_count <= 30:
            pid_cap = 500
        else:
            pid_cap = 800

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

        cur.execute("CREATE TEMP TABLE IF NOT EXISTS seeds (track_uri TEXT PRIMARY KEY) WITHOUT ROWID")
        cur.execute("DELETE FROM seeds")
        cur.executemany("INSERT INTO seeds(track_uri) VALUES (?)", [(s,) for s in seeds_all])

        # Global playlist frequency for our seeds
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

        # Check if we have extremely popular seeds
        max_seed_pop = cur.execute("SELECT MAX(cnt) FROM seed_counts").fetchone()[0]
        
        # For mega-popular tracks, we need a completely different strategy
        # These tracks appear in so many playlists that normal filtering doesn't work
        use_fast_path = seed_count <= 10 and max_seed_pop < 10000
        use_mega_popular_path = seed_count <= 20 and max_seed_pop >= 10000
        
        # Adjust parameters for very popular tracks
        if max_seed_pop >= 10000:
            # For mega-hits, we need to sample playlists very aggressively
            rare_seed_sample = min(rare_seed_sample, max(2, seed_count // 2))
            pid_cap = min(pid_cap, 500)  # Need more playlists for mega-popular tracks
            # For mega-popular, allow single seed matches (tracks rarely co-occur)
            min_seed_hits_per_playlist = 1

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

        # FAST PATH: For small playlists with moderately popular tracks, use direct co-occurrence
        # This bypasses the playlist aggregation step entirely
        if use_fast_path:
            # Define to_fetch for this path
            to_fetch = min(cand_cap, limit * 20)  # Fetch enough to account for artist deduplication
            
            # Clean up any existing temp tables first
            cur.execute("DROP TABLE IF EXISTS quick_candidates")
            
            # Direct approach: Find tracks that co-occur with our seeds
            cur.execute(f"""
                CREATE TEMP TABLE quick_candidates AS
                SELECT 
                    pt2.track_uri,
                    COUNT(DISTINCT pt2.pid) as co_count,
                    SUM(CASE 
                        WHEN pt1.track_uri IN (SELECT track_uri FROM seed_work LIMIT 1) THEN 3
                        WHEN pt1.track_uri IN (SELECT track_uri FROM seed_work) THEN 2
                        ELSE 1
                    END) as w_co
                FROM playlist_tracks pt1 {idx_hint_track}
                JOIN seed_work sw ON sw.track_uri = pt1.track_uri
                JOIN playlist_tracks pt2 {idx_hint_pid} ON pt2.pid = pt1.pid
                WHERE pt2.track_uri NOT IN (SELECT track_uri FROM seeds)
                  AND pt1.pid IN (
                      SELECT DISTINCT pid 
                      FROM playlist_tracks 
                      WHERE track_uri IN (SELECT track_uri FROM seed_work)
                      LIMIT {pid_cap * 2}
                  )
                GROUP BY pt2.track_uri
                HAVING co_count >= {min_co}
                ORDER BY w_co DESC, co_count DESC
                LIMIT {cand_cap}
            """)
            
            # Get results with track metadata
            rows = cur.execute("""
                SELECT
                    qc.track_uri,
                    t.artist, t.title, t.album, t.popularity,
                    qc.w_co,
                    qc.co_count,
                    ? AS seed_pid_count,
                    ? AS seed_weight_total,
                    qc.co_count AS track_pid_count,
                    CAST(qc.w_co AS FLOAT) / ? AS norm_w_score
                FROM quick_candidates qc
                JOIN tracks t ON t.track_uri = qc.track_uri
                ORDER BY norm_w_score DESC, qc.w_co DESC
                LIMIT ?
            """, (seed_count, seed_count * 10, max(1, seed_count * 10), to_fetch)).fetchall()
            
            seen_artists = set()
            picks: List[RecCandidate] = []
            for r in rows:
                if time() - t0 > timeout_seconds:
                    break
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
                        seed_pid_count=r["seed_pid_count"],
                        seed_weight_total=r["seed_weight_total"],
                        track_pid_count=r["track_pid_count"],
                        score=r["norm_w_score"],
                    )
                )
                if len(picks) >= limit:
                    break
            
            elapsed = time() - t0
            return [
                {
                    "track_uri": c.track_uri,
                    "artist": c.artist,
                    "title": c.title,
                    "album": c.album,
                    "score": c.score,
                    "w_co": c.w_co,
                    "co_count": c.co_count,
                }
                for c in picks
            ]

        # MEGA-POPULAR FAST PATH: For extremely popular tracks (10k+ playlists)
        # Sample playlists more aggressively and require multiple seed hits
        if use_mega_popular_path:
            # Define to_fetch for this path
            to_fetch = min(cand_cap, limit * 20)
            
            # Clean up any existing temp tables first
            cur.execute("DROP TABLE IF EXISTS mega_popular_pids")
            cur.execute("DROP TABLE IF EXISTS quick_candidates")
            
            # For mega-popular tracks, find playlists that contain MULTIPLE seeds
            # This dramatically reduces the search space
            cur.execute(f"""
                CREATE TEMP TABLE mega_popular_pids AS
                SELECT pt.pid, COUNT(DISTINCT pt.track_uri) as seed_hit_count
                FROM playlist_tracks pt {idx_hint_track}
                JOIN seed_work sw ON sw.track_uri = pt.track_uri
                GROUP BY pt.pid
                HAVING seed_hit_count >= {min_seed_hits_per_playlist}
                ORDER BY seed_hit_count DESC
                LIMIT {pid_cap}
            """)
            
            # Now find co-occurring tracks in these playlists
            cur.execute(f"""
                CREATE TEMP TABLE quick_candidates AS
                SELECT 
                    pt.track_uri,
                    COUNT(DISTINCT pt.pid) as co_count,
                    SUM(mpp.seed_hit_count) as w_co
                FROM mega_popular_pids mpp
                JOIN playlist_tracks pt {idx_hint_pid} ON pt.pid = mpp.pid
                WHERE pt.track_uri NOT IN (SELECT track_uri FROM seeds)
                GROUP BY pt.track_uri
                HAVING co_count >= {min_co}
                ORDER BY w_co DESC, co_count DESC
                LIMIT {cand_cap}
            """)
            
            # Get results with track metadata
            rows = cur.execute("""
                SELECT
                    qc.track_uri,
                    t.artist, t.title, t.album, t.popularity,
                    qc.w_co,
                    qc.co_count,
                    ? AS seed_pid_count,
                    ? AS seed_weight_total,
                    qc.co_count AS track_pid_count,
                    CAST(qc.w_co AS FLOAT) / CAST(? AS FLOAT) AS norm_w_score
                FROM quick_candidates qc
                JOIN tracks t ON t.track_uri = qc.track_uri
                ORDER BY norm_w_score DESC, qc.w_co DESC
                LIMIT ?
            """, (seed_count, seed_count * 10, max(1, seed_count * 10), to_fetch)).fetchall()
            
            seen_artists = set()
            picks: List[RecCandidate] = []
            for r in rows:
                if time() - t0 > timeout_seconds:
                    break
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
                        seed_pid_count=seed_count,
                        seed_weight_total=seed_count * 10,
                        track_pid_count=r["track_pid_count"],
                        score=float(r["norm_w_score"]),
                    )
                )
                if len(picks) >= limit:
                    break
            
            # Return early with fast results
            out: List[Dict[str, Any]] = []
            for r in picks:
                out.append({
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
                })
            return out

        # 1) Overlapping playlists, count how many rare seeds they contain
        # CRITICAL OPTIMIZATION: Add LIMIT much earlier to avoid scanning too many rows
        cur.execute("""
            CREATE TEMP TABLE IF NOT EXISTS pid_seed_hits (
                pid   INTEGER PRIMARY KEY,
                hits  INTEGER NOT NULL
            ) WITHOUT ROWID
        """)
        cur.execute("DELETE FROM pid_seed_hits")
        
        # Use a subquery with LIMIT to restrict the initial scan
        cur.execute(f"""
            INSERT INTO pid_seed_hits(pid, hits)
            SELECT pid, hits
            FROM (
                SELECT pt.pid, COUNT(*) AS hits
                FROM playlist_tracks pt {idx_hint_track}
                JOIN seed_work sw ON sw.track_uri = pt.track_uri
                GROUP BY pt.pid
                HAVING hits >= ?
                ORDER BY hits DESC
                LIMIT ?
            )
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

        # 5) Rank & fetch adaptively - small playlists need fewer candidates
        if seed_count <= 5:
            to_fetch = max(8, limit * 4)  # Much smaller for tiny playlists
        elif seed_count <= 15:
            to_fetch = max(12, limit * 6)
        else:
            to_fetch = max(16, limit * 8)  # Reduced from 12x
        
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
        """, (seed_pid_count, seed_weight_total, max(1, seed_weight_total), to_fetch)).fetchall()

        seen_artists = set()
        picks: List[RecCandidate] = []
        for r in rows:
            # Check timeout early
            if time() - t0 > timeout_seconds:
                break
                
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
