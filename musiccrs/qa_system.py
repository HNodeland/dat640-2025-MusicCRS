"""QA system answered from the local MPD-based database (no Spotify)."""

from __future__ import annotations
import re
import time
from typing import Optional, List, Tuple, Dict

from .playlist_db import (
    ensure_db,
    get_track,
    search_by_artist_title,
    count_tracks_by_artist,
)
from .ir_search import search_tracks_ir


def _clean_text(s: str) -> str:
    """Trim, remove surrounding quotes, and strip trailing punctuation like ?, . , !"""
    if s is None:
        return ""
    s = s.strip().strip('"').strip("'").strip()
    s = re.sub(r"[?\.\!]+$", "", s).strip()
    return s


class QASystem:
    """
    Supports DB-based questions (case-insensitive):

      1) Who sings/performs a song:
         - who sings <title>
         - who performs <title>
         - who is the artist of <title>

      2) Album of a song (robust):
         - what/which album is <title> by <artist> [on]
         - what/which album does <title> by <artist> appear on

      3) Count songs by an artist:
         - how many (tracks|songs) by <artist>

      4) List albums by an artist:
         - what albums does <artist> have
         - (list|show) albums by <artist>

      5) Similar artists (co-occurrence in MPD playlists, if available):
         - who sounds like <artist>
         - who is similar to <artist>

      6) Availability check:
         - do you have <title> by <artist>

      7) Fallback suggestions by title when nothing matches.
    """

    def __init__(self) -> None:
        # Who sings: "who sings <title>"
        self._re_who_sings = re.compile(
            r"^who\s+(?:sings|performs|sang|is\s+the\s+artist\s+of)\s+(.+?)\s*[?.!]*$",
            re.IGNORECASE,
        )
        # Album: "... is <title> by <artist> [on]"
        self._re_album_is_on = re.compile(
            r"^(?:what|which)\s+album\s+is\s+(.+?)\s+by\s+(.+?)(?:\s+on)?\s*[?.!]*$",
            re.IGNORECASE,
        )
        # Album: "... does <title> by <artist> appear on"
        self._re_album_appear_on = re.compile(
            r"^(?:what|which)\s+album\s+does\s+(.+?)\s+by\s+(.+?)\s+appear\s+on\s*[?.!]*$",
            re.IGNORECASE,
        )
        self._re_how_many_by = re.compile(
            r"^how\s+many\s+(?:tracks|songs)\s+by\s+(.+?)\s*[?.!]*$",
            re.IGNORECASE,
        )
        # Albums by artist
        self._re_albums_have = re.compile(
            r"^what\s+albums\s+does\s+(.+?)\s+have\s*[?.!]*$",
            re.IGNORECASE,
        )
        self._re_albums_by = re.compile(
            r"^(?:list|show)\s+albums\s+by\s+(.+?)\s*[?.!]*$",
            re.IGNORECASE,
        )
        # Similar artists
        self._re_similar = re.compile(
            r"^who\s+(?:sounds\s+like|is\s+similar\s+to)\s+(.+?)\s*[?.!]*$",
            re.IGNORECASE,
        )
        # Availability
        self._re_have = re.compile(
            r"^do\s+you\s+have\s+(.+?)\s+by\s+(.+?)\s*[?.!]*$",
            re.IGNORECASE,
        )

    # ---------- helpers ----------
    def _album_of(self, title: str, artist: str) -> Optional[str]:
        """Try exact match first; if not found, fall back to LIKE and pick most popular."""
        title_c = _clean_text(title)
        artist_c = _clean_text(artist)

        # 1) exact
        row = get_track(artist_c, title_c)
        if row:
            return row[3] or "single"

        # 2) fallback: LIKE + choose highest popularity
        con = ensure_db()
        try:
            r = con.execute(
                """
                SELECT album, COALESCE(popularity,0) AS p
                FROM tracks
                WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                ORDER BY p DESC
                LIMIT 1
                """,
                (f"%{artist_c.lower()}%", f"%{title_c.lower()}%"),
            ).fetchone()
            if r:
                return r["album"] or "single"
        finally:
            con.close()
        return None

    def _albums_by_artist(self, artist: str, limit: int = 20) -> List[Tuple[str, int]]:
        """Return [(album, count)] for artist, ordered by track count then popularity."""
        a = _clean_text(artist).lower()
        con = ensure_db()
        try:
            rows = con.execute(
                """
                SELECT album,
                       COUNT(*) AS c,
                       SUM(COALESCE(popularity,0)) AS p
                FROM tracks
                WHERE album IS NOT NULL AND album <> ''
                  AND LOWER(artist) LIKE ?
                GROUP BY album
                ORDER BY c DESC, p DESC, album COLLATE NOCASE ASC
                LIMIT ?
                """,
                (f"%{a}%", limit),
            ).fetchall()
            return [(r["album"], r["c"]) for r in rows]
        finally:
            con.close()

    def _detect_playlist_tracks_table(self) -> Optional[Tuple[str, str, str]]:
        """Try to find a playlist-to-tracks mapping table.

        Returns (table_name, playlist_col, track_col) or None.
        We look for common table names and attempt to detect likely column names.
        """
        candidates = [
            "playlist_tracks",
            "playlists_tracks",
            "mpd_playlist_tracks",
            "playlist_to_tracks",
            "tracks_in_playlists",
        ]
        con = ensure_db()
        try:
            # figure out which table exists
            tbl = None
            for name in candidates:
                row = con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (name,),
                ).fetchone()
                if row:
                    tbl = name
                    break
            if not tbl:
                return None

            # inspect columns
            cols = [r["name"] for r in con.execute(f"PRAGMA table_info({tbl})").fetchall()]
            # heuristics
            track_candidates = [c for c in cols if "track" in c.lower()]
            pl_candidates = [c for c in cols if "play" in c.lower() or c.lower() in ("pid", "playlist_id")]
            if not track_candidates or not pl_candidates:
                return None
            return (tbl, pl_candidates[0], track_candidates[0])
        finally:
            con.close()

    def _similar_artists(self, artist: str, limit: int = 10) -> Optional[List[Tuple[str, int]]]:
        """Artists frequently co-occurring with the given artist on the same playlists.

        Requires a playlist<->tracks mapping table. If not available, returns None.
        
        SUPER OPTIMIZED: Samples smaller sets for <2 second response time.
        """
        detected = self._detect_playlist_tracks_table()
        if not detected:
            return None

        tbl, pl_col, tr_col = detected
        a = _clean_text(artist).lower()

        con = ensure_db()
        con.execute("PRAGMA temp_store = MEMORY")
        try:
            # Step 1: Get sample of playlists with this artist (FAST - uses index)
            # Reduced limits for faster response: 200 tracks -> ~1000 playlists
            sample_playlists = con.execute(f"""
                SELECT DISTINCT pt.{pl_col} AS pid
                FROM {tbl} pt
                INDEXED BY idx_pt_track
                WHERE pt.{tr_col} IN (
                    SELECT track_uri FROM tracks 
                    WHERE LOWER(artist) = ?
                    LIMIT 200
                )
                LIMIT 1000
            """, (a,)).fetchall()
            
            if not sample_playlists:
                return []
            
            pids = [r["pid"] for r in sample_playlists]
            placeholders = ",".join("?" * len(pids))
            
            # Step 2: Find co-occurring artists in these playlists (FAST - small set)
            rows = con.execute(f"""
                SELECT LOWER(t.artist) AS artist, COUNT(*) AS cnt
                FROM {tbl} pt
                JOIN tracks t ON t.track_uri = pt.{tr_col}
                WHERE pt.{pl_col} IN ({placeholders})
                  AND LOWER(t.artist) <> ?
                GROUP BY LOWER(t.artist)
                ORDER BY cnt DESC
                LIMIT ?
            """, pids + [a, limit]).fetchall()
            
            return [(r["artist"], r["cnt"]) for r in rows]
        except Exception as e:
            # If query times out or fails, return empty
            return []
        finally:
            con.close()

    # ---------- help ----------
    def help_text(self) -> str:
        """HTML help for /ask."""
        return (
            "<b>/ask — supported question types</b><br/>"
            "<ol>"
            "<li><b>Who sings a song</b><br/>"
            "<code>who sings &lt;title&gt;</code><br/>"
            "<code>who performs &lt;title&gt;</code></li>"
            "<li><b>Album of a song</b><br/>"
            "<code>what album is &lt;title&gt; by &lt;artist&gt; [on]</code><br/>"
            "<code>which album does &lt;title&gt; by &lt;artist&gt; appear on</code></li>"
            "<li><b>Count songs by an artist</b><br/>"
            "<code>how many songs by &lt;artist&gt;</code></li>"
            "<li><b>List albums by an artist</b><br/>"
            "<code>what albums does &lt;artist&gt; have</code><br/>"
            "<code>list albums by &lt;artist&gt;</code> / <code>show albums by &lt;artist&gt;</code></li>"
            "<li><b>Similar artists</b> (based on playlist co-occurrence)<br/>"
            "<code>who sounds like &lt;artist&gt;</code> / <code>who is similar to &lt;artist&gt;</code><br/>"
            "<span style='opacity:.7'>Note: May take 1-2 seconds for popular artists</span></li>"
            "<li><b>Availability check</b><br/>"
            "<code>do you have &lt;title&gt; by &lt;artist&gt;</code></li>"
            "<li><b>Fallback title suggestions</b><br/>"
            "If a question doesn't match these patterns, we try it as a title and show up to 5 matches.</li>"
            "</ol>"
            "All answers come from the local MPD-derived database; matching is case-insensitive."
        )

    # ---------- public ----------
    def answer_question(self, question: str) -> str:
        start_time = time.time()
        q = (question or "").strip()

        # Who sings "<title>"
        m = self._re_who_sings.search(q)
        if m:
            title = _clean_text(m.group(1))
            # Search for tracks matching the title
            conn = ensure_db()
            try:
                results = search_tracks_ir(conn, title, limit=5, min_score=0.3)
            finally:
                conn.close()
            
            latency_ms = (time.time() - start_time) * 1000
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
            
            if results:
                if len(results) == 1:
                    # Single clear match
                    artist, track_title, album = results[0][1], results[0][2], results[0][3] or "single"
                    return f'<b>{artist}</b> sings "{track_title}" (from album "{album}").{timing_info}'
                else:
                    # Multiple matches - show all options
                    items = [f"<li><b>{r[1]}</b> – {r[2]} <span style='opacity:.7'>({r[3] or 'single'})</span></li>" for r in results]
                    return f"Found multiple matches for \"{title}\":{timing_info}<ol>{''.join(items)}</ol>"
            
            return f'I couldn\'t find a track called "{title}" in the database.{timing_info}'

        # Album of "<title>" by <artist> (two phrasings, tolerant)
        for rx in (self._re_album_is_on, self._re_album_appear_on):
            m = rx.search(q)
            if m:
                title, artist = _clean_text(m.group(1)), _clean_text(m.group(2))
                album = self._album_of(title, artist)
                latency_ms = (time.time() - start_time) * 1000
                timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
                if album:
                    return f'"{title}" by {artist} is on the album "{album}".{timing_info}'
                return f'I couldn\'t find "{title}" by {artist}.{timing_info}'

        # How many songs by <artist>
        m = self._re_how_many_by.search(q)
        if m:
            artist = _clean_text(m.group(1))
            n = count_tracks_by_artist(artist)
            latency_ms = (time.time() - start_time) * 1000
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
            return f"There are {n} tracks by {artist} in the database.{timing_info}"

        # Albums by <artist>
        m = self._re_albums_have.search(q) or self._re_albums_by.search(q)
        if m:
            # group(1) works for both patterns due to the way they are defined
            artist = _clean_text(m.group(1) or m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1))
            albs = self._albums_by_artist(artist, limit=20)
            latency_ms = (time.time() - start_time) * 1000
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
            if not albs:
                return f"I couldn't find albums by {artist}.{timing_info}"
            items = [f"<li><i>{alb}</i> <span style='opacity:.7'>(×{cnt} tracks)</span></li>" for alb, cnt in albs]
            return f"Albums by <b>{artist}</b>:{timing_info}<ol>{''.join(items)}</ol>"

        # Similar artists
        m = self._re_similar.search(q)
        if m:
            artist = _clean_text(m.group(1))
            sims = self._similar_artists(artist, limit=10)
            latency_ms = (time.time() - start_time) * 1000
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
            if sims is None:
                return f"Similar-artist lookup isn't available in this database build.{timing_info}"
            if not sims or len(sims) == 0:
                return f"I couldn't find artists similar to {artist} (or the query took too long).{timing_info}"
            items = [f"{a} (co-occur {c} times)" for a, c in sims]
            return f"Artists often played with that artist:{timing_info}<br/>" + "<br/>".join(items)

        # Availability
        m = self._re_have.search(q)
        if m:
            title, artist = _clean_text(m.group(1)), _clean_text(m.group(2))
            row = get_track(artist, title)
            latency_ms = (time.time() - start_time) * 1000
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
            if row:
                album = row[3] or "single"
                return f'Yes — "{title}" by {artist} is in the database (album "{album}").{timing_info}'
            # Suggest a few close matches
            suggestions = search_by_artist_title(artist, title, limit=5)
            if suggestions:
                opts = [f"{r[1]} – {r[2]} ({r[3] or 'single'})" for r in suggestions]
                return f"I didn't find an exact match. Did you mean:{timing_info}<br/>" + "<br/>".join(opts)
            return f'I couldn\'t find "{title}" by {artist}.{timing_info}'

        # Fallback: treat as a title and suggest up to 5 using IR search
        conn = ensure_db()
        try:
            ir_results = search_tracks_ir(conn, _clean_text(q), limit=5, min_score=0.2)
        finally:
            conn.close()
        
        latency_ms = (time.time() - start_time) * 1000
        timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Q&A: {latency_ms:.0f}ms]</span>"
        
        if ir_results:
            lis = [f"{r[1]} – {r[2]} ({r[3] or 'single'})" for r in ir_results]
            return f"Did you mean:{timing_info}<br/>" + "<br/>".join(lis)

        return f"Sorry, I couldn't answer that from the database.{timing_info}"
