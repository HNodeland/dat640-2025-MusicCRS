from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

from .playlist_db import (
    ensure_db,
    search_by_artist_title,
    get_track_by_uri,
    search_by_title,
)
from .cover_art import generate_cover


@dataclass
class Track:
    track_uri: str
    artist: str
    title: str
    album: Optional[str] = None

    def to_public(self) -> dict:
        # Frontend compat: expose both `uri` and `track_uri`
        d = asdict(self)
        d["uri"] = self.track_uri
        return d


@dataclass
class Playlist:
    name: str
    tracks: List[Track] = field(default_factory=list)
    cover_url: Optional[str] = None  # internal preferred name

    def to_public(self) -> dict:
        cover = self.cover_url or ""
        return {
            "name": self.name,
            "tracks": [t.to_public() for t in self.tracks],
            "cover_url": cover,   # field used by the frontend
            "cover": cover,       # alias for any older codepath
            "count": len(self.tracks),
        }


class PlaylistService:
    def __init__(self) -> None:
        self._by_user: Dict[str, Dict[str, Playlist]] = {}
        self._current: Dict[str, str] = {}

    # ---------- user + playlist management ----------
    def _ensure_user(self, user_id: str) -> None:
        if user_id not in self._by_user:
            self._by_user[user_id] = {"default": Playlist("default")}
            self._current[user_id] = "default"
            self._refresh_cover(user_id, "default")

    def list_playlists(self, user_id: str) -> List[str]:
        self._ensure_user(user_id)
        return list(self._by_user[user_id].keys())

    def current_playlist(self, user_id: str) -> Playlist:
        self._ensure_user(user_id)
        return self._by_user[user_id][self._current[user_id]]

    def create_playlist(self, user_id: str, name: str) -> Playlist:
        self._ensure_user(user_id)
        if name in self._by_user[user_id]:
            raise ValueError(f"Playlist '{name}' already exists.")
        self._by_user[user_id][name] = Playlist(name=name)
        self._current[user_id] = name
        self._refresh_cover(user_id, name)
        return self._by_user[user_id][name]

    def switch_playlist(self, user_id: str, name: str) -> Playlist:
        self._ensure_user(user_id)
        if name not in self._by_user[user_id]:
            raise ValueError(f"Playlist '{name}' does not exist.")
        self._current[user_id] = name
        return self._by_user[user_id][name]

    def clear(self, user_id: str) -> None:
        pl = self.current_playlist(user_id)
        pl.tracks.clear()
        self._refresh_cover(user_id, pl.name)

    # legacy alias (some original code referenced this)
    def clear_playlist(self, user_id: str) -> None:
        self.clear(user_id)

    # ---------- track operations ----------
    def add_track_by_artist_title(self, user_id: str, artist: str, title: str) -> Track:
        rows = search_by_artist_title(artist, title, limit=1)
        if not rows:
            raise ValueError("Track not found in database.")
        uri, a, t, album = rows[0]
        return self._add_by_uri_internal(user_id, uri, a, t, album)

    def add_by_uri(self, user_id: str, track_uri: str) -> Track:
        row = get_track_by_uri(track_uri)
        if not row:
            raise ValueError("Track URI not found in database.")
        uri, a, t, album = row
        return self._add_by_uri_internal(user_id, uri, a, t, album)

    def _add_by_uri_internal(self, user_id: str, uri: str, artist: str, title: str, album: Optional[str]) -> Track:
        pl = self.current_playlist(user_id)
        tr = Track(track_uri=uri, artist=artist, title=title, album=album)
        if any(x.track_uri == uri for x in pl.tracks):
            # still refresh cover (e.g., first track after clear)
            self._refresh_cover(user_id, pl.name)
            return tr
        pl.tracks.append(tr)
        self._refresh_cover(user_id, pl.name)
        return tr

    def remove(self, user_id: str, identifier: str) -> Track:
        pl = self.current_playlist(user_id)
        # by index (1-based)
        if identifier.isdigit():
            idx = int(identifier) - 1
            if 0 <= idx < len(pl.tracks):
                tr = pl.tracks.pop(idx)
                self._refresh_cover(user_id, pl.name)
                return tr
        # by URI
        for i, t in enumerate(pl.tracks):
            if t.track_uri == identifier or getattr(t, "uri", None) == identifier:
                tr = pl.tracks.pop(i)
                self._refresh_cover(user_id, pl.name)
                return tr
        raise ValueError("Track not found in current playlist.")

    # legacy alias (some original code referenced this)
    def remove_track(self, user_id: str, identifier: str) -> Track:
        return self.remove(user_id, identifier)

    # ---------- DB search with MPD popularity ranking ----------
    def search_tracks_by_title(
        self, user_id: str, title: str, fetch_all: bool = False, limit: int = 20
    ) -> List[Tuple[str, str, str, Optional[str]]]:
        """
        Ranked by:
          1) whether artist already appears in current playlist,
          2) MPD popularity (tracks.popularity),
          3) artist/title alphabetical.
        """
        self._ensure_user(user_id)
        pl = self.current_playlist(user_id)
        existing = set(t.artist.lower() for t in pl.tracks)

        base_limit = 500 if fetch_all else max(20, limit)
        rows = search_by_title(title, limit=base_limit)  # (uri, artist, title, album, popularity)
        rows4 = [(r[0], r[1], r[2], r[3]) for r in rows]

        # popularity map (rows already include popularity; keep robust)
        pop_map: dict[str, int] = {}
        if rows:
            for r in rows:
                pop_map[r[0]] = int(r[4] or 0)

        def key(item):
            uri, artist, title, album = item
            in_pl = 1 if artist.lower() in existing else 0
            pop = pop_map.get(uri, 0)
            return (-in_pl, -pop, artist.lower(), title.lower())

        rows_sorted = sorted(rows4, key=key)
        return rows_sorted if fetch_all else rows_sorted[:limit]

    def get_popularity(self, track_uri: str) -> int:
        con = ensure_db()
        try:
            r = con.execute(
                "SELECT COALESCE(popularity,0) AS p FROM tracks WHERE track_uri=?",
                (track_uri,),
            ).fetchone()
            return int(r["p"] or 0) if r else 0
        finally:
            con.close()

    # ---------- stats ----------
    def get_playlist_stats(self, user_id: str) -> dict:
        self._ensure_user(user_id)
        pl = self.current_playlist(user_id)
        name = pl.name
        n = len(pl.tracks)
        if n == 0:
            return {
                "playlist_name": name,
                "total_tracks": 0,
                "unique_artists": 0,
                "unique_albums": 0,
                "avg_popularity": 0,
                "avg_popularity_k": "0k",
                "top_artists": [],
                "top_albums": [],
            }

        from collections import Counter
        artist_counts: Dict[str, int] = Counter(t.artist for t in pl.tracks)
        album_counts: Dict[str, int] = Counter((t.album or "single") for t in pl.tracks)
        pops = [self.get_popularity(t.track_uri) for t in pl.tracks]
        avg_pop = sum(pops) // len(pops) if pops else 0

        def fmt_k(n: int) -> str:
            v = n / 1000.0
            s = f"{v:.1f}"
            if s.endswith(".0"):
                s = s[:-2]
            return f"{s}k"

        return {
            "playlist_name": name,
            "total_tracks": n,
            "unique_artists": len(artist_counts),
            "unique_albums": len(album_counts),
            "avg_popularity": avg_pop,
            "avg_popularity_k": fmt_k(avg_pop),
            "top_artists": artist_counts.most_common(5),
            "top_albums": album_counts.most_common(5),
        }

    # ---------- cover + payload ----------
    def _refresh_cover(self, user_id: str, name: str):
        pl = self._by_user[user_id][name]
        pl.cover_url = generate_cover(user_id, pl)

    def serialize_state(self, user_id: str) -> dict:
        """(Kept) full state — not used by the web UI parser, but available."""
        self._ensure_user(user_id)
        current = self._current[user_id]
        by_name = {k: v.to_public() for k, v in self._by_user[user_id].items()}
        return {
            "current": current,
            "current_playlist": current,
            "playlists": by_name,
            "playlists_by_name": by_name,
        }

    def serialize_current_playlist(self, user_id: str) -> dict:
        """Single playlist object — exactly what the web UI expects in <!--PLAYLIST:{...}-->."""
        self._ensure_user(user_id)
        return self.current_playlist(user_id).to_public()
