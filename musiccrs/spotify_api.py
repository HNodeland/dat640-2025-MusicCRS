"""Lightweight Spotify helper used only for optional preview/cover lookups.

We do NOT use Spotify for ranking or stats anymore.

Env:
  SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET
"""

from __future__ import annotations
import os
import base64
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")


class _SpotifyLite:
    def __init__(self) -> None:
        self._token: Optional[str] = None

    # -------- internal auth --------
    def _get_token(self) -> Optional[str]:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            return None
        if self._token:
            return self._token
        auth_b64 = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
        try:
            r = requests.post(
                "https://accounts.spotify.com/api/token",
                headers={"Authorization": f"Basic {auth_b64}"},
                data={"grant_type": "client_credentials"},
                timeout=10,
            )
            if r.status_code == 200:
                self._token = r.json().get("access_token")
        except Exception:
            return None
        return self._token

    def _headers(self) -> Dict[str, str]:
        tok = self._get_token()
        return {"Authorization": f"Bearer {tok}"} if tok else {}

    # -------- features we still use --------
    def find_preview_url(self, artist: str, title: str) -> Optional[str]:
        """Search a track and return a 30s preview URL if available."""
        if not title:
            return None
        q = f"track:{title}"
        if artist:
            q += f" artist:{artist}"
        try:
            r = requests.get(
                "https://api.spotify.com/v1/search",
                headers=self._headers(),
                params={"q": q, "type": "track", "limit": 5},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            items = r.json().get("tracks", {}).get("items", []) or []
            for it in items:
                url = it.get("preview_url")
                if url:
                    return url
        except Exception:
            return None
        return None

    def find_cover_image(self, artist: str, title: str) -> Optional[str]:
        """Optional helper: return a cover image URL (largest)."""
        if not title:
            return None
        q = f"track:{title}"
        if artist:
            q += f" artist:{artist}"
        try:
            r = requests.get(
                "https://api.spotify.com/v1/search",
                headers=self._headers(),
                params={"q": q, "type": "track", "limit": 1},
                timeout=10,
            )
            if r.status_code != 200:
                return None
            items = r.json().get("tracks", {}).get("items", []) or []
            if not items:
                return None
            images = (items[0].get("album", {}) or {}).get("images", []) or []
            return images[0]["url"] if images else None
        except Exception:
            return None

    # -------- FIX: accept extra args to be compatible with older callers --------
    def get_track_details(self, track_id: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Fetch track details by Spotify track id.

        Accepts extra positional/keyword args (ignored) for compatibility with
        older callers that passed (track_id, market=...) or similar.
        """
        if not track_id:
            return None
        try:
            r = requests.get(
                f"https://api.spotify.com/v1/tracks/{track_id}",
                headers=self._headers(),
                timeout=10,
            )
            if r.status_code != 200:
                return None
            return r.json()
        except Exception:
            return None


def get_spotify_api() -> Optional[_SpotifyLite]:
    # Always return an instance; methods no-op gracefully if credentials are missing.
    try:
        return _SpotifyLite()
    except Exception:
        return None
