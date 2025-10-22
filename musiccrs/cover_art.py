"""Playlist cover generation with real Spotify covers and a 2x2 collage.

Order of attempts:
1) Spotify oEmbed (no auth) via track URI -> thumbnail_url
2) Spotify Web API /v1/tracks (if credentials available)
3) Web API search by artist/title (if credentials)
4) Build a 2x2 collage (Pillow). If Pillow isn't installed or downloads fail,
   fall back to returning the first image URL.
5) Final fallback is a deterministic SVG.
"""

from __future__ import annotations

import base64
import io
from hashlib import md5
from typing import List, Optional
from urllib.parse import quote

import requests

from .spotify_api import get_spotify_api


# -------------------- helpers: fallback SVG -------------------- #
def _inline_svg_cover(text: str) -> str:
    """Create a simple deterministic SVG cover as a data URL."""
    title = (text or "Playlist")
    h = md5(title.encode("utf-8")).hexdigest()
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    bg = f"rgb({r},{g},{b})"
    title_short = title[:22]
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='512' height='512'>
  <rect width='100%' height='100%' fill='{bg}'/>
  <text x='50%' y='50%' dominant-baseline='middle' text-anchor='middle'
        font-family='Arial, Helvetica, sans-serif' font-size='36' fill='white'>{title_short}</text>
</svg>"""
    return f"data:image/svg+xml;utf8,{quote(svg)}"""


# -------------------- helpers: Spotify lookups -------------------- #
def _track_id_from_uri(uri: str) -> Optional[str]:
    if not uri or not uri.startswith("spotify:track:"):
        return None
    parts = uri.split(":")
    return parts[-1] if len(parts) >= 3 else None


def _oembed_album_image(track_uri: str) -> Optional[str]:
    """Tokenless: use Spotify oEmbed to get a thumbnail URL for a track."""
    tid = _track_id_from_uri(track_uri)
    if not tid:
        return None
    try:
        # Works without auth; returns JSON with thumbnail_url
        r = requests.get(
            "https://open.spotify.com/oembed",
            params={"url": f"spotify:track:{tid}"},
            timeout=8,
            headers={"User-Agent": "MusicCRS/cover-art"},
        )
        if r.status_code != 200:
            return None
        data = r.json()
        url = data.get("thumbnail_url")
        return url
    except Exception:
        return None


def _album_image_from_track_api(track_uri: str) -> Optional[str]:
    """If credentials are available, read album image from /v1/tracks/{id}."""
    tid = _track_id_from_uri(track_uri)
    if not tid:
        return None
    sp = get_spotify_api()
    if not sp:
        return None
    details = sp.get_track_details(tid)  # tolerant signature
    if not details:
        return None
    album = (details.get("album") or {})
    images = (album.get("images") or [])
    return images[0]["url"] if images else None


def _search_album_image(artist: Optional[str], title: Optional[str]) -> Optional[str]:
    """Fallback: search by (artist, title) and return the largest album image URL."""
    if not title:
        return None
    sp = get_spotify_api()
    if not sp:
        return None
    return sp.find_cover_image(artist or "", title)


# -------------------- helpers: HTTP + collage -------------------- #
def _download_image(url: str):
    """Download an image and return a Pillow Image, or None on failure."""
    try:
        r = requests.get(url, headers={"User-Agent": "MusicCRS/cover-art"}, timeout=10)
        if r.status_code != 200 or not r.content:
            return None
        from PIL import Image  # lazy import
        img = Image.open(io.BytesIO(r.content))
        return img.convert("RGB")
    except Exception:
        return None


def _make_collage(urls: List[str], size: int = 512) -> Optional[str]:
    """Compose a 2x2 collage from up to 4 image URLs and return as data URL (JPEG).
    If Pillow isn't installed or all downloads fail, return None.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    if not urls:
        return None

    # Download up to 4 images
    imgs = []
    for u in urls[:4]:
        img = _download_image(u)
        if img:
            imgs.append(img)
    if not imgs:
        return None

    N = len(imgs)
    tile = size // 2
    canvas = Image.new("RGB", (size, size), (30, 30, 30))

    def fit(img):
        return img.resize((tile, tile), Image.LANCZOS)

    if N == 1:
        im = imgs[0].resize((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90, optimize=True)
    elif N == 2:
        canvas.paste(fit(imgs[0]), (0, 0))
        canvas.paste(fit(imgs[1]), (tile, 0))
        canvas.paste(fit(imgs[0]), (0, tile))
        canvas.paste(fit(imgs[1]), (tile, tile))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90, optimize=True)
    elif N == 3:
        canvas.paste(fit(imgs[0]), (0, 0))
        canvas.paste(fit(imgs[1]), (tile, 0))
        canvas.paste(fit(imgs[2]), (0, tile))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90, optimize=True)
    else:
        canvas.paste(fit(imgs[0]), (0, 0))
        canvas.paste(fit(imgs[1]), (tile, 0))
        canvas.paste(fit(imgs[2]), (0, tile))
        canvas.paste(fit(imgs[3]), (tile, tile))
        buf = io.BytesIO()
        canvas.save(buf, format="JPEG", quality=90, optimize=True)

    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# -------------------- main API -------------------- #
def generate_cover(user_id: str, playlist) -> str:
    """Return a URL (or data: URL) for the playlist cover. Never raises.

    * Uses the first up to 4 tracks of the playlist.
    * Returns a real album cover as soon as you add the first song.
    * Collage (2x2) used when multiple covers are available (requires Pillow).
    """
    name = getattr(playlist, "name", "") or "Playlist"
    tracks = getattr(playlist, "tracks", None) or []

    urls: List[str] = []
    seen = set()

    # Collect up to 4 covers for the first 4 tracks
    for t in tracks[:4]:
        uri = getattr(t, "track_uri", "") or getattr(t, "uri", "")
        artist = getattr(t, "artist", None)
        title = getattr(t, "title", None)

        # 1) Tokenless oEmbed
        for u in (
            _oembed_album_image(uri),
            _album_image_from_track_api(uri),   # 2) Web API (auth)
            _search_album_image(artist, title), # 3) Search (auth)
        ):
            if u and u not in seen:
                urls.append(u)
                seen.add(u)
                break

    # Try to build a collage; if that fails but we have URLs, return the first URL
    collage = _make_collage(urls, size=512)
    if collage:
        return collage
    if urls:
        return urls[0]

    # Final fallback
    return _inline_svg_cover(name)
