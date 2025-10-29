"""MusicCRS — original features + paging + reduced Spotify reliance.

Adds: /ask help — lists all supported question types for /ask.
(Also keeps earlier fixes: paging, popularity formatting, single-playlist payload, INFORM acts.)
"""

from __future__ import annotations
import json, os, re
from typing import Optional, List, Tuple
from musiccrs.recommender import recommend_by_cooccurrence

try:
    import ollama
except Exception:
    ollama = None

from dotenv import load_dotenv
from dialoguekit.core.annotated_utterance import AnnotatedUtterance
from dialoguekit.core.dialogue_act import DialogueAct
from dialoguekit.core.intent import Intent
from dialoguekit.core.slot_value_annotation import SlotValueAnnotation
from dialoguekit.participant.participant import DialogueParticipant
from dialoguekit.platforms import FlaskSocketPlatform
from dialoguekit.participant.agent import Agent

from .playlist_service import PlaylistService
from .qa_system import QASystem

load_dotenv()
OLLAMA_HOST = os.getenv("OLLAMA_HOST")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

_INTENT_OPTIONS = Intent("OPTIONS")
_INTENT_INFORM = Intent("INFORM")

def _playlist_marker_from_obj(playlist_obj: dict | None) -> str:
    if not playlist_obj:
        return ""
    return f"<!--PLAYLIST:{json.dumps(playlist_obj, separators=(',', ':'))}-->"

def _format_pop_k(n: int) -> str:
    v = (n or 0) / 1000.0
    s = f"{v:.1f}"
    if s.endswith(".0"):
        s = s[:-2]
    return f"{s}k"

_NUMBER_ONLY = re.compile(r"^\s*(\d+)\s*[\.)]?\s*$")

class MusicCRS(Agent):
    def __init__(self, use_llm: bool = True):
        super().__init__(id="MusicCRS")
        if use_llm and OLLAMA_HOST and OLLAMA_MODEL:
            self._llm = ollama.Client(
                host=OLLAMA_HOST,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None,
            )
        else:
            self._llm = None

        self._ps = PlaylistService()
        self._qa = QASystem()
        self._user_key = "default"

        self._disambig: Optional[dict] = None  # {'query','rows','page','page_size'}

        # commands
        self._cmd_add_exact = re.compile(r"^/add\s+([^:]+)\s*:\s*(.+)$", re.IGNORECASE)
        self._cmd_add_any   = re.compile(r"^/add\s+(.+)$", re.IGNORECASE)
        self._cmd_remove    = re.compile(r"^/remove\s+(.+)$", re.IGNORECASE)
        self._cmd_view      = re.compile(r"^/view$", re.IGNORECASE)
        self._cmd_clear     = re.compile(r"^/clear$", re.IGNORECASE)
        self._cmd_create    = re.compile(r"^/create\s+(.+)$", re.IGNORECASE)
        self._cmd_switch    = re.compile(r"^/switch\s+(.+)$", re.IGNORECASE)
        self._cmd_list      = re.compile(r"^/list$", re.IGNORECASE)
        self._cmd_help      = re.compile(r"^/help$|^/options$", re.IGNORECASE)
        self._cmd_ask       = re.compile(r"^/ask\s+(.+)$", re.IGNORECASE)
        self._cmd_ask_help  = re.compile(r"^/ask\s+help\s*$", re.IGNORECASE)  # NEW
        self._cmd_stats     = re.compile(r"^/stats$", re.IGNORECASE)
        self._cmd_play      = re.compile(r"^/play(?:\s+(\d+))?$", re.IGNORECASE)
        self._cmd_preview   = re.compile(r"^/preview\s+(.+)$", re.IGNORECASE)
        self._cmd_recommend = re.compile(r"^/recommend(?:\s+(\d+))?$", re.IGNORECASE)


        # paging
        self._cmd_next      = re.compile(r"^/(?:next|more)$", re.IGNORECASE)
        self._cmd_prev      = re.compile(r"^/(?:prev|previous|back)$", re.IGNORECASE)
        self._cmd_page      = re.compile(r"^/page\s+(\d+)$", re.IGNORECASE)

    # lifecycle
    def welcome(self) -> None:
        self._send_text("Hello, I'm MusicCRS. Type /help to see what I can do.")

    def goodbye(self) -> None:
        self._send_text("It was nice talking to you. Bye")

    # dispatcher
    def receive_utterance(self, utterance) -> None:
        text = (utterance.text or "").strip()
        if not text:
            return
        try:
            # numeric choice during disambiguation
            if self._disambig:
                mnum = _NUMBER_ONLY.match(text)
                if mnum:
                    self._handle_disambig_choice(int(mnum.group(1)))
                    return

            if text.startswith("/info"):
                self._send_text(self._info(), include_playlist=False)
                return
            if text.startswith("/ask_llm "):
                prompt = text[9:]
                self._send_text(self._ask_llm(prompt), include_playlist=False)
                return
            if text.startswith("/options"):
                options = ["Play some jazz music", "Recommend me some pop songs", "Create a workout playlist"]
                self._send_options(options)
                return
            if text == "/quit":
                self.goodbye()
                return

            if text.strip().lower().startswith("/bulkadd"):
                payload = text[len("/bulkadd"):].strip()
                self._handle_bulkadd(payload)
                return

            
            # paging
            if self._cmd_next.match(text):
                self._paginate(+1)
                return
            if self._cmd_prev.match(text):
                self._paginate(-1)
                return
            if m := self._cmd_page.match(text):
                page = max(1, int(m.group(1)))
                self._paginate(0, set_to=page - 1)
                return

            # playback/preview
            if m := self._cmd_play.match(text):
                track_num = m.group(1)
                self._handle_play(int(track_num) if track_num else None)
                return
            if m := self._cmd_preview.match(text):
                query = m.group(1).strip()
                self._handle_preview_search(query)
                return

            # --- ASK HELP (new) ---
            if self._cmd_ask_help.match(text):
                self._send_text(self._qa.help_text(), include_playlist=False)
                return

            if self._cmd_stats.match(text):
                stats = self._ps.get_playlist_stats(self._user_key)
                html = self._format_stats(stats)
                self._send_playlist_text(html)
                return

            if m := self._cmd_ask.match(text):
                q = m.group(1).strip()
                ans = self._qa.answer_question(q)
                self._send_text(ans, include_playlist=False)
                return
            
            if m := self._cmd_recommend.match(text or ""):
                k = int(m.group(1) or 5)
                try:
                    self._handle_recommend(limit=k)
                except Exception as e:
                    self._send_text(f"Sorry — recommendation failed: {e}")
                return

            # playlist ops
            if m := self._cmd_add_exact.match(text):
                artist = m.group(1).strip()
                title = m.group(2).strip()
                track = self._ps.add_track_by_artist_title(self._user_key, artist, title)
                self._send_playlist_text(f"Added <b>{track.artist} – {track.title}</b>.")
                return

            if m := self._cmd_add_any.match(text):
                query = m.group(1).strip()
                artist = None
                title = None
                if ":" in query:
                    parts = query.split(":", 1)
                    if parts[0].strip() and parts[1].strip():
                        artist, title = parts[0].strip(), parts[1].strip()
                if not (artist and title) and " by " in query.lower():
                    idx = query.lower().index(" by ")
                    title, artist = query[:idx].strip(), query[idx + 4 :].strip()
                if not (artist and title) and " - " in query:
                    parts = query.split(" - ", 1)
                    if len(parts) == 2:
                        artist, title = parts[0].strip(), parts[1].strip()

                if artist and title:
                    from .playlist_db import get_track, search_by_artist_title_fuzzy
                    row = get_track(artist, title)
                    if row:
                        track = self._ps.add_by_uri(self._user_key, row[0])
                        self._send_playlist_text(f"Added <b>{track.artist} – {track.title}</b>.")
                        return
                    fuzzy = search_by_artist_title_fuzzy(artist, title, limit=10)
                    if fuzzy:
                        if len(fuzzy) == 1:
                            track = self._ps.add_by_uri(self._user_key, fuzzy[0][0])
                            self._send_playlist_text(f"Added <b>{track.artist} – {track.title}</b>.")
                        else:
                            self._start_disambiguation(f"{artist} - {title}", fuzzy)
                        return
                    self._send_text(
                        f"No tracks found with artist <b>{artist}</b> and title <b>{title}</b>.",
                        include_playlist=False,
                    )
                    return

                title = query
                full_rows = self._ps.search_tracks_by_title(self._user_key, title, fetch_all=True)
                if not full_rows:
                    self._send_text(f"No tracks found with title <b>{title}</b>.", include_playlist=False)
                elif len(full_rows) == 1:
                    track = self._ps.add_by_uri(self._user_key, full_rows[0][0])
                    self._send_playlist_text(f"Added <b>{track.artist} – {track.title}</b>.")
                else:
                    self._start_disambiguation(title, full_rows)
                return

            if m := self._cmd_remove.match(text):
                ident = m.group(1).strip()
                track = self._ps.remove(self._user_key, ident)
                self._send_playlist_text(f"Removed <b>{track.artist} – {track.title}</b>.")
                return

            if self._cmd_view.match(text):
                pl = self._ps.current_playlist(self._user_key)
                if pl.tracks:
                    lines = "".join([f"<li>{i+1}. {t.artist} – {t.title}</li>" for i, t in enumerate(pl.tracks)])
                    html = f"Playlist <b>{pl.name}</b> ({len(pl.tracks)} tracks):<ol>{lines}</ol>"
                else:
                    html = f"Playlist <b>{pl.name}</b> is empty."
                self._send_playlist_text(html)
                return

            if self._cmd_clear.match(text):
                self._ps.clear(self._user_key)
                self._send_playlist_text("Cleared the current playlist.")
                return

            if m := self._cmd_create.match(text):
                name = m.group(1).strip()
                self._ps.create_playlist(self._user_key, name)
                self._send_playlist_text(f"Created and switched to playlist <b>{name}</b>.")
                return

            if m := self._cmd_switch.match(text):
                name = m.group(1).strip()
                self._ps.switch_playlist(self._user_key, name)
                self._send_playlist_text(f"Switched to playlist <b>{name}</b>.")
                return

            if self._cmd_list.match(text):
                names = self._ps.list_playlists(self._user_key)
                html = "Your playlists: " + ", ".join(f"<b>{n}</b>" for n in names)
                self._send_playlist_text(html)
                return

            if self._cmd_help.match(text):
                self._send_playlist_text(self._help_text())
                return

            self._send_text("I'm sorry, I don't understand that command. Type /help.", include_playlist=False)

        except Exception as e:
            self._send_text(f"Error: {str(e)}", include_playlist=False)

    # helpers
    def _info(self) -> str:
        return "I am MusicCRS, a conversational recommender system for music."

    def _ask_llm(self, prompt: str) -> str:
        if not self._llm:
            return "The agent is not configured to use an LLM"
        llm_response = self._llm.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={"stream": False, "temperature": 0.7, "max_tokens": 100},
        )
        try:
            return llm_response["response"]
        except Exception:
            return str(llm_response)

    def _send_options(self, options: List[str]) -> None:
        dialogue_acts = [
            DialogueAct(
                intent=_INTENT_OPTIONS,
                annotations=[SlotValueAnnotation("option", o) for o in options],
            )
        ]
        self._send_text("Here are some options:", include_playlist=False, dialogue_acts=dialogue_acts)

    def _send_text(
        self,
        text_html: str,
        *,
        include_playlist: bool = True,
        dialogue_acts: Optional[list] = None
    ) -> None:
        playlist_obj = self._ps.serialize_current_playlist(self._user_key) if include_playlist else None
        text = text_html + _playlist_marker_from_obj(playlist_obj)
        self._dialogue_connector.register_agent_utterance(
            AnnotatedUtterance(
                text,
                participant=DialogueParticipant.AGENT,
                dialogue_acts=dialogue_acts or [],
            )
        )

    def _send_playlist_text(self, text_html: str) -> None:
        acts = [DialogueAct(intent=_INTENT_INFORM)]
        self._send_text(text_html, include_playlist=True, dialogue_acts=acts)

    def _help_text(self) -> str:
        return (
            "I can manage playlists for you. Use these commands:<br/>"
            "<ul>"
            "<li><b>Adding songs</b><ul>"
            "<li><code>/add [title]</code> — (use <code>/next</code>, <code>/previous</code>, <code>/page N</code> to browse)</li>"
            "<li><code>/add [title] by [artist]</code></li>"
            "<li><code>/add [artist] - [title]</code></li>"
            "<li><code>/add [artist]: [title]</code></li>"
            "</ul></li>"
            "<li><b>Managing playlists</b><ul>"
            "<li><code>/remove [index|uri]</code></li>"
            "<li><code>/view</code>, <code>/clear</code>, <code>/create [name]</code>, <code>/switch [name]</code>, <code>/list</code></li>"
            "</ul></li>"
            "<li><b>Playback & info</b><ul>"
            "<li><code>/play [number]</code> (Spotify embed if URI is a Spotify track)</li>"
            "<li><code>/preview Artist/Title</code></li>"
            "<li><code>/stats</code></li>"
            "</ul></li>"
            "<li><b>Questions & search</b><ul>"
            "<li><code>/ask [question]</code> (DB-based)</li>"
            "<li><code>/ask help</code> — show the supported question types</li>"
            "</ul></li>"
            "<li><b>Paging controls</b> — <code>/next</code>, <code>/previous</code>, <code>/page N</code></li>"
            "</ul>"
        )

    # ---- disambiguation with paging ----
    def _start_disambiguation(self, title: str, rows: List[Tuple[str, str, str, Optional[str]]]) -> None:
        self._disambig = {"query": title, "rows": rows, "page": 0, "page_size": 10}
        self._render_disambig_page()

    def _paginate(self, delta: int, set_to: Optional[int] = None) -> None:
        if not self._disambig:
            self._send_text("No active selection. Use <code>/add &lt;title&gt;</code> first.", include_playlist=False)
            return
        total = len(self._disambig["rows"])
        page_size = self._disambig["page_size"]
        pages = max(1, (total + page_size - 1) // page_size)
        newp = set_to if set_to is not None else self._disambig["page"] + delta
        if newp < 0 or newp >= pages:
            self._send_text("No more pages.", include_playlist=False)
            return
        self._disambig["page"] = newp
        self._render_disambig_page()

    def _render_disambig_page(self) -> None:
        info = self._disambig
        if not info:
            return
        rows = info["rows"]
        page = info["page"]
        page_size = info["page_size"]
        query = info["query"]
        total = len(rows)
        start = page * page_size
        end = min(start + page_size, total)

        banner = (
            f"I found <b>{total}</b> tracks with the title <b>{query}</b>. "
            f"Showing <b>{start+1}–{end}</b> of <b>{total}</b>.<br/>"
            f"Type the number to add that track (absolute, e.g. <code>12</code>), or <code>/next</code>, <code>/previous</code> "
            f"(you can also <code>/page N</code>)."
        )

        html = banner + f"<br/><ol start='{start+1}'>"
        for (uri, artist, title, album) in rows[start:end]:
            pop = self._ps.get_popularity(uri)
            album_label = album or "single"
            html += f"<li><b>{artist}</b> – {title} (from <i>{album_label}</i> • {_format_pop_k(pop)})</li>"
        html += "</ol>"

        acts = [
            DialogueAct(intent=Intent("INFORM"), annotations=[SlotValueAnnotation("option", "/previous")]),
            DialogueAct(intent=Intent("INFORM"), annotations=[SlotValueAnnotation("option", "/next")]),
        ]
        self._send_text(html, include_playlist=False, dialogue_acts=acts)

    def _handle_disambig_choice(self, choice: int) -> None:
        info = self._disambig
        if not info:
            self._send_text("No pending selection.", include_playlist=False)
            return
        rows = info["rows"]
        total = len(rows)
        page = info["page"]
        page_size = info["page_size"]
        start = page * page_size
        end = min(start + page_size, total)

        if 1 <= choice <= total:
            idx0 = choice - 1
        elif 1 <= choice <= (end - start):
            idx0 = start + (choice - 1)
        else:
            self._send_text("Invalid selection number.", include_playlist=False)
            return

        uri, artist, title, album = rows[idx0]
        track = self._ps.add_by_uri(self._user_key, uri)
        self._disambig = None
        self._send_playlist_text(f"Added <b>{track.artist} – {track.title}</b>.")

    # ---- stats rendering ----
    def _format_stats(self, stats: dict) -> str:
        if not stats or stats.get("total_tracks", 0) == 0:
            return "Your current playlist is empty. Add some tracks first!"
        html = f"<h3>📊 Statistics for <b>{stats['playlist_name']}</b></h3>"
        html += "<ul>"
        html += f"<li><b>Total tracks:</b> {stats['total_tracks']}</li>"
        html += f"<li><b>Unique artists:</b> {stats['unique_artists']}</li>"
        html += f"<li><b>Unique albums:</b> {stats['unique_albums']}</li>"
        html += f"<li><b>Average popularity (MPD):</b> {stats['avg_popularity_k']}</li>"
        html += "</ul>"
        if stats.get("top_artists"):
            html += "<b>Top artists:</b><ol>"
            for artist, count in stats["top_artists"]:
                html += f"<li>{artist} <span style='opacity:.7'>(×{count})</span></li>"
            html += "</ol>"
        if stats.get("top_albums"):
            html += "<b>Top albums:</b><ol>"
            for album, count in stats["top_albums"]:
                safe = album or "single"
                html += f"<li><i>{safe}</i> <span style='opacity:.7'>(×{count})</span></li>"
            html += "</ol>"
        return html

    # ---- playback helpers ----
    def _handle_play(self, track_num: int | None) -> None:
        pl = self._ps.current_playlist(self._user_key)
        if not pl.tracks:
            self._send_text("Your playlist is empty. Add some tracks first!", include_playlist=False)
            return

        if track_num is None:
            html = f"<b>{pl.name}</b> tracks:<br/><ol>"
            for i, track in enumerate(pl.tracks, 1):
                html += f"<li>{track.artist} – {track.title}</li>"
            html += "</ol>Use <code>/play [number]</code> to get the Spotify link for a track."
            self._send_text(html, include_playlist=False)
            return

        if track_num < 1 or track_num > len(pl.tracks):
            self._send_text(f"Invalid track number. Please choose between 1 and {len(pl.tracks)}.", include_playlist=False)
            return

        track = pl.tracks[track_num - 1]
        if track.track_uri.startswith("spotify:track:"):
            track_id = track.track_uri.split(":")[-1]
            spotify_url = f"https://open.spotify.com/track/{track_id}"
            html = f"🎵 <b>{track.artist} – {track.title}</b><br/>"
            html += f"<a href='{spotify_url}' target='_blank'>▶️ Play on Spotify</a><br/>"
            html += (
                f"<br/><iframe style='border-radius:12px' "
                f"src='https://open.spotify.com/embed/track/{track_id}' "
                f"width='100%' height='152' frameBorder='0' allowfullscreen='' "
                f"allow='autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture' "
                f"loading='lazy'></iframe>"
            )
            self._send_text(html, include_playlist=False)
        else:
            self._send_text(f"I don't have a Spotify link for <b>{track.artist} – {track.title}</b>.", include_playlist=False)

    def _handle_preview_search(self, query: str) -> None:
        from .spotify_api import get_spotify_api
        sp = get_spotify_api()
        if not sp:
            self._send_text("Preview lookup is disabled (Spotify credentials not set).", include_playlist=False)
            return

        artist, title = None, None
        if "/" in query:
            parts = [p.strip() for p in query.split("/", 1)]
            if len(parts) == 2:
                artist, title = parts
        elif " - " in query or ":" in query:
            parts = [p.strip() for p in query.replace(":", " - ").split(" - ", 1)]
            if len(parts) == 2:
                artist, title = parts
        else:
            title = query

        if not title:
            self._send_text("Use <code>/preview Artist/Title</code> or <code>/preview Artist - Title</code>.", include_playlist=False)
            return

        preview = sp.find_preview_url(artist or "", title)
        if preview:
            label = f"{artist} – {title}" if artist else title
            self._send_text(f"<a href='{preview}' target='_blank'>Preview: {label}</a>", include_playlist=False)
        else:
            self._send_text("No preview found.", include_playlist=False)

    def _handle_bulkadd(self, payload: str) -> None:
        """
        Add multiple tracks in one go. Payload is a newline- or '||'-separated list
        of 'Artist : Title' entries (no quotes).
        """
        from html import escape as _e
        import re

        payload = (payload or "").strip()
        if not payload:
            self._send_text("Nothing selected to add.")
            return

        # Split on newlines OR '||'
        parts = [p.strip() for p in re.split(r"(?:\n|\|\|)", payload) if p.strip()]
        added = []
        errors = []

        for p in parts:
            if ":" not in p:
                errors.append(p)
                continue
            artist, title = p.split(":", 1)
            artist = artist.strip()
            title = title.strip()
            if not artist or not title:
                errors.append(p)
                continue
            try:
                track = self._ps.add_track_by_artist_title(self._user_key, artist, title)
                added.append(f"{_e(track.artist)} – {_e(track.title)}")
            except Exception:
                errors.append(p)

        if added:
            self._send_playlist_text("Added: " + ", ".join(f"<b>{s}</b>" for s in added))
        if errors:
            self._send_text("Couldn't add: " + ", ".join(_e(e) for e in errors))


    def _handle_recommend(self, *, limit: int = 3) -> None:
        """
        Recommend tracks related to the current playlist and render them
        with:
        - checkboxes + green "Add to playlist" button (unchanged)
        - a friendly one-liner reason (humanized)
        - an expandable detailed explanation that describes how/why it was chosen
        """
        from html import escape as _e
        import re as _re

        # ---------- helpers (local-only) ----------
        def _parse_stats(raw_reason: str, rec: dict):
            """Extract co-occurrence strength and playlist-count from either the reason text or fields in `rec`."""
            raw = raw_reason or ""

            # try parsing numbers out of the existing reason string
            m_co = _re.search(r'(?:co[- ]?occurrence|cooccurrence)\s+(\d+)', raw, _re.I)
            m_pl = (_re.search(r'(\d+)\s+(?:highly-)?matching\s+playlists', raw, _re.I)
                    or _re.search(r'across\s+(\d+)\s+.*playlists', raw, _re.I))

            co_val = None
            pl_val = None

            try:
                if m_co:
                    co_val = int(m_co.group(1))
            except Exception:
                pass
            try:
                if m_pl:
                    pl_val = int(m_pl.group(1))
            except Exception:
                pass

            # fallbacks from structured fields (if present)
            if co_val is None:
                for k in ("cooccurrence", "co_occurrence", "overlap_score", "score"):
                    if k in rec and rec[k] is not None:
                        try:
                            co_val = int(rec[k]) if float(rec[k]).is_integer() else float(rec[k])
                        except Exception:
                            pass
                        break
            if pl_val is None:
                for k in ("playlist_count", "lists", "playlist_hits", "num_playlists"):
                    if k in rec and rec[k] is not None:
                        try:
                            pl_val = int(rec[k])
                        except Exception:
                            pass
                        break

            # qualitative label from strength
            strength = None
            try:
                if co_val is not None:
                    v = float(co_val)
                    if v >= 2000:
                        strength = "very strong"
                    elif v >= 1000:
                        strength = "strong"
                    elif v >= 400:
                        strength = "good"
                    else:
                        strength = "light"
            except Exception:
                pass

            return co_val, pl_val, strength

        def _friendly_reason(raw_reason: str, rec: dict) -> str:
            """Short, human explanation."""
            co_val, pl_val, strength = _parse_stats(raw_reason, rec)
            if pl_val and strength:
                return f"Often added together in {pl_val} playlists like yours ({strength} match)."
            if pl_val:
                return f"Often added together in {pl_val} playlists like yours."
            if strength:
                return f"Often added together with your songs ({strength} match)."
            return raw_reason or "Often added together by listeners with similar taste."

        def _collect_overlap_seeds(rec: dict):
            """
            Pull a few seed tracks/artists this rec overlaps with, if your recommender
            exposed any such fields. We check several common keys and gracefully
            fall back if none exist.
            """
            candidates = []
            for k in (
                "matched_seeds",
                "seed_matches",
                "overlap_seeds",
                "seed_titles",
                "overlap_tracks",
                "seed_examples",
            ):
                v = rec.get(k)
                if isinstance(v, (list, tuple)):
                    candidates = [str(x) for x in v if str(x).strip()]
                    if candidates:
                        break
            # Keep it tidy
            return candidates[:5]

        def _detailed_reason(raw_reason: str, rec: dict) -> str:
            """
            More explicit explanation for curious users.
            Returns small HTML (no nested f-string quotes).
            """
            co_val, pl_val, strength = _parse_stats(raw_reason, rec)
            seeds = _collect_overlap_seeds(rec)
            lines = []

            if pl_val:
                lines.append(f"Found in {_e(str(pl_val))} community playlists similar to yours.")
            if co_val is not None:
                lines.append(f"Frequently appears alongside your picks (co-occurrence score {_e(str(co_val))}).")
            if strength:
                lines.append(f"Overall match strength: {_e(strength.capitalize())}.")

            if seeds:
                # Try to format seeds nicely; if items look like 'Artist – Title' already, use as-is.
                seed_html = ", ".join(_e(s) for s in seeds)
                lines.append(f"Notable overlaps with your current songs: {seed_html}.")

            # If the raw technical reason exists, include it as a footnote for transparency
            if raw_reason and raw_reason.strip():
                lines.append(f"<span class='text-muted'>Data note: {_e(raw_reason)}</span>")

            if not lines:
                return "Recommended because people with playlists like yours often add this song."

            # Build a compact list
            li_html = "".join(f"<li>{ln}</li>" for ln in lines)
            return f"<ul class='mb-0'>{li_html}</ul>"

        # ---------- main body ----------
        # Pull user's current playlist
        pl = self._ps.current_playlist(self._user_key)
        seed_uris = [t.track_uri for t in getattr(pl, "tracks", []) if getattr(t, "track_uri", None)]

        if not seed_uris:
            self._send_text(
                "Your playlist is empty. Add a few songs first with <code>/add Artist : Title</code>."
            )
            return

        # Get recommendations
        recs = recommend_by_cooccurrence(seed_uris, limit=max(3, limit))
        if not recs:
            self._send_text("I couldn't find related tracks right now.")
            return

        items_html_list = []
        for i, r in enumerate(recs[:limit], start=1):
            artist = r.get("artist", "") or ""
            title = r.get("title", "") or ""
            album = r.get("album") or ""
            raw_reason = r.get("reason") or ""

            friendly = _friendly_reason(raw_reason, r)
            detailed_html = _detailed_reason(raw_reason, r)

            album_html = f' <span class="text-muted">({_e(album)})</span>' if album else ""
            reason_html = f'<div class="small text-muted">{_e(friendly)}</div>' if friendly else ""

            # Use native <details> so we don't need any frontend JS changes.
            details_block = (
                "<details class='mt-1'>"
                "<summary class='small text-muted' style='cursor:pointer;'>Why this is recommended</summary>"
                f"<div class='small mt-1'>{detailed_html}</div>"
                "</details>"
            )

            item_html = (
                f"<li class='mb-3'>"
                f"  <div class='form-check'>"
                f"    <input class='form-check-input reco-item' type='checkbox' id='reco-{i}' "
                f"           data-artist='{_e(artist)}' data-title='{_e(title)}' />"
                f"    <label class='form-check-label' for='reco-{i}'>"
                f"      {i}. {_e(artist)} – {_e(title)}{album_html}"
                f"    </label>"
                f"  </div>"
                f"  {reason_html}"
                f"  {details_block}"
                f"</li>"
            )
            items_html_list.append(item_html)

        # Inline styles so this works regardless of your CSS stack
        html = (
            "<div class='recommend-block'>"
            "<p>Here are some related picks:</p>"
            "<ul class='list-unstyled' style='margin:0;padding-left:0'>"
            f"{''.join(items_html_list)}"
            "</ul>"
            "<button type='button' data-action='bulk-add-selected' "
            "        style='margin-top:10px;padding:8px 12px;border:0;border-radius:8px;"
            "               background:#16a34a;color:#fff;font-weight:600;'>"
            "  Add to playlist"
            "</button>"
            "</div>"
        )

        self._send_text(html, include_playlist=False)

    
# runner
if __name__ == "__main__":
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()
