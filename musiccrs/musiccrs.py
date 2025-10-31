"""MusicCRS — original features + paging + reduced Spotify reliance.

Adds: /ask help — lists all supported question types for /ask.
(Also keeps earlier fixes: paging, popularity formatting, single-playlist payload, INFORM acts.)
"""

from __future__ import annotations
import json, os, re
from typing import Optional, List, Tuple
from musiccrs.recommender import recommend_by_cooccurrence
from collections import defaultdict, Counter

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
from .nl_handler import NaturalLanguageHandler, extract_song_artist

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
        self._nl = NaturalLanguageHandler()
        self._user_key = "default"

        self._disambig: Optional[dict] = None  # {'query','rows','page','page_size'}
        self._last_recommendations: Optional[List[dict]] = None  # Store last recommendations for R5.2

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
            
            # R5: Natural language handling (if not a command)
            if not text.startswith('/') and self._nl.is_natural_language(text):
                self._handle_natural_language(text)
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
            
            if text.strip().lower().startswith("/auto"):
                query = text[len("/auto"):].strip()
                self._handle_auto(query)
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
                # Note: add_track_by_artist_title doesn't support defer_cover, so refresh after
                self._send_playlist_text(f"Added <b>{track.artist} – {track.title}</b>.")
                self._ps.force_refresh_cover(self._user_key)
                return

            if m := self._cmd_add_any.match(text):
                query = m.group(1).strip()
                artist = None
                title = None
                
                # Try explicit delimiters first
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
                
                # If no delimiter found, try title-only search first (R5 approach)
                # Only attempt artist/title splitting if title search fails
                if not (artist and title):
                    # First, try as a complete title (most common case)
                    title_rows = self._ps.search_tracks_by_title(self._user_key, query, fetch_all=True, limit=30)
                    if title_rows:
                        # Found results with title-only search - use them!
                        if len(title_rows) == 1:
                            uri, artist, title, album = title_rows[0]
                            self._add_track_and_notify(uri, f"Added <b>{artist} – {title}</b>.")
                            return
                        else:
                            self._start_disambiguation(query, title_rows)
                            return
                    
                    # Title search found nothing - try artist/title splitting as fallback
                    from .ir_search import search_artist_title_ir
                    from .playlist_db import ensure_db
                    words = query.split()
                    if len(words) >= 2:
                        # Try most likely split points first to minimize queries
                        # For 2-3 words: try middle splits
                        # For 4+ words: try 2-word artist first (common: "Kendrick Lamar")
                        
                        if len(words) == 2:
                            # Try both: word1=artist word2=title, and word1=title word2=artist
                            candidates = [
                                (words[0], words[1]),
                                (words[1], words[0])
                            ]
                        elif len(words) == 3:
                            # Try: "word1 word2" + "word3", or "word1" + "word2 word3"
                            candidates = [
                                (" ".join(words[:2]), words[2]),
                                (words[0], " ".join(words[1:])),
                                (" ".join(words[1:]), words[0]),
                                (words[2], " ".join(words[:2]))
                            ]
                        else:
                            # For 4+ words, try multiple splits
                            # Common patterns: "Title Artist" or "Artist Title"
                            candidates = [
                                # Try last word as artist, rest as title (common: "song name artist")
                                (words[-1], " ".join(words[:-1])),
                                # Try first word as artist, rest as title
                                (words[0], " ".join(words[1:])),
                                # Try last 2 words as artist, rest as title (common: "song Artist Name")
                                (" ".join(words[-2:]), " ".join(words[:-2])),
                                # Try first 2 words as artist, rest as title (common: "Artist Name song")
                                (" ".join(words[:2]), " ".join(words[2:])),
                                # Try middle splits
                                (" ".join(words[:len(words)//2]), " ".join(words[len(words)//2:])),
                                (" ".join(words[len(words)//2:]), " ".join(words[:len(words)//2]))
                            ]
                        
                        # Try each candidate
                        conn = ensure_db()
                        try:
                            for potential_artist, potential_title in candidates:
                                fuzzy = search_artist_title_ir(conn, potential_artist, potential_title, limit=5)
                                if fuzzy:
                                    artist, title = potential_artist, potential_title
                                    break
                        finally:
                            conn.close()

                if artist and title:
                    from .playlist_db import get_track, ensure_db
                    from .ir_search import search_artist_title_ir
                    row = get_track(artist, title)
                    if row:
                        uri, a, t, album = row
                        self._add_track_and_notify(uri, f"Added <b>{a} – {t}</b>.")
                        return
                    conn = ensure_db()
                    try:
                        fuzzy = search_artist_title_ir(conn, artist, title, limit=10)
                    finally:
                        conn.close()
                    if fuzzy:
                        if len(fuzzy) == 1:
                            uri, a, t, album = fuzzy[0]
                            self._add_track_and_notify(uri, f"Added <b>{a} – {t}</b>.")
                        else:
                            self._start_disambiguation(f"{artist} - {title}", fuzzy)
                        return
                    # Splitting failed - already tried title-only above, so nothing found
                    self._send_text(
                        f"No tracks found matching <b>{query}</b>.",
                        include_playlist=False,
                    )
                    return
                
                # No artist/title split attempted (single word or splitting disabled)
                # This shouldn't happen since we try title-only first, but just in case
                self._send_text(f"No tracks found matching <b>{query}</b>.", include_playlist=False)
                return

            if m := self._cmd_remove.match(text):
                ident = m.group(1).strip()
                track = self._ps.remove(self._user_key, ident)
                self._send_playlist_text(f"Removed <b>{track.artist} – {track.title}</b>.")
                return

            if self._cmd_view.match(text):
                pl = self._ps.current_playlist(self._user_key)
                if pl.tracks:
                    # Use <ol> which provides automatic numbering
                    lines = "".join([f"<li>{t.artist} – {t.title}</li>" for t in pl.tracks])
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
    
    def _handle_natural_language(self, text: str) -> None:
        """
        Handle natural language input (R5.1-R5.6).
        Classifies intent and dispatches to appropriate handler.
        """
        intent = self._nl.classify_intent(text)
        
        if intent.confidence < 0.3:
            # Low confidence: treat as general query or ask
            self._send_text(
                "I'm not sure what you want. Try <code>/help</code> for commands, or ask a specific question.",
                include_playlist=False
            )
            return
        
        # Dispatch based on intent
        if intent.intent_type == 'add_track':
            self._handle_nl_add_track(intent)
        elif intent.intent_type == 'remove_track':
            self._handle_nl_remove_track(intent)
        elif intent.intent_type == 'view_playlist':
            self._handle_nl_view_playlist(intent)
        elif intent.intent_type == 'clear_playlist':
            self._handle_nl_clear_playlist(intent)
        elif intent.intent_type == 'recommend':
            self._handle_nl_recommend(intent)
        elif intent.intent_type == 'ask_question':
            self._handle_nl_ask(intent)
        else:
            self._send_text(
                f"I understood you want to '{intent.intent_type}', but I'm not sure how to help. Try <code>/help</code>.",
                include_playlist=False
            )
    
    def _handle_nl_add_track(self, intent) -> None:
        """Handle natural language track addition (R5.1, R5.3, R5.6)."""
        entities = intent.entities
        
        # Check if we're responding to a recommendation (R5.2)
        if self._last_recommendations:
            selection = self._nl.parse_selection(intent.raw_text)
            self._handle_nl_recommendation_selection(selection)
            return
        
        # Extract title/artist from entities
        if 'title' in entities and 'artist' in entities:
            # Clear pattern: "add humble by kendrick"
            from .playlist_db import ensure_db
            from .ir_search import search_artist_title_ir
            conn = ensure_db()
            try:
                fuzzy = search_artist_title_ir(conn, entities['artist'], entities['title'], limit=10)
            finally:
                conn.close()
            if fuzzy:
                if len(fuzzy) == 1:
                    uri, artist, title, album = fuzzy[0]
                    self._add_track_and_notify(uri, f"Added <b>{artist} – {title}</b>.")
                else:
                    self._start_disambiguation(f"{entities['artist']} - {entities['title']}", fuzzy)
            else:
                self._send_text(
                    f"No tracks found matching <b>{entities['artist']} - {entities['title']}</b>.",
                    include_playlist=False
                )
        elif 'query' in entities:
            # General query: "add one dance"
            query = entities['query']
            full_rows = self._ps.search_tracks_by_title(self._user_key, query, fetch_all=True)
            if not full_rows:
                self._send_text(f"No tracks found matching <b>{query}</b>.", include_playlist=False)
            elif len(full_rows) == 1:
                uri, artist, title, album = full_rows[0]
                self._add_track_and_notify(uri, f"Added <b>{artist} – {title}</b>.")
            else:
                self._start_disambiguation(query, full_rows)
        else:
            self._send_text("I couldn't understand which track to add. Try: 'add [song] by [artist]'", include_playlist=False)
    
    def _handle_nl_remove_track(self, intent) -> None:
        """Handle natural language track removal (R5.1)."""
        entities = intent.entities
        
        if 'track_number' in entities:
            try:
                track = self._ps.remove(self._user_key, str(entities['track_number']))
                self._send_playlist_text(f"Removed <b>{track.artist} – {track.title}</b>.")
            except Exception as e:
                self._send_text(f"Error: {str(e)}", include_playlist=False)
        elif 'query' in entities:
            # Try to find track by name
            query = entities['query']
            pl = self._ps.current_playlist(self._user_key)
            # Find matching track in playlist
            for i, track in enumerate(pl.tracks, 1):
                if query.lower() in track.title.lower() or query.lower() in track.artist.lower():
                    removed = self._ps.remove(self._user_key, str(i))
                    self._send_playlist_text(f"Removed <b>{removed.artist} – {removed.title}</b>.")
                    return
            self._send_text(f"No track matching <b>{query}</b> found in playlist.", include_playlist=False)
        else:
            self._send_text("I couldn't understand which track to remove. Try: 'remove track 3'", include_playlist=False)
    
    def _handle_nl_view_playlist(self, intent) -> None:
        """Handle natural language playlist viewing (R5.1)."""
        pl = self._ps.current_playlist(self._user_key)
        if pl.tracks:
            # Use <ol> which provides automatic numbering, no need for manual {i+1}
            lines = "".join([f"<li>{t.artist} – {t.title}</li>" for t in pl.tracks])
            html = f"Here's your playlist <b>{pl.name}</b> ({len(pl.tracks)} tracks):<ol>{lines}</ol>"
        else:
            html = f"Your playlist <b>{pl.name}</b> is empty."
        self._send_playlist_text(html)
    
    def _handle_nl_clear_playlist(self, intent) -> None:
        """Handle natural language playlist clearing (R5.1)."""
        self._ps.clear(self._user_key)
        self._send_playlist_text("Cleared your playlist.")
    
    def _handle_nl_recommend(self, intent) -> None:
        """Handle natural language recommendation requests (R5.5)."""
        entities = intent.entities
        count = entities.get('count', 5)
        count = max(1, min(10, count))  # Clamp to 1-10
        
        self._handle_recommend(limit=count)
    
    def _handle_nl_ask(self, intent) -> None:
        """Handle natural language questions (R5.4)."""
        # Use the existing /ask functionality
        query = intent.raw_text
        try:
            result = self._qa.answer_question(query)
            if result:
                self._send_text(result, include_playlist=False)
            else:
                self._send_text("I don't have an answer for that question.", include_playlist=False)
        except Exception as e:
            self._send_text(f"Error: {str(e)}", include_playlist=False)
    
    def _handle_nl_recommendation_selection(self, selection: dict) -> None:
        """Handle natural language selection from recommendations (R5.2)."""
        if not self._last_recommendations:
            self._send_text("No recommendations to select from. Use 'recommend' first.", include_playlist=False)
            return
        
        tracks_to_add = []
        
        if selection['type'] == 'all':
            tracks_to_add = self._last_recommendations
        elif selection['type'] == 'range':
            start, end = selection['start'], selection['end']
            tracks_to_add = self._last_recommendations[start:end]
        elif selection['type'] == 'index':
            idx = selection['index']
            if 0 <= idx < len(self._last_recommendations):
                tracks_to_add = [self._last_recommendations[idx]]
        elif selection['type'] == 'exclude_artist':
            artist_to_exclude = selection['artist'].lower()
            tracks_to_add = [
                t for t in self._last_recommendations
                if artist_to_exclude not in t['artist'].lower()
            ]
        
        if not tracks_to_add:
            self._send_text("No tracks match that selection.", include_playlist=False)
            return
        
        # Add tracks
        added_count = 0
        for track_data in tracks_to_add:
            try:
                self._ps.add_by_uri(self._user_key, track_data['track_uri'], defer_cover=True)
                added_count += 1
            except Exception:
                pass
        
        # Refresh cover once at the end
        if added_count > 0:
            self._ps.force_refresh_cover(self._user_key)
            self._send_playlist_text(f"Added {added_count} track(s) to your playlist.")
        else:
            self._send_text("No tracks were added.", include_playlist=False)
        
        # Clear last recommendations
        self._last_recommendations = None

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
    
    def _add_track_and_notify(self, track_uri: str, message: str) -> None:
        """Add track, send notification, then refresh cover asynchronously.
        
        This ensures the user gets immediate feedback while cover generation
        happens in the background without blocking the response.
        """
        # Add track without cover refresh (defer it)
        track = self._ps.add_by_uri(self._user_key, track_uri, defer_cover=True)
        
        # Send immediate response
        self._send_playlist_text(message)
        
        # Refresh cover after response is sent (non-blocking for user experience)
        self._ps.force_refresh_cover(self._user_key)

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
        self._disambig = None
        self._add_track_and_notify(uri, f"Added <b>{artist} – {title}</b>.")

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
        
        OPTIMIZED: 
        - Reuses single database connection for all searches
        - Uses exact match first, fallback to LIKE if needed
        - Defers cover generation until all tracks are added
        """
        from html import escape as _e
        import re
        from .playlist_db import ensure_db

        payload = (payload or "").strip()
        if not payload:
            self._send_text("Nothing selected to add.")
            return

        # Split on newlines OR '||'
        parts = [p.strip() for p in re.split(r"(?:\n|\|\|)", payload) if p.strip()]
        added = []
        errors = []

        # OPTIMIZATION: Use single database connection for all searches
        conn = ensure_db()
        try:
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
                    # OPTIMIZATION: Try exact match first (uses index efficiently)
                    rows = conn.execute(
                        """
                        SELECT track_uri, artist, title, album
                        FROM tracks
                        WHERE LOWER(artist) = ? AND LOWER(title) = ?
                        LIMIT 1
                        """,
                        (artist.lower(), title.lower()),
                    ).fetchall()
                    
                    # Fallback to LIKE if exact match fails
                    if not rows:
                        rows = conn.execute(
                            """
                            SELECT track_uri, artist, title, album
                            FROM tracks
                            WHERE LOWER(artist) LIKE ? AND LOWER(title) LIKE ?
                            LIMIT 1
                            """,
                            (f"%{artist.lower()}%", f"%{title.lower()}%"),
                        ).fetchall()
                    
                    if rows:
                        uri, a, t, album = rows[0]
                        # Use internal method with defer_cover=True
                        track = self._ps._add_by_uri_internal(self._user_key, uri, a, t, album, defer_cover=True)
                        added.append(f"{_e(track.artist)} – {_e(track.title)}")
                    else:
                        errors.append(p)
                except Exception:
                    errors.append(p)
        finally:
            conn.close()

        # OPTIMIZATION: Generate cover ONCE after all tracks are added
        if added:
            self._ps.force_refresh_cover(self._user_key)
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
        
        # Store for R5.2 (natural language selection)
        self._last_recommendations = recs

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
        


    def _search_mpd_playlists(self, query: str, limit: int = 12):
        """
        Search playlist titles via the small index in ./data/mpd_titles_fts.sqlite.
        Returns: [{'id': pid, 'name': str, 'ntracks': int|None}, ...]
        """
        from . import title_index
        # not building here; assume you've run tools/build_title_index.py already
        return title_index.search_titles(query, limit=limit)


    def _get_mpd_playlist_tracks(self, playlist_id: int):
        """
        Fast track fetch by playlist id from ./data/mpd.sqlite.
        Schema expected from tools/build_mpd_sqlite.py:
        - playlists(pid PRIMARY KEY, name, num_tracks, ...)
        - tracks(track_uri PRIMARY KEY, artist, title, album, ...)
        - playlist_tracks(pid, track_uri)
        Returns list of (uri, artist, title).
        """
        import sqlite3, os
        db_path = os.path.join(os.path.dirname(__file__), "..", "data", "mpd.sqlite")
        db_path = os.path.abspath(db_path)
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(
                """
                SELECT pt.track_uri AS uri,
                    t.artist     AS artist,
                    t.title      AS title
                FROM playlist_tracks AS pt
                JOIN tracks AS t ON t.track_uri = pt.track_uri
                WHERE pt.pid = ?
                """,
                (int(playlist_id),),
            )
            out = []
            for r in cur.fetchall():
                uri = r["uri"]; artist = r["artist"]; title = r["title"]
                if uri and artist and title:
                    out.append((uri, artist, title))
            return out
        finally:
            con.close()


    def _handle_auto(self, query: str) -> None:
        """
        /auto <natural language>
        - Title-only search (BM25/LIKE) over playlist names via ./data/mpd_titles_fts.sqlite
        - Aggregate tracks from those matched playlist IDs via ./data/mpd.sqlite
        - Rank by frequency (decaying weight by rank), enforce <=2 tracks per artist
        """
        from html import escape as _e
        from collections import defaultdict
        import re

        q = (query or "").strip()
        if not q:
            self._send_text("Usage: <code>/auto your vibe here</code>")
            return

        matched = self._search_mpd_playlists(q, limit=12)
        if not matched:
            self._send_text(f"I couldn't find any playlists for: <b>{_e(q)}</b>.")
            return

        # keep only entries with an id
        mpd_with_id = [m for m in matched if m.get("id") is not None]
        if not mpd_with_id:
            self._send_text(f"I couldn't resolve matching playlists for: <b>{_e(q)}</b>.")
            return

        mpd_names = [m["name"] for m in mpd_with_id if m.get("name")]

        # target length: median num_tracks when available; else based on wordiness
        sizes = [m.get("ntracks") for m in mpd_with_id if isinstance(m.get("ntracks"), int)]
        sizes = [s for s in sizes if 5 <= s <= 200]
        if sizes:
            sizes.sort()
            target_len = max(18, min(50, sizes[len(sizes)//2]))
        else:
            words = [w for w in re.findall(r"[A-Za-z0-9']+", q)]
            target_len = max(20, min(40, 22 + max(0, len(words) - 2)))

        # derive a name from the matched titles (no hard-coded vocab)
        def _auto_name_from_titles(qtext: str, names: list[str]) -> str:
            import re
            from collections import Counter
            toks = []
            stop = {"a","an","the","and","or","for","of","to","in","on","at","by","with",
                    "my","your","our","playlist","mix","music","songs","hits"}
            for nm in names[:15]:
                toks += [w.lower() for w in re.findall(r"[A-Za-z0-9']+", nm)]
            kept = [w for w in toks if w not in stop and len(w) >= 3]
            if kept:
                common = [w for w,_ in Counter(kept).most_common(4)]
                title = " ".join(w.capitalize() for w in common[:3]).strip()
                if title:
                    return title if any(k in title.lower() for k in ("vibes","mood","energy","focus","party","chill")) else f"{title} Vibes"
            qtoks = [w for w in re.findall(r"[A-Za-z0-9']+", qtext)]
            return (" ".join(qtoks[:3]).title()) if qtoks else "Personal Mix"

        name = _auto_name_from_titles(q, mpd_names)

        # aggregate tracks with decaying weight by rank
        uri_score = defaultdict(float)
        uri_meta  = {}
        for rank, pl in enumerate(mpd_with_id, start=1):
            w = max(0.40, 1.0 - 0.07*(rank-1))  # 1.00, 0.93, 0.86, ...
            tracks = self._get_mpd_playlist_tracks(pl["id"])
            for uri, artist, title in tracks:
                if not uri or not artist or not title:
                    continue
                uri_score[uri] += w
                if uri not in uri_meta:
                    uri_meta[uri] = (artist, title)

        if not uri_score:
            self._send_text(f"I couldn't assemble a playlist from: <b>{_e(q)}</b>.")
            return

        ranked = sorted(uri_score.items(), key=lambda kv: kv[1], reverse=True)

        # create & switch to new playlist using your existing playlist_service
        created_ok = False
        for create_call in (
            lambda: getattr(self._ps, "new_playlist")(self._user_key, name),
            lambda: getattr(self._ps, "create_playlist")(self._user_key, name),
            lambda: getattr(self._ps, "new")(self._user_key, name),
        ):
            try:
                create_call()
                created_ok = True
                break
            except Exception:
                continue
        if not created_ok:
            self._send_text("Sorry, I couldn't create a new playlist in this environment.")
            return

        # OPTIMIZATION: Add tracks with deferred cover generation
        add_by_uri = self._ps.add_by_uri
        artist_counts = {}
        added = 0

        def artist_ok(a: str) -> bool:
            return artist_counts.get(a, 0) < 2

        for uri, _s in ranked:
            a, t = uri_meta.get(uri, ("", ""))
            if not a or not t or not artist_ok(a):
                continue
            try:
                # CRITICAL: defer_cover=True to skip expensive HTTP requests during bulk add
                add_by_uri(self._user_key, uri, defer_cover=True)
                artist_counts[a] = artist_counts.get(a, 0) + 1
                added += 1
            except Exception:
                continue
            if added >= target_len:
                break

        if added == 0:
            self._send_text(f"I couldn't assemble a playlist from: <b>{_e(q)}</b>.")
            return

        # OPTIMIZATION: Generate cover ONCE after all tracks are added
        self._ps.force_refresh_cover(self._user_key)

        blurb = ""
        if mpd_names:
            show = ", ".join(_e(n) for n in mpd_names[:3])
            blurb = f"<br/><span class='text-muted'>Inspired by MPD playlists like: {show}</span>"

        self._send_playlist_text(
            f"Created <b>{_e(name)}</b> — {added} songs, from playlists matching \"{_e(q)}\".{blurb}"
        )



    
# runner
if __name__ == "__main__":
    platform = FlaskSocketPlatform(MusicCRS)
    platform.start()
