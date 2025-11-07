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
from .autocorrect_integration import AutocorrectIntegration
from .command_corrector import CommandCorrector

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
            headers = {}
            if OLLAMA_API_KEY:
                headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"
            # CRITICAL: Content-Type header is required by the UiS OLLAMA server
            headers["Content-Type"] = "application/json"
            
            self._llm = ollama.Client(
                host=OLLAMA_HOST,
                headers=headers,
            )
        else:
            self._llm = None

        self._ps = PlaylistService()
        self._qa = QASystem()
        
        # Initialize R7 autocorrect with loading callback
        # Note: This callback is called during first initialization only (singleton)
        self._autocorrect = AutocorrectIntegration(
            enabled=True,
            loading_callback=self._send_loading_message
        )
        
        self._command_corrector = CommandCorrector()  # R7.3: OLLAMA command correction
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
        self._cmd_ask_help  = re.compile(r"^/ask\s+help\s*$", re.IGNORECASE)
        self._cmd_stats     = re.compile(r"^/stats$", re.IGNORECASE)
        self._cmd_play      = re.compile(r"^/play(?:\s+(\d+))?$", re.IGNORECASE)
        self._cmd_preview   = re.compile(r"^/preview\s+(.+)$", re.IGNORECASE)
        self._cmd_recommend = re.compile(r"^/recommend(?:\s+(\d+))?$", re.IGNORECASE)


        # paging
        self._cmd_next      = re.compile(r"^/(?:next|more)$", re.IGNORECASE)
        self._cmd_prev      = re.compile(r"^/(?:prev|previous|back)$", re.IGNORECASE)
        self._cmd_page      = re.compile(r"^/page\s+(\d+)$", re.IGNORECASE)
    
    def _format_r7_timing(self, correction) -> str:
        """Format R7 lookup timing with debug info."""
        if not correction:
            return ""
        
        timing_ms = correction.latency_ms
        trigrams = correction.query_trigrams or []
        matched = correction.matched_trigrams or []
        candidates = correction.candidates_found
        
        # Format: [lookup: Xms, n-grams: [abc, bcd, ...], matched: [abc, xyz], candidates: N]
        debug_parts = []
        debug_parts.append(f"lookup: {timing_ms:.0f}ms")
        
        if trigrams:
            trigrams_str = ', '.join(trigrams[:5])  # Show first 5
            if len(trigrams) > 5:
                trigrams_str += f', ... ({len(trigrams)} total)'
            debug_parts.append(f"n-grams: [{trigrams_str}]")
        
        if matched:
            matched_str = ', '.join(matched[:5])  # Show first 5
            if len(matched) > 5:
                matched_str += f', ... ({len(matched)} total)'
            debug_parts.append(f"matched: [{matched_str}]")
        
        if candidates > 0:
            debug_parts.append(f"candidates: {candidates}")
        
        debug_info = ', '.join(debug_parts)
        return f" <span style='opacity:0.6; font-size:0.9em'>[{debug_info}]</span>"
    
    def _format_detailed_timing(self, ollama_time: float = 0, correction=None, db_time: float = 0) -> str:
        """
        Format detailed timing breakdown as an expandable dropdown.
        
        Args:
            ollama_time: Time spent on OLLAMA intent detection (ms)
            correction: CorrectionResult with R7 timing details (optional)
            db_time: Time spent on database search (ms, optional)
            
        Returns:
            HTML with summary timing + expandable detailed breakdown
        """
        # Calculate total time
        total_time = ollama_time
        if correction:
            total_time += correction.latency_ms
        if db_time > 0:
            total_time += db_time
        
        # Build summary
        summary_parts = []
        if ollama_time > 0:
            summary_parts.append(f"Ollama: {ollama_time:.0f}ms")
        if correction:
            summary_parts.append(f"R7: {correction.latency_ms:.0f}ms")
        if db_time > 0:
            summary_parts.append(f"DB: {db_time:.0f}ms")
        
        summary = ', '.join(summary_parts) if summary_parts else f"Total: {total_time:.0f}ms"
        
        # If no detailed timing available, just show summary
        if not correction or not correction.timing_details:
            return f" <span style='opacity:0.6; font-size:0.9em'>[{summary}]</span>"
        
        # Build detailed breakdown
        gen_timings = correction.timing_details.get('gen_timings', {})
        rank_timings = correction.timing_details.get('rank_timings', {})
        
        # Build HTML for expandable section
        details_html = "<ul style='margin:0.5em 0; padding-left:1.5em; font-size:0.9em;'>"
        
        # OLLAMA timing
        if ollama_time > 0:
            details_html += f"<li><b>OLLAMA</b> intent detection: {ollama_time:.0f}ms</li>"
        
        # R7 Candidate Generation breakdown
        if gen_timings:
            details_html += f"<li><b>R7 Candidate Generation</b>: {gen_timings.get('total', 0)*1000:.0f}ms<ul style='margin:0.3em 0;'>"
            if gen_timings.get('alias_map', 0) > 0:
                details_html += f"<li>Alias mapping: {gen_timings['alias_map']*1000:.1f}ms</li>"
            if gen_timings.get('normalize', 0) > 0:
                details_html += f"<li>Normalization: {gen_timings['normalize']*1000:.1f}ms</li>"
            if gen_timings.get('extract_trigrams', 0) > 0:
                details_html += f"<li>Trigram extraction: {gen_timings['extract_trigrams']*1000:.1f}ms</li>"
            if gen_timings.get('fuzzy_expansion', 0) > 0:
                details_html += f"<li>Fuzzy expansion: {gen_timings['fuzzy_expansion']*1000:.1f}ms</li>"
            if gen_timings.get('index_lookup', 0) > 0:
                details_html += f"<li>Inverted index lookup: {gen_timings['index_lookup']*1000:.1f}ms</li>"
            if gen_timings.get('scoring', 0) > 0:
                details_html += f"<li>Candidate scoring: {gen_timings['scoring']*1000:.1f}ms</li>"
            if gen_timings.get('edit_distance_rerank', 0) > 0:
                details_html += f"<li>Edit distance reranking: {gen_timings['edit_distance_rerank']*1000:.1f}ms</li>"
            details_html += "</ul></li>"
        
        # R7 ML Ranking breakdown
        if rank_timings:
            details_html += f"<li><b>R7 ML Ranking</b>: {rank_timings.get('total', 0)*1000:.0f}ms<ul style='margin:0.3em 0;'>"
            if rank_timings.get('feature_extraction', 0) > 0:
                details_html += f"<li>Feature extraction: {rank_timings['feature_extraction']*1000:.1f}ms</li>"
            if rank_timings.get('sort', 0) > 0:
                details_html += f"<li>Sorting: {rank_timings['sort']*1000:.1f}ms</li>"
            details_html += "</ul></li>"
        
        # Database timing
        if db_time > 0:
            details_html += f"<li><b>Database</b> search: {db_time:.0f}ms</li>"
        
        details_html += "</ul>"
        
        # Return summary + expandable details
        return (
            f" <span style='opacity:0.6; font-size:0.9em'>[{summary}]</span>"
            f"<details style='margin-top:0.5em; opacity:0.8;'>"
            f"<summary style='cursor:pointer; font-size:0.9em;'>⏱️ Detailed timing breakdown</summary>"
            f"{details_html}"
            f"</details>"
        )

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
            # Quick greetings check (avoid OLLAMA overhead for simple hi/bye)
            text_lower = text.lower().rstrip('!.?')  # Remove trailing punctuation
            if text_lower in ['hi', 'hello', 'hey', 'hi there', 'hello there']:
                self._send_text("Hello! I'm MusicCRS. Try 'add [song name]' or ask 'help' to see what I can do.", include_playlist=False)
                return
            if text_lower in ['bye', 'goodbye', 'see you', 'exit']:
                self.goodbye()
                return
            
            # Numeric choice during disambiguation
            if self._disambig:
                mnum = _NUMBER_ONLY.match(text)
                if mnum:
                    self._handle_disambig_choice(int(mnum.group(1)))
                    return
            
            # Check for command syntax (commands starting with /) BEFORE OLLAMA
            if text.startswith('/'):
                self._handle_command(text)
                return
            
            # ALL other messages go through OLLAMA for intent detection
            self._handle_with_ollama(text)
            return

        except Exception as e:
            self._send_text(f"Error: {str(e)}", include_playlist=False)

    def _handle_command(self, text: str) -> None:
        """Handle command-based interactions (starting with /)"""
        # /add commands
        m_add_exact = self._cmd_add_exact.match(text)
        if m_add_exact:
            artist = m_add_exact.group(1).strip()
            title = m_add_exact.group(2).strip()
            query = f"{artist} {title}"
            self._handle_add_track_with_timing({"artist": artist, "track_name": title}, 0)
            return
        
        m_add_any = self._cmd_add_any.match(text)
        if m_add_any:
            query = m_add_any.group(1).strip()
            self._handle_add_track_with_timing({"query": query}, 0)
            return
        
        # /list command
        if self._cmd_list.match(text):
            pl = self._ps.current_playlist(self._user_key)
            if not pl.tracks:
                self._send_playlist_text("Your playlist is empty. Add some tracks first!")
            else:
                html = f"<h3>🎵 Playlist: <b>{pl.name}</b></h3><ol>"
                for i, track in enumerate(pl.tracks, 1):
                    html += f"<li><b>{track.artist}</b> – {track.title}"
                    if track.album:
                        html += f" <span style='opacity:0.7'>({track.album})</span>"
                    html += "</li>"
                html += f"</ol><p style='opacity:0.7'>Total: {len(pl.tracks)} track(s)</p>"
                self._send_playlist_text(html)
            return
        
        # /view command
        if self._cmd_view.match(text):
            stats = self._ps.get_playlist_stats(self._user_key)
            html = self._format_stats(stats)
            self._send_playlist_text(html)
            return
        
        # /clear command
        if self._cmd_clear.match(text):
            self._ps.clear_playlist(self._user_key)
            self._send_playlist_text("Playlist cleared.")
            return
        
        # /help command
        if self._cmd_help.match(text):
            self._send_text(self._help(), include_playlist=False)
            return
        
        # /stats command
        if self._cmd_stats.match(text):
            stats = self._ps.get_playlist_stats(self._user_key)
            html = self._format_stats(stats)
            self._send_playlist_text(html)
            return
        
        # /recommend command
        m_recommend = self._cmd_recommend.match(text)
        if m_recommend:
            count = int(m_recommend.group(1)) if m_recommend.group(1) else 5
            self._send_text(f"Generating {count} recommendations...", include_playlist=False)
            try:
                self._handle_recommend(limit=count)
            except Exception as e:
                self._send_text(f"Recommendation failed: {e}", include_playlist=False)
            return
        
        # /remove command
        m_remove = self._cmd_remove.match(text)
        if m_remove:
            target = m_remove.group(1).strip()
            self._handle_remove_track_with_timing({"target": target}, 0)
            return
        
        # /quit or /exit command (not in regex but common)
        if text.lower() in ['/quit', '/exit', '/bye']:
            self.goodbye()
            return
        
        # /ask command
        m_ask = self._cmd_ask.match(text)
        if m_ask:
            query = m_ask.group(1).strip()
            ans = self._qa.answer_question(query)
            self._send_text(ans, include_playlist=False)
            return
        
        # /auto command - generate playlist from natural language description
        if text.lower().startswith('/auto '):
            description = text[6:].strip()  # Remove '/auto ' prefix
            self._handle_auto(description)
            return
        
        # Unknown command
        self._send_text(f"Unknown command: {text}. Type /help to see available commands.", include_playlist=False)

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

    def _detect_intent_with_ollama(self, text: str) -> dict:
        """
        Use OLLAMA to detect user intent and extract entities.
        Handles ALL user input (commands, natural language, questions).
        Returns dict with: intent, entities, ollama_time_ms, confidence
        """
        import time
        import json
        
        if not self._llm:
            return {"intent": "unknown", "entities": {}, "ollama_time_ms": 0, "error": "LLM not configured"}
        
        prompt = f"""You are a music playlist assistant. Analyze this user message and return ONLY a JSON object with intent detection.

User message: "{text}"

Return this exact format (no other text):
{{"intent": "INTENT_NAME", "entities": {{}}, "confidence": 0.9}}

Intents:
- add_track: wants to add/play a song (entities: query or track_name + artist)
- create_playlist: wants to generate/create a playlist from description (entities: description, playlist_name)
- ask_question: asking about tracks/albums/artists in the database, or about similarity (entities: query)
- show_playlist, show_stats, recommend, remove_track, help, greeting, unknown

Examples:
"add starboy" → {{"intent": "add_track", "entities": {{"query": "starboy"}}, "confidence": 0.95}}
"i want starboy" → {{"intent": "add_track", "entities": {{"query": "starboy"}}, "confidence": 0.95}}
"play humble" → {{"intent": "add_track", "entities": {{"query": "humble"}}, "confidence": 0.95}}
"who sings humble" → {{"intent": "ask_question", "entities": {{"query": "who sings humble"}}, "confidence": 0.95}}
"who sings track 1" → {{"intent": "ask_question", "entities": {{"query": "who sings track 1"}}, "confidence": 0.9}}
"what album is shape of you by ed sheeran on" → {{"intent": "ask_question", "entities": {{"query": "what album is shape of you by ed sheeran on"}}, "confidence": 0.95}}
"who sounds like drake" → {{"intent": "ask_question", "entities": {{"query": "who sounds like drake"}}, "confidence": 0.95}}
"who is similar to kendrick lamar" → {{"intent": "ask_question", "entities": {{"query": "who is similar to kendrick lamar"}}, "confidence": 0.95}}
"how many songs by taylor swift" → {{"intent": "ask_question", "entities": {{"query": "how many songs by taylor swift"}}, "confidence": 0.95}}
"create a playlist with upbeat pop songs" → {{"intent": "create_playlist", "entities": {{"description": "upbeat pop songs", "playlist_name": "Upbeat Pop"}}, "confidence": 0.95}}
"make me a workout playlist" → {{"intent": "create_playlist", "entities": {{"description": "workout", "playlist_name": "Workout Mix"}}, "confidence": 0.95}}

JSON only:"""

        start_time = time.time()
        try:
            llm_response = self._llm.generate(
                model=OLLAMA_MODEL,
                prompt=prompt,
                options={"stream": False, "temperature": 0.5, "max_tokens": 150},
            )
            ollama_time_ms = (time.time() - start_time) * 1000
            
            response_text = llm_response.get("response", "") if isinstance(llm_response, dict) else ""
            response_text = response_text.strip() if response_text else ""
            
            # If response is empty, try fallback intent detection
            if not response_text:
                return self._fallback_intent_detection(text, ollama_time_ms)
            
            # Try to extract JSON from response
            # OLLAMA might wrap it in markdown or add extra text
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            # Find JSON object in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                response_text = response_text[json_start:json_end]
            else:
                # No JSON found, use fallback
                return self._fallback_intent_detection(text, ollama_time_ms)
            
            result = json.loads(response_text)
            result["ollama_time_ms"] = ollama_time_ms
            return result
            
        except json.JSONDecodeError as e:
            ollama_time_ms = (time.time() - start_time) * 1000
            # Use fallback instead of returning error
            return self._fallback_intent_detection(text, ollama_time_ms)
            
        except Exception as e:
            ollama_time_ms = (time.time() - start_time) * 1000
            return {
                "intent": "unknown",
                "entities": {},
                "ollama_time_ms": ollama_time_ms,
                "confidence": 0.0,
                "error": str(e)
            }

    def _fallback_intent_detection(self, text: str, ollama_time_ms: float) -> dict:
        """
        Fallback intent detection using simple pattern matching.
        Used when OLLAMA fails to return valid JSON.
        """
        import re
        
        text_lower = text.lower().strip()
        
        # PRIORITY 0: Questions about playlist context (track numbers) should be Q&A
        if re.search(r'(?:who\s+sings|what\s+(?:is|genre|artist|album))\s+track\s+\d+', text_lower):
            return {
                "intent": "ask_question",
                "entities": {"query": text},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        # PRIORITY 1: Simple "add X" or "play X" or "i want X" patterns (most common)
        simple_add_patterns = [
            r'^(?:add|play|queue)\s+(.+)$',  # "add starboy", "play humble"
            r'^i\s+want\s+(.+)$',  # "i want starboy"
            r'^(?:can\s+you\s+)?(?:add|play|put)\s+(.+)$',  # "can you add starboy"
        ]
        
        for pattern in simple_add_patterns:
            match = re.match(pattern, text_lower)
            if match:
                query = match.group(1).strip()
                # Remove trailing filler words like "please", "too", "as well"
                query = re.sub(r'\s+(please|too|as\s+well|also|pls)$', '', query)
                
                # Try to extract artist with "by" or "from" delimiter
                artist = None
                track_name = None
                if ' by ' in query:
                    parts = query.split(' by ', 1)
                    track_name = parts[0].strip()
                    artist = parts[1].strip()
                elif ' from ' in query:
                    parts = query.split(' from ', 1)
                    track_name = parts[0].strip()
                    artist = parts[1].strip()
                
                entities = {}
                if artist and track_name:
                    entities = {"track_name": track_name, "artist": artist}
                else:
                    entities = {"query": query}
                
                return {
                    "intent": "add_track",
                    "entities": entities,
                    "confidence": 0.9,
                    "ollama_time_ms": ollama_time_ms,
                    "fallback": True
                }
        
        # PRIORITY 2: Check for add_track intent with more flexible patterns
        add_patterns = [
            r'\b(add|play|queue|put|want|need|like)\b.*\b(song|track|music)\b',
            r'\b(play|add|queue)\b.*(?:by|from)',
            r'\b(want|need|like)\b.*\b(in|to)\b.*\bplaylist\b',
            r'^(?:do you have|got|find)\b.*\bby\b',  # "do you have starboy by the weeknd"
            r'\b(play|add)\b.*\bby\b',
        ]
        
        is_add_intent = any(re.search(pattern, text_lower) for pattern in add_patterns)
        
        # Also check if it contains music-related words and "by" (artist indicator)
        has_music_terms = any(word in text_lower for word in ['song', 'track', 'music', 'album'])
        has_by = ' by ' in text_lower or ' from ' in text_lower
        has_want_need = any(word in text_lower for word in ['want', 'need', 'like', 'love', 'add', 'play'])
        
        # If it looks like they're asking about a specific song/artist, treat as add
        if (has_by and (has_want_need or has_music_terms)) or is_add_intent:
            # Extract track query - keep everything, remove common filler words
            query = text_lower
            for word in ['i', 'want', 'need', 'like', 'love', 'add', 'play', 'queue', 'put', 
                        'song', 'track', 'music', 'to', 'my', 'in', 'the', 'a', 'an',
                        'playlist', 'too', 'also', 'as well', 'please', 'can you',
                        'do you have', 'got', 'find', 'search']:
                query = re.sub(r'\b' + re.escape(word) + r'\b', '', query)
            query = ' '.join(query.split())  # Clean up extra spaces
            
            # Try to extract artist with "by" or "from" delimiter
            artist = None
            track_name = None
            if ' by ' in query:
                parts = query.split(' by ', 1)
                track_name = parts[0].strip()
                artist = parts[1].strip()
            elif ' from ' in query:
                parts = query.split(' from ', 1)
                track_name = parts[0].strip()
                artist = parts[1].strip()
            
            entities = {}
            if artist and track_name:
                entities = {"track_name": track_name, "artist": artist}
            elif query:
                entities = {"query": query}
            
            return {
                "intent": "add_track",
                "entities": entities,
                "confidence": 0.7,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(remove|delete|drop)\b', text_lower):
            # Try to extract track number
            match = re.search(r'(?:track|song|number)?\s*(\d+)', text_lower)
            entities = {}
            if match:
                entities = {"track_number": int(match.group(1))}
            
            return {
                "intent": "remove_track",
                "entities": entities,
                "confidence": 0.7,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(show|display|list|view|what)\b.*\b(playlist|songs|tracks)\b', text_lower):
            return {
                "intent": "show_playlist",
                "entities": {},
                "confidence": 0.8,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\bstats?\b', text_lower):
            return {
                "intent": "show_stats",
                "entities": {},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(recommend|suggest|find)\b', text_lower):
            match = re.search(r'(\d+)\s*(?:song|track)', text_lower)
            count = int(match.group(1)) if match else 5
            
            return {
                "intent": "recommend",
                "entities": {"count": count},
                "confidence": 0.8,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(create|generate|make|build)\s+(?:a\s+)?playlist', text_lower):
            # Extract description after "playlist with/for/of"
            match = re.search(r'playlist\s+(?:with|for|of|that)\s+(.+)', text_lower)
            description = match.group(1).strip() if match else text_lower
            
            return {
                "intent": "create_playlist",
                "entities": {"description": description},
                "confidence": 0.85,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(help|commands?|options?)\b', text_lower):
            return {
                "intent": "help",
                "entities": {},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(next|more)\b', text_lower):
            return {
                "intent": "navigation",
                "entities": {"direction": "next"},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        elif re.search(r'\b(prev|previous|back)\b', text_lower):
            return {
                "intent": "navigation",
                "entities": {"direction": "previous"},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        # Special case: Questions about tracks/artists/albums in the database
        # "who sings X" = asking about the artist of song X
        # "what album is X on" = asking about the album
        # "who sounds like X" = asking for similar artists
        if re.match(r'^who\s+sings\s+(.+)', text_lower):
            return {
                "intent": "ask_question",
                "entities": {"query": text},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        # Check for database question patterns
        question_patterns = [
            r'^who\s+(?:sings|performs|is\s+the\s+artist|sang)',  # "who sings X", "who is the artist of X"
            r'^what\s+album\s+(?:is|does)',  # "what album is X on"
            r'^who\s+(?:sounds\s+like|is\s+similar\s+to)',  # "who sounds like X"
            r'^how\s+many\s+(?:songs|tracks|albums)\s+by',  # "how many songs by X"
            r'^what\s+albums\s+does',  # "what albums does X have"
            r'^(?:list|show)\s+albums\s+by',  # "list albums by X"
            r'^do\s+you\s+have\s+.+\s+by\s+',  # "do you have X by Y"
        ]
        
        if any(re.search(pattern, text_lower) for pattern in question_patterns):
            return {
                "intent": "ask_question",
                "entities": {"query": text},
                "confidence": 0.9,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        # Check for question patterns about playlist context (not track searches)
        elif re.search(r'^(who|what|when|where|why|how|which|whose)\b', text_lower):
            # Only treat as Q&A if it's about playlist context (track 1, track 3, this song, etc.)
            if re.search(r'track\s+\d+|this\s+(song|track|album)', text_lower):
                return {
                    "intent": "ask_question",
                    "entities": {"query": text},
                    "confidence": 0.8,
                    "ollama_time_ms": ollama_time_ms,
                    "fallback": True
                }
            # Otherwise, treat as track search (e.g., "what is humble" = "find humble")
            else:
                # Extract the query part after the question word
                query_match = re.match(r'^(?:who|what|when|where|why|how|which|whose)\s+(?:is|are|was|were)?\s*(.+)', text_lower)
                if query_match:
                    track_query = query_match.group(1).strip()
                    return {
                        "intent": "add_track",
                        "entities": {"query": track_query},
                        "confidence": 0.7,
                        "ollama_time_ms": ollama_time_ms,
                        "fallback": True
                    }
                # Fallback to Q&A if we can't extract
                return {
                    "intent": "ask_question",
                    "entities": {"query": text},
                    "confidence": 0.5,
                    "ollama_time_ms": ollama_time_ms,
                    "fallback": True
                }
        
        elif re.search(r'\b(genre|artist|album|year|track)\b.*\?', text_lower):
            # "what genre is this?" or similar
            return {
                "intent": "ask_question",
                "entities": {"query": text},
                "confidence": 0.8,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }
        
        else:
            return {
                "intent": "unknown",
                "entities": {},
                "confidence": 0.3,
                "ollama_time_ms": ollama_time_ms,
                "fallback": True
            }

    def _send_options(self, options: List[str]) -> None:
        dialogue_acts = [
            DialogueAct(
                intent=_INTENT_OPTIONS,
                annotations=[SlotValueAnnotation("option", o) for o in options],
            )
        ]
        self._send_text("Here are some options:", include_playlist=False, dialogue_acts=dialogue_acts)
    
    def _handle_with_ollama(self, text: str) -> None:
        """
        Universal handler using OLLAMA for ALL intent detection.
        Routes to appropriate handlers based on detected intent.
        Shows detailed timing for each step.
        """
        import time
        
        # Step 1: OLLAMA intent detection
        intent_result = self._detect_intent_with_ollama(text)
        intent = intent_result.get("intent", "unknown")
        entities = intent_result.get("entities", {})
        ollama_time = intent_result.get("ollama_time_ms", 0)
        confidence = intent_result.get("confidence", 0)
        is_fallback = intent_result.get("fallback", False)
        
        if "error" in intent_result and not is_fallback:
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Ollama: {ollama_time:.0f}ms]</span>"
            self._send_text(f"Error detecting intent: {intent_result['error']}{timing_info}", include_playlist=False)
            return
        
        # Route to appropriate handler with timing
        if intent == "add_track":
            self._handle_add_track_with_timing(entities, ollama_time)
        
        elif intent == "remove_track":
            self._handle_remove_track_with_timing(entities, ollama_time)
        
        elif intent == "show_playlist":
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            stats = self._ps.get_playlist_stats(self._user_key)
            html = self._format_stats(stats) + timing_info
            self._send_playlist_text(html)
        
        elif intent == "show_stats":
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            stats = self._ps.get_playlist_stats(self._user_key)
            html = self._format_stats(stats) + timing_info
            self._send_playlist_text(html)
        
        elif intent == "recommend":
            count = entities.get("count", 5)
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            self._send_text(f"Generating {count} recommendations...{timing_info}", include_playlist=False)
            try:
                self._handle_recommend(limit=count)
            except Exception as e:
                self._send_text(f"Recommendation failed: {e}", include_playlist=False)
        
        elif intent == "clear_playlist":
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            self._ps.clear_playlist(self._user_key)
            self._send_playlist_text(f"Playlist cleared.{timing_info}")
        
        elif intent == "ask_question":
            query = entities.get("query", text)
            ans = self._qa.answer_question(query)
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            self._send_text(ans + timing_info, include_playlist=False)
        
        elif intent == "create_playlist":
            description = entities.get("description", text)
            playlist_name = entities.get("playlist_name", None)
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            self._send_text(f"Creating playlist based on: \"{description}\"...{timing_info}", include_playlist=False)
            try:
                self._handle_auto(description, playlist_name)
            except Exception as e:
                self._send_text(f"Playlist creation failed: {e}", include_playlist=False)
        
        elif intent == "help":
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            help_text = self._help() + timing_info
            self._send_text(help_text, include_playlist=False)
        
        elif intent == "navigation":
            direction = entities.get("direction", "").lower()
            page_num = entities.get("page_number")
            
            if page_num:
                self._paginate(0, set_to=page_num - 1)
            elif direction == "next":
                self._paginate(+1)
            elif direction in ["previous", "prev", "back"]:
                self._paginate(-1)
            else:
                timing_info = self._format_detailed_timing(ollama_time=ollama_time)
                self._send_text(f"Navigation command unclear.{timing_info}", include_playlist=False)
        
        elif intent == "preview":
            track_num = entities.get("track_number")
            if track_num:
                self._handle_play(track_num)
            else:
                timing_info = self._format_detailed_timing(ollama_time=ollama_time)
                self._send_text(f"Please specify a track number to preview.{timing_info}", include_playlist=False)
        
        elif intent == "quit":
            self.goodbye()
        
        elif intent == "greeting":
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            self._send_text(f"Hello! I'm MusicCRS. Try 'add [song name]' or ask 'help'.{timing_info}", include_playlist=False)
        
        else:
            # Unknown intent - provide helpful guidance
            timing_info = self._format_detailed_timing(ollama_time=ollama_time)
            help_examples = """
<b>I'm not sure what you want to do.</b>{timing_info}<br/><br/>
Here are some things I can help with:<br/>
• <b>Add songs</b>: "add humble", "play shape of you by ed sheeran"<br/>
• <b>Remove songs</b>: "remove track 3", "delete last song"<br/>
• <b>View playlist</b>: "show playlist", "what's in my playlist"<br/>
• <b>Get recommendations</b>: "recommend 5 songs"<br/>
• <b>Ask questions</b>: "who sings track 3", "what genre is this"<br/>
• <b>Get help</b>: "help", "show commands"<br/><br/>
Or try typing <code>/help</code> for all available commands.
""".format(timing_info=timing_info)
            self._send_text(help_examples, include_playlist=False)
    
    def _handle_add_track_with_timing(self, entities: dict, ollama_time: float) -> None:
        """Handle add track intent with detailed timing breakdown."""
        import time
        
        # Extract track info from OLLAMA entities
        track_name = entities.get("track_name") or entities.get("query")
        artist = entities.get("artist")
        
        if not track_name:
            timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Ollama: {ollama_time:.0f}ms]</span>"
            self._send_text(
                f"I couldn't understand which track to add.{timing_info}",
                include_playlist=False
            )
            return
        
        # Build search query
        if artist:
            query = f"{artist} {track_name}"
        else:
            query = track_name
        
        # Step 2: R7 autocorrect search
        start_search = time.time()
        correction = self._autocorrect.correct_track_query(query)
        search_time = (time.time() - start_search) * 1000
        
        # Build detailed timing with expandable dropdown
        timing_info = self._format_detailed_timing(ollama_time=ollama_time, correction=correction)
        
        if correction:
            if correction.corrected:
                # High-confidence auto-fix
                self._add_track_and_notify(
                    correction.track_uri,
                    f"✓ Added <b>{correction.artist} – {correction.title}</b>{timing_info}"
                )
                return
            elif correction.suggestions:
                # Show R7 suggestions
                rows = [
                    (uri, artist, title, None)
                    for uri, artist, title, conf in correction.suggestions
                ]
                self._start_disambiguation(query, rows, timing_info)
                return
        
        # Fallback to exact DB search
        start_db = time.time()
        title_rows = self._ps.search_tracks_by_title(self._user_key, query, fetch_all=True, limit=30)
        db_time = (time.time() - start_db) * 1000
        
        # Add DB timing to the detailed breakdown
        timing_info = self._format_detailed_timing(ollama_time=ollama_time, correction=correction, db_time=db_time)
        
        if title_rows:
            if len(title_rows) == 1:
                uri, artist, title, album = title_rows[0]
                self._add_track_and_notify(uri, f"Added <b>{artist} – {title}</b>.{timing_info}")
            else:
                self._start_disambiguation(query, title_rows, timing_info)
        else:
            self._send_text(
                f"No tracks found for <b>{query}</b>.{timing_info}",
                include_playlist=False
            )
    
    def _handle_remove_track_with_timing(self, entities: dict, ollama_time: float) -> None:
        """Handle remove track with timing."""
        timing_info = self._format_detailed_timing(ollama_time=ollama_time)
        
        if "track_number" in entities:
            try:
                track = self._ps.remove(self._user_key, str(entities["track_number"]))
                self._send_playlist_text(f"Removed <b>{track.artist} – {track.title}</b>.{timing_info}")
            except Exception as e:
                self._send_text(f"Error: {str(e)}{timing_info}", include_playlist=False)
        else:
            self._send_text(
                f"Please specify a track number to remove.{timing_info}",
                include_playlist=False
            )
    
    def _send_text(
        self,
        text_html: str,
        *,
        include_playlist: bool = True,
        exclude_cover: bool = True,  # Exclude cover art by default (large binary data)
        dialogue_acts: Optional[list] = None
    ) -> None:
        playlist_obj = self._ps.serialize_current_playlist(self._user_key, exclude_cover=exclude_cover) if include_playlist else None
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
    
    def _send_loading_message(self, message: str) -> None:
        """Send a loading message to the frontend (used during R7 initialization).
        
        This is called by AutocorrectIntegration during first-time loading of the
        inverted index. Since the agent might not be fully connected yet, we
        handle the case where _dialogue_connector is not available.
        """
        try:
            if hasattr(self, '_dialogue_connector') and self._dialogue_connector:
                self._send_text(message, include_playlist=False)
            else:
                # Agent not fully initialized yet, just print to console
                print(f"[Loading] {message}")
        except Exception as e:
            # Silently ignore errors during loading notification
            print(f"[Loading notification failed] {message}: {e}")
    
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
    def _start_disambiguation(self, title: str, rows: List[Tuple[str, str, str, Optional[str]]], timing_info: str = "") -> None:
        self._disambig = {"query": title, "rows": rows, "page": 0, "page_size": 10, "timing_info": timing_info}
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
        timing_info = info.get("timing_info", "")
        total = len(rows)
        start = page * page_size
        end = min(start + page_size, total)

        banner = (
            f"I found <b>{total}</b> tracks with the title <b>{query}</b>{timing_info}. "
            f"Showing <b>{start+1}–{end}</b> of <b>{total}</b>.<br/>"
            f"Type the number to add that track (e.g. <code>12</code>), or use natural language like "
            f"<code>add first two</code>, <code>add all</code>, <code>add all except adele</code>.<br/>"
            f"Navigation: <code>/next</code>, <code>/previous</code>, <code>/page N</code>."
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
    
    def _handle_disambig_selection(self, selection: dict) -> None:
        """Handle natural language selection from disambiguation results (R5.2)."""
        info = self._disambig
        if not info:
            self._send_text("No pending selection.", include_playlist=False)
            return
        
        rows = info["rows"]
        tracks_to_add = []
        
        if selection['type'] == 'all':
            tracks_to_add = rows
        elif selection['type'] == 'range':
            start, end = selection['start'], selection['end']
            tracks_to_add = rows[start:end]
        elif selection['type'] == 'index':
            idx = selection['index']
            if 0 <= idx < len(rows):
                tracks_to_add = [rows[idx]]
        elif selection['type'] == 'exclude_artist':
            artist_to_exclude = selection['artist'].lower()
            tracks_to_add = [
                r for r in rows
                if artist_to_exclude not in r[1].lower()  # r[1] is artist
            ]
        
        # Apply exclusion filter if present (for combinations like "all except X")
        if 'exclude_artist' in selection and selection['type'] != 'exclude_artist':
            artist_to_exclude = selection['exclude_artist'].lower()
            tracks_to_add = [
                r for r in tracks_to_add
                if artist_to_exclude not in r[1].lower()
            ]
        
        if not tracks_to_add:
            self._send_text("No tracks match that selection.", include_playlist=False)
            return
        
        # Add tracks
        added_count = 0
        added_names = []
        for uri, artist, title, album in tracks_to_add:
            try:
                self._ps.add_by_uri(self._user_key, uri, defer_cover=True)
                added_count += 1
                added_names.append(f"<b>{artist} – {title}</b>")
                if added_count >= 10:  # Limit to prevent overwhelming
                    break
            except Exception:
                pass
        
        # Clear disambiguation state
        self._disambig = None
        
        # Refresh cover once at the end
        if added_count > 0:
            self._ps.force_refresh_cover(self._user_key)
            if added_count == 1:
                self._send_playlist_text(f"Added {added_names[0]}.")
            elif added_count <= 3:
                self._send_playlist_text(f"Added {', '.join(added_names)}.")
            else:
                self._send_playlist_text(f"Added {added_count} track(s) to your playlist.")
        else:
            self._send_text("No tracks were added.", include_playlist=False)

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
        
        # Extract timing info from first recommendation (they all have the same latency)
        latency_ms = recs[0].get('latency_ms', 0) if recs else 0
        timing_info = f" <span style='opacity:0.6; font-size:0.9em'>[Recommendation engine: {latency_ms:.0f}ms]</span>"
        
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
            f"<p>Here are some related picks:{timing_info}</p>"
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


    def _handle_auto(self, query: str, playlist_name: str = None) -> None:
        """
        /auto <natural language>
        - Search individual words from query over playlist names via ./data/mpd_titles_fts.sqlite
        - Pick the most popular (most tracks) playlist from results
        - Aggregate tracks from matched playlist IDs via ./data/mpd.sqlite
        - Rank by frequency (decaying weight by rank), enforce <=2 tracks per artist
        """
        from html import escape as _e
        import re

        q = (query or "").strip()
        if not q:
            self._send_text("Usage: <code>/auto your vibe here</code>")
            return

        # Extract individual words from query
        words = [w for w in re.findall(r"[A-Za-z0-9']+", q) if len(w) >= 3]
        
        # Search for each word individually and collect all results
        all_matched = []
        for word in words:
            matched = self._search_mpd_playlists(word, limit=20)
            all_matched.extend(matched)
        
        # Remove duplicates and sort by ntracks (popularity)
        seen_ids = set()
        unique_matched = []
        for m in all_matched:
            if m.get("id") and m.get("id") not in seen_ids:
                seen_ids.add(m["id"])
                unique_matched.append(m)
        
        # Sort by number of tracks (most popular first)
        unique_matched.sort(key=lambda x: x.get("ntracks", 0) or 0, reverse=True)
        
        # Take top 12 playlists
        matched = unique_matched[:12]
        
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

        # Use the playlist_name from OLLAMA if provided, otherwise derive from matched titles
        if playlist_name:
            name = playlist_name
        else:
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
    # Disable Flask auto-reload to prevent re-loading 1GB R7 index on file changes
    import os
    os.environ['FLASK_ENV'] = 'production'  # Disable debug mode
    platform.start()
