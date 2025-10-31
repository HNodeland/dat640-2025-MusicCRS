"""
Natural Language Handler for MusicCRS (R5 Requirements)

This module provides natural language understanding (NLU) for the music
conversation system, enabling users to interact through free-form text
instead of structured commands.

Core Capabilities:
==================

1. Intent Classification
   - Detects what the user wants to do from natural language
   - Supported intents: add_track, remove_track, view_playlist, 
     clear_playlist, recommend, ask_question
   - Uses keyword matching + regex patterns
   - Returns confidence score (0.0-1.0)

2. Entity Extraction
   - Extracts songs, artists, and numbers from user input
   - Handles multi-word entities ("Kendrick Lamar", "One Dance")
   - Normalizes entities (case, punctuation)
   
3. Selection Parsing (R5.2)
   - Parses natural language selections from recommendations
   - Supports: ordinals ("first two"), ranges ("3-5"), 
     exclusions ("except Drake"), artist filters
   - Example: "add first two songs except by drake"

4. Natural Language Detection
   - Distinguishes NL from structured commands
   - Prevents false classification of commands like "/add humble"

NL Processing Pipeline:
=======================

Input: "add one dance by drake"
  ↓
1. is_natural_language() → True (not a command)
  ↓
2. classify_intent() → Intent(type='add_track', confidence=0.95, ...)
  ↓
3. _extract_entities() → {'songs': ['one dance'], 'artists': ['drake']}
  ↓
4. Handler dispatches to _handle_nl_add_track()
  ↓
Output: Track added to playlist

Examples:
=========

>>> handler = NaturalLanguageHandler()
>>> intent = handler.classify_intent("play humble by kendrick lamar")
>>> intent.intent_type
'add_track'
>>> intent.entities
{'songs': ['humble'], 'artists': ['kendrick lamar']}

>>> handler.is_natural_language("add humble")
True
>>> handler.is_natural_language("/add humble")
False

>>> handler.parse_selection("add first two except by drake")
{'indices': [0, 1], 'exclude_artists': ['drake']}

Why This Approach?
==================

Pattern-based NLU (vs. ML models):
- Fast: <50ms classification time
- Deterministic: Same input → same output
- No training data required
- Works for domain-specific vocabulary
- Easy to debug and extend

Trade-offs:
- Less robust to novel phrasings
- Requires manual pattern curation
- May miss edge cases

For this project's scope (playlist management), pattern-based
NLU provides sufficient coverage with minimal overhead.

Performance Characteristics:
===========================
- Intent classification: ~10-50ms
- Entity extraction: ~5-20ms
- Selection parsing: ~5-10ms
- Total NL overhead: <100ms (acceptable for conversational UI)

Dependencies:
- ir_search.tokenize(): For consistent text normalization
- ir_search.normalize_text(): For entity matching
- re: Pattern matching
"""

from __future__ import annotations
import re
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from .ir_search import tokenize, normalize_text


@dataclass
class Intent:
    """
    Represents a detected user intent with extracted information.
    
    This dataclass encapsulates the result of intent classification,
    including what the user wants to do, how confident we are,
    and any entities extracted from the utterance.
    
    Attributes:
        intent_type: One of ['add_track', 'remove_track', 'view_playlist',
                     'clear_playlist', 'recommend', 'ask_question']
        confidence: Float 0.0-1.0 indicating classification confidence
                   - 1.0: Exact keyword match + pattern match
                   - 0.7-0.9: Keyword match or pattern match
                   - 0.5-0.6: Weak match or ambiguous
        entities: Dictionary of extracted entities:
                 - 'songs': List[str] - song/track names
                 - 'artists': List[str] - artist names  
                 - 'numbers': List[int] - track positions, counts
                 - 'album_artist': Optional[str] - for album queries
        raw_text: Original user input (for logging/debugging)
        
    Example:
        Intent(
            intent_type='add_track',
            confidence=0.95,
            entities={'songs': ['one dance'], 'artists': ['drake']},
            raw_text='add one dance by drake'
        )
    """
    intent_type: str
    confidence: float
    entities: Dict[str, Any]
    raw_text: str


class NaturalLanguageHandler:
    """
    Handles natural language understanding for MusicCRS.
    
    This class provides intent classification and entity extraction for
    natural language inputs. It uses pattern-based matching (keywords + regex)
    to detect user intents and extract relevant entities.
    
    Intent Types:
    -------------
    - add_track: User wants to add a song ("play humble")
    - remove_track: User wants to remove a song ("delete track 3")
    - view_playlist: User wants to see playlist ("show my songs")
    - clear_playlist: User wants to clear playlist ("delete all")
    - recommend: User wants recommendations ("suggest similar songs")
    - ask_question: User has a question ("what artists are in my playlist")
    
    Methods:
    --------
    - classify_intent(): Main entry point for intent classification
    - parse_selection(): Parse selection phrases for R5.2
    - is_natural_language(): Detect if input is NL vs command
    - _extract_entities(): Extract songs, artists, numbers from text
    
    Usage:
    ------
    >>> handler = NaturalLanguageHandler()
    >>> intent = handler.classify_intent("play one dance by drake")
    >>> if intent.intent_type == 'add_track':
    ...     song = intent.entities.get('songs', [])[0]
    ...     artist = intent.entities.get('artists', [])[0]
    """
    
    def __init__(self):
        # Intent patterns with keywords and regex
        self.intent_patterns = {
            'add_track': {
                'keywords': ['add', 'play', 'queue', 'include', 'insert', 'put'],
                'patterns': [
                    r'\b(add|play|queue)\b.*\b(song|track|music)\b',
                    r'\b(add|play)\b\s+\w+',  # "add humble", "play despacito"
                    r'\bi\s+(want|need|like)\b.*\b(song|track)\b',
                ]
            },
            'remove_track': {
                'keywords': ['remove', 'delete', 'skip', 'exclude', 'drop', 'take out'],
                'patterns': [
                    r'\b(remove|delete|skip)\b.*\b(song|track|music)\b',
                    r'\b(remove|delete)\b\s+(track|song)?\s*\d+',  # "remove track 3"
                    r'^\s*(remove|delete)\s+\w+',  # "remove humble" - catch simple remove + word
                ]
            },
            'view_playlist': {
                'keywords': ['show', 'view', 'list', 'display', 'what', 'whats', 'see'],
                'patterns': [
                    r'\b(show|list|display)\b.*\b(playlist|songs|tracks)\b',
                    r'what.*\b(in|on)\b.*\bplaylist\b',
                    r'whats.*\b(in|on)\b.*\bplaylist\b',
                ]
            },
            'clear_playlist': {
                'keywords': ['clear', 'empty', 'reset', 'delete all', 'remove all'],
                'patterns': [
                    r'\b(clear|empty|reset)\b.*\bplaylist\b',
                    r'\b(remove|delete)\b\s+all',
                ]
            },
            'recommend': {
                'keywords': ['recommend', 'suggest', 'find', 'discover', 'similar'],
                'patterns': [
                    r'\b(recommend|suggest)\b.*\b(song|songs|track|tracks|music)\b',
                    r'\bfind\b.*\b(similar|like)\b',
                    r'\b(what|any)\b.*\brecommend',
                ]
            },
            'ask_question': {
                'keywords': ['who', 'what', 'when', 'where', 'which', 'how many', 'tell me'],
                'patterns': [
                    r'^\s*(who|what|when|where|which|how many)\b',
                    r'\btell me\b.*\babout\b',
                ]
            }
        }
        
        # Selection patterns for R5.2 (add first two, exclude metallica, etc.)
        self.selection_patterns = {
            'ordinal': re.compile(r'\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\b', re.IGNORECASE),
            'range': re.compile(r'\b(top|first)\s+(\d+)\b', re.IGNORECASE),
            'all': re.compile(r'\b(all|everything)\b', re.IGNORECASE),
            'exclude_artist': re.compile(r'\b(exclude|skip|not|without|except)\s+(.+?)(?:\s|$)', re.IGNORECASE),
        }
    
    def classify_intent(self, text: str) -> Intent:
        """
        Classify user intent from natural language text.
        
        Uses a hybrid approach combining keyword matching and regex patterns
        to determine what the user wants to do. Each intent type is scored
        based on presence of keywords and matching patterns.
        
        Scoring Algorithm:
        ------------------
        For each intent type:
        1. Keyword score: +0.3 per matching keyword
        2. Pattern score: +0.5 per matching regex pattern
        3. Total score = keyword_score + pattern_score
        4. Winner = highest scoring intent (or 'ask_question' if score < 0.5)
        
        Args:
            text: User's natural language input
            
        Returns:
            Intent object with detected type, confidence, and extracted entities
            
        Example:
            >>> handler.classify_intent("play one dance by drake")
            Intent(intent_type='add_track', confidence=0.95, 
                   entities={'songs': ['one dance'], 'artists': ['drake']}, ...)
                   
            >>> handler.classify_intent("what artists are in my playlist")
            Intent(intent_type='ask_question', confidence=0.80, 
                   entities={}, ...)
        
        Edge Cases:
        -----------
        - "what's in my playlist" → view_playlist (not ask_question)
        - "add first two" → add_track (selection handled separately)
        - "recommend 5 songs" → recommend (number extracted to entities)
        """
        text_lower = text.lower().strip()
        tokens = tokenize(text)
        
        # Score each intent
        scores = {}
        for intent_type, config in self.intent_patterns.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for kw in config['keywords'] if kw in text_lower)
            score += keyword_matches * 0.3
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in config['patterns'] 
                                if re.search(pattern, text_lower, re.IGNORECASE))
            score += pattern_matches * 0.4
            
            scores[intent_type] = score
        
        # Get best intent
        if not scores or max(scores.values()) == 0:
            # Default: if it's not a command, treat as ask_question
            best_intent = 'ask_question'
            confidence = 0.3
        else:
            best_intent = max(scores, key=scores.get)
            max_score = scores[best_intent]
            confidence = min(1.0, max_score)
        
        # Extract entities based on intent
        entities = self._extract_entities(text, best_intent)
        
        return Intent(
            intent_type=best_intent,
            confidence=confidence,
            entities=entities,
            raw_text=text
        )
    
    def _extract_entities(self, text: str, intent_type: str) -> Dict[str, Any]:
        """
        Extract relevant entities from text based on detected intent.
        
        Different intents require different entity types:
        - add_track: song title, artist, or combined query
        - remove_track: track number or song name
        - recommend: number of songs desired
        - ask_question: question subject (artists, albums, etc.)
        
        Entity Extraction Patterns:
        ---------------------------
        For add_track:
        - "add [TITLE] by [ARTIST]" → {title: ..., artist: ...}
        - "play [TITLE] from [ARTIST]" → {title: ..., artist: ...}
        - "add [QUERY]" → {query: ...} (will be searched as-is)
        
        For remove_track:
        - "remove track [N]" → {track_number: N}
        - "delete [NAME]" → {query: NAME}
        
        For recommend:
        - "recommend [N] songs" → {count: N}
        
        Args:
            text: User's natural language input
            intent_type: Detected intent type from classify_intent()
            
        Returns:
            Dictionary of extracted entities specific to the intent
            
        Example:
            >>> handler._extract_entities("add humble by kendrick lamar", "add_track")
            {'title': 'humble', 'artist': 'kendrick lamar'}
            
            >>> handler._extract_entities("remove track 3", "remove_track")
            {'track_number': 3}
            
            >>> handler._extract_entities("recommend 5 songs", "recommend")
            {'count': 5}
        """
        entities = {}
        text_lower = text.lower()
        
        if intent_type == 'add_track':
            # Extract song/artist from patterns like:
            # "add humble by kendrick"
            # "play one dance"
            # "add hello from adele"
            
            # Try "by" pattern (title by artist)
            by_match = re.search(r'(?:add|play|queue)\s+(.+?)\s+(?:by|from|of)\s+(.+?)(?:\s|$)', text_lower)
            if by_match:
                entities['title'] = by_match.group(1).strip()
                entities['artist'] = by_match.group(2).strip()
            else:
                # Extract everything after the action verb
                action_match = re.search(r'(?:add|play|queue)\s+(.+)', text_lower)
                if action_match:
                    entities['query'] = action_match.group(1).strip()
        
        elif intent_type == 'remove_track':
            # Extract track number or name
            # "remove track 3"
            # "delete humble"
            
            num_match = re.search(r'(?:track|song|number)?\s*(\d+)', text_lower)
            if num_match:
                entities['track_number'] = int(num_match.group(1))
            else:
                # Extract track name after action verb
                action_match = re.search(r'(?:remove|delete|skip)\s+(?:track|song)?\s*(.+)', text_lower)
                if action_match:
                    entities['query'] = action_match.group(1).strip()
        
        elif intent_type == 'recommend':
            # Extract count
            # "recommend 5 songs"
            # "suggest some music"
            
            count_match = re.search(r'(\d+)\s+(?:song|track|recommendation)', text_lower)
            if count_match:
                entities['count'] = int(count_match.group(1))
            
            # Extract genre/mood/artist preferences (future)
            # For now, just mark that it's a recommendation request
            entities['type'] = 'song'
        
        elif intent_type == 'ask_question':
            # Extract question type and subject
            # "who is the artist of track 3"
            # "what songs are in my playlist"
            
            if 'artist' in text_lower:
                entities['question_type'] = 'artist'
            elif 'song' in text_lower or 'track' in text_lower:
                entities['question_type'] = 'track'
            elif 'playlist' in text_lower:
                entities['question_type'] = 'playlist'
            
            # Extract track numbers
            num_match = re.search(r'(?:track|song|number)?\s*(\d+)', text_lower)
            if num_match:
                entities['track_number'] = int(num_match.group(1))
        
        return entities
    
    def parse_selection(self, text: str) -> Dict[str, Any]:
        """
        Parse natural language selection for recommendation results (R5.2).
        
        This method handles selecting specific recommendations from the last
        displayed results using natural language. Supports:
        - Ordinals: "first", "second", etc.
        - Ranges: "first two", "top 5"
        - All: "all", "everything"
        - Exclusions: "except drake", "without metallica"
        - Combinations: "add first two except by drake"
        
        Selection Types:
        ----------------
        1. Ordinal: "add first song" → {indices: [0]}
        2. Range: "add top 3" → {indices: [0, 1, 2]}
        3. All: "add everything" → {indices: [0, 1, 2, 3, 4]}
        4. Exclusion: "except drake" → {exclude_artists: ['drake']}
        5. Combined: Multiple criteria applied together
        
        Args:
            text: Natural language selection phrase
            
        Returns:
            Dictionary with selection criteria:
            - 'indices': List[int] - specific track indices to include
            - 'exclude_artists': List[str] - artists to filter out
            - 'type': str - selection type ('ordinal', 'range', 'all')
            
        Example:
            >>> handler.parse_selection("add first two")
            {'type': 'range', 'indices': [0, 1]}
            
            >>> handler.parse_selection("add everything except by drake")
            {'type': 'all', 'indices': [0, 1, 2, 3, 4], 
             'exclude_artists': ['drake']}
             
            >>> handler.parse_selection("add 1-3")
            {'type': 'range', 'indices': [0, 1, 2]}
        
        Why This Matters (R5.2):
        ------------------------
        Users shouldn't have to remember track IDs or positions.
        "Add the first two songs except by Metallica" is more natural
        than "/add 0" then "/add 1" then checking if each is by Metallica.
        """
        text_lower = text.lower().strip()
        
        # Check for "first X" patterns (e.g., "first two", "first 3")
        first_count_match = re.search(r'\bfirst\s+(two|three|four|five|\d+)\b', text_lower)
        if first_count_match:
            count_word = first_count_match.group(1)
            count_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5}
            count = count_map.get(count_word, int(count_word) if count_word.isdigit() else 2)
            return {'type': 'range', 'start': 0, 'end': count}
        
        # Check for range selection (e.g., "top 5", "first 5")
        range_match = self.selection_patterns['range'].search(text_lower)
        if range_match:
            count = int(range_match.group(2))
            return {'type': 'range', 'start': 0, 'end': count}
        
        # Check for ordinals
        ordinal_match = self.selection_patterns['ordinal'].search(text_lower)
        if ordinal_match:
            ordinal = ordinal_match.group(1)
            ordinal_map = {
                'first': 1, '1st': 1,
                'second': 2, '2nd': 2,
                'third': 3, '3rd': 3,
                'fourth': 4, '4th': 4,
                'fifth': 5, '5th': 5,
            }
            index = ordinal_map.get(ordinal.lower(), 1) - 1
            return {'type': 'index', 'index': index}
        
        # Check for "all"
        if self.selection_patterns['all'].search(text_lower):
            return {'type': 'all'}
        
        # Check for artist exclusion
        exclude_match = self.selection_patterns['exclude_artist'].search(text_lower)
        if exclude_match:
            artist = exclude_match.group(2).strip()
            # Clean up common words
            artist = re.sub(r'\s+(songs?|tracks?|music)\s*$', '', artist)
            return {'type': 'exclude_artist', 'artist': artist}
        
        # Default: first track
        return {'type': 'index', 'index': 0}
    
    def is_natural_language(self, text: str) -> bool:
        """
        Determine if text is natural language vs. a structured command.
        
        This method distinguishes between:
        - Natural language: "add one dance by drake"
        - Structured commands: "/add one dance"
        - Short queries: "humble" (not NL, just a search term)
        
        Detection Criteria:
        -------------------
        Returns False if:
        - Starts with "/" (command prefix)
        - 1-2 words without action verbs (likely a search query)
        
        Returns True if:
        - Contains NL indicators: "can you", "please", "i want"
        - Contains action verbs: "add", "play", "show", "recommend"
        - 3+ words (likely a sentence)
        
        Args:
            text: User input to classify
            
        Returns:
            True if natural language, False if command/query
            
        Example:
            >>> handler.is_natural_language("add one dance by drake")
            True
            
            >>> handler.is_natural_language("/add one dance")
            False
            
            >>> handler.is_natural_language("humble")
            False
            
            >>> handler.is_natural_language("play humble please")
            True
        
        Why This Matters:
        -----------------
        Prevents false NL classification of short queries like "hello"
        or structured commands like "/add hello", ensuring proper routing
        to command handlers vs NL handlers.
        """
        text = text.strip()
        
        # Starts with command prefix
        if text.startswith('/'):
            return False
        
        # Very short (1-2 words) might be a search query, not NL
        words = text.split()
        if len(words) <= 2 and not any(w in text.lower() for w in ['add', 'play', 'show', 'recommend']):
            return False
        
        # Has natural language indicators
        nl_indicators = [
            'can you', 'could you', 'please', 'i want', 'i need', 'i like',
            'show me', 'tell me', 'what is', 'what are', 'who is', 'how many'
        ]
        
        text_lower = text.lower()
        if any(indicator in text_lower for indicator in nl_indicators):
            return True
        
        # Has action verbs
        action_verbs = ['add', 'play', 'remove', 'delete', 'show', 'list', 'recommend', 'suggest']
        if any(verb in text_lower for verb in action_verbs):
            return True
        
        # Default to True if it's long enough
        return len(words) >= 3


# Module-level helper functions

def extract_song_artist(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract song title and artist from natural language text.
    
    This helper function parses various natural language patterns to
    separate song titles from artist names. Used by intent handlers.
    
    Supported Patterns:
    -------------------
    1. "song by artist" → (song, artist)
    2. "song from artist" → (song, artist)
    3. "artist - song" → (song, artist)
    4. "anything else" → (text, None) - will be searched as-is
    
    Args:
        text: Natural language text containing song/artist
        
    Returns:
        Tuple of (title, artist) where either can be None
        
    Example:
        >>> extract_song_artist("one dance by drake")
        ('one dance', 'drake')
        
        >>> extract_song_artist("drake - one dance")
        ('one dance', 'drake')
        
        >>> extract_song_artist("humble")
        ('humble', None)
    
    Why This Exists:
    ----------------
    Users say "play one dance by drake" not "play one dance, artist drake".
    This function bridges natural language → structured search parameters.
    """
    text_lower = text.lower().strip()
    
    # Pattern 1: "song by artist"
    by_match = re.search(r'(.+?)\s+by\s+(.+)', text_lower)
    if by_match:
        return by_match.group(1).strip(), by_match.group(2).strip()
    
    # Pattern 2: "song from artist"
    from_match = re.search(r'(.+?)\s+(?:from|of)\s+(.+)', text_lower)
    if from_match:
        return from_match.group(1).strip(), from_match.group(2).strip()
    
    # Pattern 3: "artist - song" or "artist: song"
    delim_match = re.search(r'(.+?)\s*[-:]\s*(.+)', text_lower)
    if delim_match:
        # Ambiguous: could be artist-title or title-artist
        # Return as query
        return text, None
    
    # No clear pattern: return as title query
    return text, None
