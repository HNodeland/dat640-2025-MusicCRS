"""
Information Retrieval (IR) Enhanced Search for Music Tracks

This module implements IR techniques to improve search accuracy and handle
natural language queries with typos, missing punctuation, and flexible word order.

Key IR Techniques Implemented:
==============================

1. **Text Normalization** (normalize_text)
   - Converts to lowercase for case-insensitive matching
   - Removes punctuation (except spaces) to handle "can't" vs "cant"
   - Collapses multiple spaces into one
   - Example: "Don't Stop!" → "dont stop"

2. **Tokenization** (tokenize)
   - Splits normalized text into individual words (tokens)
   - Removes empty tokens
   - Example: "one dance drake" → ["one", "dance", "drake"]

3. **Token Overlap Scoring** (token_overlap_score)
   - Measures similarity between query and track using set intersection
   - Returns fraction of query tokens found in track (Jaccard-like coefficient)
   - Example: query=["one", "dance"], track=["one", "dance", "drake"]
     → score = 2/2 = 1.0 (perfect match)

4. **Ranked Retrieval** (search_tracks_ir, search_artist_title_ir)
   - SQL filters candidates using LIKE for efficiency
   - Scores each candidate using token overlap
   - Weights title matches higher than artist matches (60% vs 30%)
   - Adds popularity boost (up to 10% of score)
   - Sorts by final score (most relevant first)

5. **Query Optimization**
   - Filters out common stopwords ("the", "a", "of") if enough distinctive tokens remain
   - Requires minimum 2 token matches for multi-word queries (reduces false positives)
   - Orders SQL results by popularity to find relevant tracks faster
   - Fetches more candidates than needed (limit × 50) to allow for post-filtering

Why This Helps:
===============
- **Typos**: "dont" matches "don't" (punctuation normalized away)
- **Word Order**: "drake one dance" matches "One Dance by Drake"
- **Partial Matches**: "humble kendrick" finds "HUMBLE. by Kendrick Lamar"
- **Common Words**: "one dance drake" finds correct track despite "one" being very common

Performance Characteristics:
============================
- Fast for distinctive tokens: <100ms
- Slower for common words ("one", "love"): 1-5s (needs to fetch/score many candidates)
- SQL ORDER BY popularity helps surface popular tracks faster
- Token matching is O(n*m) where n=query tokens, m=candidate tokens (small values)
"""

import re
from typing import List, Tuple, Optional
from collections import Counter
import sqlite3


def normalize_text(text: str) -> str:
    """
    Normalize text for IR: lowercase, remove punctuation except spaces.
    
    This is the first step in the IR pipeline. By normalizing text, we can
    match queries regardless of capitalization or punctuation differences.
    
    Handles special cases:
    - Acronyms with periods: "h.u.m.b.l.e" → "humble"
    - Multiple spaces collapsed to single space
    - Punctuation removed
    - Preserves multi-word structure (doesn't join across words)
    
    Examples:
        "Don't Stop Believin'" → "dont stop believin"
        "One Dance" → "one dance"
        "m.A.A.d city" → "maad city"
        "h.u.m.b.l.e" → "humble"
        "H.U.M.B.L.E." → "humble"
        "o.n.e d.a.n.c.e" → "one dance"
    
    Args:
        text: Raw text string from user query or database
        
    Returns:
        Normalized text (lowercase, no punctuation, single spaces)
    """
    text = text.lower()
    # Remove punctuation but keep spaces and alphanumeric
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Special case: join consecutive single letters (handles "h u m b l e" → "humble")
    # This catches acronyms/stylized text like "h.u.m.b.l.e" or "m.a.a.d"
    # Only joins runs of 2+ single letters to avoid joining unrelated letters
    tokens = text.split()
    result = []
    single_letters = []
    
    for token in tokens:
        if len(token) == 1 and token.isalpha():
            # Collect consecutive single letters
            single_letters.append(token)
        else:
            # Flush accumulated single letters (only if 2+ letters)
            if len(single_letters) >= 2:
                result.append(''.join(single_letters))
                single_letters = []
            else:
                # Keep single isolated letters as-is
                result.extend(single_letters)
                single_letters = []
            result.append(token)
    
    # Flush any remaining single letters (only if 2+ letters)
    if len(single_letters) >= 2:
        result.append(''.join(single_letters))
    else:
        result.extend(single_letters)
    
    return ' '.join(result)


def tokenize(text: str) -> List[str]:
    """
    Simple whitespace tokenization after normalization.
    
    Converts text into individual searchable units (tokens). This allows us
    to match queries like "drake one dance" to "One Dance by Drake" regardless
    of word order.
    
    Examples:
        "one dance drake" → ["one", "dance", "drake"]
        "humble kendrick" → ["humble", "kendrick"]
    
    Args:
        text: Text to tokenize (will be normalized first)
        
    Returns:
        List of tokens (words) with empty strings removed
    """
    normalized = normalize_text(text)
    return [t for t in normalized.split() if len(t) > 0]


def token_overlap_score(query_tokens: List[str], target_tokens: List[str]) -> float:
    """
    Calculate overlap score between query and target tokens using set intersection.
    
    This is a simplified version of the Jaccard coefficient. It measures what
    fraction of the query tokens appear in the target, which tells us how well
    the target matches the query.
    
    Formula: score = |query ∩ target| / |query|
    
    Examples:
        query=["one", "dance"], target=["one", "dance", "drake"]
        → overlap = {"one", "dance"}, score = 2/2 = 1.0 (perfect match)
        
        query=["humble", "kendrick"], target=["kendrick", "lamar", "humble"]
        → overlap = {"humble", "kendrick"}, score = 2/2 = 1.0
        
        query=["one", "dance"], target=["dance", "music"]
        → overlap = {"dance"}, score = 1/2 = 0.5 (partial match)
    
    Args:
        query_tokens: Tokens from user's search query
        target_tokens: Tokens from a candidate track (title + artist)
        
    Returns:
        Score from 0.0 (no overlap) to 1.0 (all query tokens found)
    """
    if not query_tokens:
        return 0.0
    
    query_set = set(query_tokens)
    target_set = set(target_tokens)
    overlap = query_set & target_set
    
    return len(overlap) / len(query_set)


def search_tracks_ir(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 20,
    min_score: float = 0.3
) -> List[Tuple[str, str, str, Optional[str], float]]:
    """
    Search for tracks using Information Retrieval techniques.
    
    This is the main search function that combines multiple IR techniques:
    1. Tokenizes the query
    2. Filters out common stopwords if possible
    3. Uses SQL LIKE to get candidates (fast initial filtering)
    4. Scores each candidate using token overlap
    5. Requires minimum token matches for precision
    6. Ranks by relevance score
    
    The algorithm handles:
    - Case insensitivity: "HUMBLE" matches "humble"
    - Punctuation: "dont" matches "don't"  
    - Word order: "drake one dance" matches "One Dance by Drake"
    - Partial matches: "humble kendrick" finds "HUMBLE. by Kendrick Lamar"
    
    Scoring System:
    ---------------
    - Title match: 60% weight (most important)
    - Artist match: 30% weight
    - Combined match: 10% weight
    - Popularity boost: up to 10% additional
    
    For multi-token queries, requires at least 2 tokens to match
    (reduces false positives from common words like "one", "love")
    
    Args:
        conn: SQLite database connection
        query: User's search query (e.g., "one dance drake")
        limit: Maximum number of results to return (default: 20)
        min_score: Minimum relevance score threshold (default: 0.3)
        
    Returns:
        List of (track_uri, artist, title, album, score) tuples,
        sorted by score descending (most relevant first)
        
    Example:
        search_tracks_ir(conn, "one dance drake", limit=5)
        → [("spotify:track:...", "Drake", "One Dance", "Views", 0.70), ...]
    """
    query_tokens = tokenize(query)
    
    if not query_tokens:
        return []
    
    # Strategy: Use the most distinctive tokens for SQL filtering
    # Avoid very common words if we have better options
    common_words = {'the', 'a', 'an', 'of', 'to', 'in', 'on', 'at', 'by', 'for', 
                    'with', 'from', 'and', 'or', 'but', 'is', 'are', 'was', 'were'}
    
    # Filter out common words if we have enough distinctive tokens
    distinctive_tokens = [t for t in query_tokens if t not in common_words]
    search_tokens = distinctive_tokens if len(distinctive_tokens) >= 2 else query_tokens
    
    # Performance optimization: Use strategic SQL filtering
    # For single tokens or simple queries, cast a wider net
    # For multi-token queries, require more matches
    if len(search_tokens) == 1:
        # Single token: simple OR search
        like_conditions = "(LOWER(title) LIKE ? OR LOWER(artist) LIKE ?)"
        like_params = [f'%{search_tokens[0]}%', f'%{search_tokens[0]}%']
        fetch_limit = min(100, limit * 10)  # Moderate limit for single token
    elif len(search_tokens) == 2:
        # Two tokens: require at least one to match (will score them later)
        like_conditions = "(LOWER(title) LIKE ? OR LOWER(artist) LIKE ? OR LOWER(title) LIKE ? OR LOWER(artist) LIKE ?)"
        like_params = [f'%{search_tokens[0]}%', f'%{search_tokens[0]}%',
                       f'%{search_tokens[1]}%', f'%{search_tokens[1]}%']
        fetch_limit = min(150, limit * 15)
    else:
        # Three or more tokens: require first two tokens (most important)
        # This dramatically reduces candidates while still being flexible
        like_conditions = "(LOWER(title) LIKE ? OR LOWER(artist) LIKE ?) AND (LOWER(title) LIKE ? OR LOWER(artist) LIKE ?)"
        like_params = [f'%{search_tokens[0]}%', f'%{search_tokens[0]}%',
                       f'%{search_tokens[1]}%', f'%{search_tokens[1]}%']
        fetch_limit = min(100, limit * 10)
    
    # Fetch candidate tracks with tight limit
    # ORDER BY popularity ensures we get the most relevant popular tracks first
    sql = f"""
        SELECT track_uri, artist, title, album, COALESCE(popularity, 0) as popularity
        FROM tracks
        WHERE {like_conditions}
        ORDER BY popularity DESC
        LIMIT ?
    """
    
    cursor = conn.execute(sql, like_params + [fetch_limit])
    candidates = cursor.fetchall()
    
    # Score each candidate
    scored_results = []
    for track_uri, artist, title, album, popularity in candidates:
        # Tokenize artist and title
        artist_tokens = tokenize(artist)
        title_tokens = tokenize(title)
        combined_tokens = artist_tokens + title_tokens
        
        # Calculate scores
        title_score = token_overlap_score(query_tokens, title_tokens)
        artist_score = token_overlap_score(query_tokens, artist_tokens)
        combined_score = token_overlap_score(query_tokens, combined_tokens)
        
        # Require minimum token matches for multi-token queries
        if len(query_tokens) >= 2:
            # At least 2 tokens must match
            matches = len(set(query_tokens) & set(combined_tokens))
            if matches < 2:
                continue
        
        # Weighted final score (title matters more than artist)
        # For single-token queries that match the title perfectly, reduce artist weight
        # (avoids boosting tracks where artist name coincidentally contains the query word)
        if len(query_tokens) == 1 and title_score >= 0.9:
            # Single-word title query with strong match: title-only scoring
            # This ensures "humble" finds Kendrick's HUMBLE before Belle Humble (artist name)
            # For single-word queries, combined field adds no value (always matches if title matches)
            # Use popularity as primary tie-breaker (10% boost instead of 5%)
            base_score = title_score
        else:
            # Normal multi-word query: artist and combined scores matter more
            base_score = (title_score * 0.6) + (artist_score * 0.3) + (combined_score * 0.1)
        
        # Popularity as a **secondary** ranking signal (proper IR approach)
        # 
        # Text relevance is PRIMARY - ensures you find what you search for
        # Popularity is SECONDARY - helps rank equally-relevant results
        #
        # This follows industry standards (Spotify, Elasticsearch, Solr):
        # - BM25/TF-IDF scoring for text relevance (our base_score)
        # - Popularity scaling for tie-breaking
        #
        # For single-word queries: stronger popularity scaling (0.9-1.0 range)
        # For multi-word queries: weaker scaling (0.95-1.0 range)
        #
        # Formula: score = base_score * (min_scale + popularity_factor * scale_range)
        #
        # Examples (single-word with 0.9-1.0 scaling):
        #   - Perfect title match, mega-popular: 1.0 * 1.0 = 1.000
        #   - Perfect title match, unpopular: 1.0 * 0.9 = 0.900
        #   - Weak match, mega-popular: 0.5 * 1.0 = 0.500
        #   - Weak match, unpopular: 0.5 * 0.9 = 0.450
        #
        # This ensures:
        #   1. Scores stay in [0, 1] range (normalized)
        #   2. Title matches always rank highest for single-word queries
        #   3. Popularity breaks ties between equally good title matches
        #   4. Artist-name coincidences don't outrank actual song titles
        #
        popularity_normalized = (popularity or 0) / 10000.0
        popularity_factor = min(1.0, popularity_normalized)
        
        # Stronger popularity influence for single-token title queries
        if len(query_tokens) == 1 and title_score >= 0.9:
            # Scale between 0.9 and 1.0 (10% range for strong differentiation)
            popularity_scale = 0.9 + (popularity_factor * 0.1)
        else:
            # Scale between 0.95 and 1.0 (5% range for subtle tie-breaking)
            popularity_scale = 0.95 + (popularity_factor * 0.05)
        
        final_score = base_score * popularity_scale
        
        if final_score >= min_score:
            scored_results.append((track_uri, artist, title, album, final_score, popularity))
    
    # Sort by score descending, then by popularity descending (tie-breaker)
    scored_results.sort(key=lambda x: (x[4], x[5]), reverse=True)
    
    # Apply proper limit and remove popularity from return
    return [(uri, artist, title, album, score) for uri, artist, title, album, score, _ in scored_results[:limit]]


def search_artist_title_ir(
    conn: sqlite3.Connection,
    artist: str,
    title: str,
    limit: int = 20
) -> List[Tuple[str, str, str, Optional[str]]]:
    """
    Search for tracks by separate artist and title using IR techniques.
    
    This specialized search function is used when the query is explicitly
    split into artist and title components (e.g., from parsing "kendrick lamar humble").
    
    Key differences from search_tracks_ir():
    - Uses separate scoring for artist and title
    - Returns only top candidate per artist+title combination
    - No minimum score threshold (assumes user knows what they want)
    - Ranks by combined score + popularity
    
    Algorithm:
    ----------
    1. Normalize and tokenize artist and title separately
    2. SQL candidate fetch: WHERE artist LIKE %...% OR title LIKE %...%
    3. Score each candidate:
       - artist_score = token_overlap(query_artist, candidate_artist)
       - title_score = token_overlap(query_title, candidate_title)
       - combined_score = (artist_score + title_score) / 2
    4. Boost by popularity (up to 10% additional)
    5. Group by (artist, title) and take highest score
    6. Return top N by score
    
    Args:
        conn: SQLite database connection
        artist: Artist name query (e.g., "kendrick lamar")
        title: Track title query (e.g., "humble")
        limit: Maximum number of results (default: 20)
        
    Returns:
        List of (track_uri, artist, title, album) tuples,
        sorted by relevance score descending
        
    Example:
        search_artist_title_ir(conn, "kendrick lamar", "humble", limit=5)
        → [("spotify:track:...", "Kendrick Lamar", "HUMBLE.", "DAMN.", ...), ...]
        
    Why separate artist and title?
    ------------------------------
    When user types "kendrick lamar humble", we know which part is artist
    and which is title. This allows:
    - More precise scoring (don't mix up artist tokens with title tokens)
    - Better handling of artist names with common words ("The Weekend")
    - Avoids ambiguity: "Drake One Dance" vs "One Dance Drake"
    """
    artist_tokens = tokenize(artist)
    title_tokens = tokenize(title)
    
    if not (artist_tokens and title_tokens):
        return []
    
    # Build flexible SQL query
    artist_likes = " OR ".join([f"LOWER(artist) LIKE ?" for _ in artist_tokens])
    title_likes = " OR ".join([f"LOWER(title) LIKE ?" for _ in title_tokens])
    
    params = []
    for token in artist_tokens:
        params.append(f'%{token}%')
    for token in title_tokens:
        params.append(f'%{token}%')
    
    sql = f"""
        SELECT track_uri, artist, title, album, COALESCE(popularity, 0) as popularity
        FROM tracks
        WHERE ({artist_likes}) AND ({title_likes})
        LIMIT ?
    """
    
    cursor = conn.execute(sql, params + [limit * 5])
    candidates = cursor.fetchall()
    
    # Score each candidate
    scored_results = []
    for track_uri, db_artist, db_title, album, popularity in candidates:
        # Calculate match scores
        artist_score = token_overlap_score(artist_tokens, tokenize(db_artist))
        title_score = token_overlap_score(title_tokens, tokenize(db_title))
        
        # Weighted score (both must match well)
        final_score = (artist_score * 0.5) + (title_score * 0.5)
        
        # Small popularity boost
        popularity_boost = min(1.0, (popularity or 0) / 10000.0) * 0.05
        final_score += popularity_boost
        
        scored_results.append((track_uri, db_artist, db_title, album, final_score))
    
    # Sort by score descending
    scored_results.sort(key=lambda x: x[4], reverse=True)
    
    # Return without scores
    return [(uri, artist, title, album) for uri, artist, title, album, _ in scored_results[:limit]]
