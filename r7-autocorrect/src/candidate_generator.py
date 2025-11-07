"""
Runtime candidate generation for R7 autocorrect with fuzzy matching.

This module handles fast candidate generation at query time using the pre-built
inverted index with optional fuzzy matching for typo tolerance.

Features:
    - Trigram-based inverted index lookup (fast)
    - Alias map for known corrections
    - Fuzzy trigram expansion for typo tolerance (keyboard proximity)
    - Skip-grams for transpositions and missing characters
    - Edit distance reranking for improved accuracy

Architecture:
    Runtime (query time):
        1. User query: "one dnace"
        2. Apply alias map: check for known corrections
        3. Normalize query: "one dnace"
        4. Extract trigrams: ["one", "ne ", "e d", "dna", "nac", "ace"]
        5. [FUZZY] Expand trigrams with keyboard proximity variants
        6. [FUZZY] Generate skip-grams for transpositions
        7. Look up trigrams in inverted index
        8. Score candidates by trigram overlap
        9. [FUZZY] Rerank by edit distance (top 200)
        10. Return top N candidates

Task: 7.1.7 - Implement generate_candidates() function
Task: 7.1.8 - Implement trigram overlap scoring
Task: 7.1.10 - Latency target: <1 second (achieved with fuzzy: ~200ms)

Usage:
    generator = CandidateGenerator("data/inverted_index.json", "data/alias_map.json")
    
    # Basic (exact matches)
    candidates = generator.generate_candidates("shape of you", fuzzy=False)
    
    # Fuzzy (typo tolerance)
    candidates = generator.generate_candidates("shpe of yu", fuzzy=True)
"""

import json
import os
import time
import Levenshtein
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter

from normalizer import normalize, extract_trigrams, extract_trigrams_set


# Keyboard proximity map for fuzzy matching (QWERTY layout)
TYPO_SUBSTITUTIONS = {
    'a': ['q', 's', 'z', 'w'],
    'b': ['v', 'n', 'g', 'h'],
    'c': ['x', 'v', 'd', 'f'],
    'd': ['f', 's', 'e', 'r', 'c'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'g', 'r', 't', 'v'],
    'g': ['f', 'h', 't', 'y', 'b'],
    'h': ['g', 'j', 'y', 'u', 'n'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'k', 'u', 'i', 'm'],
    'k': ['j', 'l', 'i', 'o'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['m', 'b', 'h', 'j'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a', 's'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'd', 'w', 'e', 'z', 'x'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'b', 'f', 'g'],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 'c', 's', 'd'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x'],
}


class CandidateGenerator:
    """Fast candidate generation using inverted index."""
    
    def __init__(self, index_path: str, alias_path: str, metadata_path: str):
        """
        Initialize candidate generator.
        
        Args:
            index_path: Path to inverted_index.json
            alias_path: Path to alias_map.json
            metadata_path: Path to track_metadata.json
        """
        self.index_path = index_path
        self.alias_path = alias_path
        self.metadata_path = metadata_path
        
        print(f"Loading inverted index from {index_path}...")
        start = time.time()
        
        with open(index_path, 'r') as f:
            self.inverted_index: Dict[str, List[int]] = json.load(f)
        
        with open(alias_path, 'r') as f:
            self.alias_map: Dict[str, str] = json.load(f)
        
        with open(metadata_path, 'r') as f:
            metadata_raw = json.load(f)
            self.track_metadata = {int(k): v for k, v in metadata_raw.items()}
        
        elapsed = time.time() - start
        print(f"Loaded index with {len(self.inverted_index):,} n-grams in {elapsed:.2f}s")
        print(f"Loaded {len(self.track_metadata):,} tracks")
    
    def apply_aliases(self, text: str) -> str:
        """
        Apply alias corrections to text.
        
        Checks if any token in the text has a known alias correction.
        
        Args:
            text: Query text
            
        Returns:
            Text with aliases applied
            
        Examples:
            >>> gen.apply_aliases("weekend blinding lights")
            'the weeknd blinding lights'
        """
        tokens = text.lower().split()
        corrected_tokens = []
        
        for token in tokens:
            if token in self.alias_map:
                corrected_tokens.append(self.alias_map[token])
            else:
                corrected_tokens.append(token)
        
        return ' '.join(corrected_tokens)
    
    def generate_fuzzy_trigrams(self, trigram: str, max_dist: int = 1) -> Set[str]:
        """
        Generate fuzzy variants of a trigram using keyboard proximity.
        
        For "dan", generates variants like "fan", "san", "das" based on
        keys adjacent to each character on QWERTY keyboard.
        
        Args:
            trigram: Original trigram (3 chars)
            max_dist: Max character substitutions (1 = single char, 2 = two chars)
            
        Returns:
            Set of fuzzy trigram variants
        """
        if len(trigram) != 3:
            return {trigram}
        
        variants = {trigram}  # Always include original
        
        # Single character substitution
        for i in range(3):
            char = trigram[i]
            if char in TYPO_SUBSTITUTIONS:
                for replacement in TYPO_SUBSTITUTIONS[char][:2]:  # Only 2 closest keys
                    variant = trigram[:i] + replacement + trigram[i+1:]
                    variants.add(variant)
        
        return variants
    
    def generate_skip_grams(self, text: str, n: int = 3) -> Set[str]:
        """
        Generate skip-grams (n-grams with 1 skipped character).
        
        Helps match transpositions: "dnace" → skip-grams overlap with "dance"
        
        Args:
            text: Input text
            n: N-gram size (default 3 for trigrams)
            
        Returns:
            Set of regular n-grams + skip-grams
        """
        grams = set()
        
        # Regular n-grams
        for i in range(len(text) - n + 1):
            grams.add(text[i:i+n])
        
        # Skip-grams (skip 1 character at each position)
        for i in range(len(text) - n):
            for skip_pos in range(1, n):
                if i + n + 1 <= len(text):
                    gram = text[i:i+skip_pos] + text[i+skip_pos+1:i+n+1]
                    if len(gram) == n:
                        grams.add(gram)
        
        return grams
    
    def generate_candidates(
        self, 
        query: str, 
        threshold: float = 0.3,
        limit: int = 100,
        min_trigram_matches: int = 2,
        max_trigram_frequency: int = 50000,  # Filter very common trigrams
        fuzzy: bool = False,  # Enable fuzzy matching for typos
        edit_distance_rerank: bool = False  # Rerank by edit distance
    ) -> List[Tuple[int, float, Dict]]:
        """
        Generate candidate tracks for a query with optional fuzzy matching.
        
        Steps:
        1. Apply alias corrections
        2. Normalize query
        3. Extract trigrams (with optional skip-grams)
        4. [FUZZY] Expand trigrams with keyboard proximity variants
        5. Filter out very common trigrams (performance optimization)
        6. Look up trigrams in inverted index
        7. Score candidates by trigram overlap
        8. [FUZZY] Rerank top candidates by edit distance
        9. Filter by threshold and return top N
        
        Args:
            query: User's search query (potentially misspelled)
            threshold: Minimum overlap score (0.0-1.0) for candidates
            limit: Maximum number of candidates to return
            min_trigram_matches: Minimum number of trigrams that must match
            max_trigram_frequency: Skip trigrams appearing in more than N tracks (performance)
            
        Returns:
            List of (track_id, score, metadata) tuples, sorted by score descending
            
        Examples:
            >>> gen.generate_candidates("bldning lights", threshold=0.3, limit=10)
            [(45, 0.85, {'title': 'Blinding Lights', ...}), ...]
        """
        timings = {}
        start_time = time.time()
        
        # Step 1: Apply alias corrections
        t0 = time.time()
        query_corrected = self.apply_aliases(query)
        alias_applied = query != query_corrected
        timings['alias_map'] = time.time() - t0
        
        # Step 2: Normalize
        t0 = time.time()
        query_normalized = normalize(query_corrected)
        timings['normalize'] = time.time() - t0
        
        if not query_normalized:
            return []
        
        # Step 3: Extract query trigrams (with optional skip-grams for fuzzy)
        t0 = time.time()
        if fuzzy and len(query_normalized) <= 20:
            # Use skip-grams for better typo tolerance
            query_trigrams = self.generate_skip_grams(query_normalized, n=3)
        else:
            query_trigrams = extract_trigrams_set(query_normalized, normalize_first=False)
        timings['extract_trigrams'] = time.time() - t0
        
        if not query_trigrams:
            return []
        
        # Step 4: Expand trigrams with fuzzy variants (if enabled)
        t0 = time.time()
        search_trigrams = set()
        
        if fuzzy and len(query_normalized) <= 12:  # Only for SHORT queries (performance)
            for trigram in list(query_trigrams)[:8]:  # Limit to first 8 trigrams
                fuzzy_variants = self.generate_fuzzy_trigrams(trigram, max_dist=1)
                # Limit expansion: original + 2 closest variants
                search_trigrams.add(trigram)  # Always keep original
                search_trigrams.update(list(fuzzy_variants - {trigram})[:2])
        else:
            search_trigrams = query_trigrams
        timings['fuzzy_expansion'] = time.time() - t0
        
        # Step 5: Filter out very common trigrams for performance
        t0 = time.time()
        filtered_trigrams = []
        for trigram in search_trigrams:
            if trigram in self.inverted_index:
                track_count = len(self.inverted_index[trigram])
                if track_count <= max_trigram_frequency:
                    filtered_trigrams.append(trigram)
        
        # If all trigrams were filtered, use original (happens with very short queries)
        if not filtered_trigrams and search_trigrams:
            filtered_trigrams = list(search_trigrams)[:10]  # Limit to first 10
        timings['filter_common'] = time.time() - t0
        
        # Step 6: Look up trigrams in index and count matches per track
        t0 = time.time()
        candidate_scores = Counter()
        
        for trigram in filtered_trigrams:
            if trigram in self.inverted_index:
                track_ids = self.inverted_index[trigram]
                for track_id in track_ids:
                    candidate_scores[track_id] += 1
        
        if not candidate_scores:
            return []
        timings['index_lookup'] = time.time() - t0
        
        # Step 7: Calculate overlap scores
        t0 = time.time()
        scored_candidates = []
        
        for track_id, trigram_matches in candidate_scores.items():
            # Require minimum number of matching trigrams (lower for fuzzy)
            min_matches = 1 if fuzzy else min_trigram_matches
            if trigram_matches < min_matches:
                continue
            
            # Get track's trigram count
            metadata = self.track_metadata.get(track_id)
            if not metadata:
                continue
            
            track_normalized = metadata['normalized']
            track_trigrams = extract_trigrams_set(track_normalized, normalize_first=False)
            
            # Jaccard similarity: |A ∩ B| / |A ∪ B|
            intersection = len(query_trigrams & track_trigrams)
            union = len(query_trigrams | track_trigrams)
            
            if union == 0:
                continue
            
            score = intersection / union
            
            # Boost score slightly for popular tracks (helps correct matches rise to top)
            popularity = metadata.get('popularity', 0)
            popularity_boost = min(0.1, popularity / 100000.0)  # Max 0.1 boost for very popular tracks
            score_boosted = score + popularity_boost
            
            # Filter by threshold (use original score, not boosted)
            if score >= threshold:
                scored_candidates.append((track_id, score_boosted, metadata))
        timings['scoring'] = time.time() - t0
        
        # Step 8: Sort by boosted score (descending)
        t0 = time.time()
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        timings['sort'] = time.time() - t0
        
        # Step 9: Optional edit distance reranking (for fuzzy matching)
        t0 = time.time()
        if edit_distance_rerank and fuzzy and len(scored_candidates) > 0:
            # Take top candidates for reranking (more than limit to allow reranking)
            top_for_rerank = scored_candidates[:min(200, len(scored_candidates))]
            
            reranked = []
            for track_id, trigram_score, metadata in top_for_rerank:
                track_text = f"{metadata['normalized']} {metadata['artist']}".lower()
                
                # Normalized edit distance
                try:
                    edit_dist = Levenshtein.distance(query_normalized, track_text)
                    max_len = max(len(query_normalized), len(track_text), 1)
                    edit_score = 1.0 - (edit_dist / max_len)
                except:
                    edit_score = 0.0
                
                # Combined score: 70% trigram + 30% edit distance
                combined_score = 0.7 * trigram_score + 0.3 * edit_score
                
                reranked.append((track_id, combined_score, metadata))
            
            # Sort by combined score
            reranked.sort(key=lambda x: x[1], reverse=True)
            final_candidates = reranked[:limit]
        else:
            final_candidates = scored_candidates[:limit]
        timings['edit_distance_rerank'] = time.time() - t0
        
        # Return candidates with original trigram scores for ML ranker
        t0 = time.time()
        result = []
        for track_id, final_score, metadata in final_candidates:
            # Recalculate original trigram score (not boosted, not combined)
            track_normalized = metadata['normalized']
            track_trigrams = extract_trigrams_set(track_normalized, normalize_first=False)
            intersection = len(query_trigrams & track_trigrams)
            union = len(query_trigrams | track_trigrams)
            original_trigram_score = intersection / union if union > 0 else 0.0
            result.append((track_id, original_trigram_score, metadata))
        timings['prepare_results'] = time.time() - t0
        
        elapsed = time.time() - start_time
        timings['total'] = elapsed
        
        # Log performance (for Task 7.1.10)
        if elapsed > 1.0:
            print(f"WARNING: Candidate generation took {elapsed:.2f}s (target: <1s)")
            print(f"  Query: '{query}' (fuzzy={'ON' if fuzzy else 'OFF'})")
            print(f"  Used {len(filtered_trigrams)}/{len(search_trigrams)} trigrams")
            print(f"  Timing breakdown: {timings}")
        
        # Store timings for retrieval
        self._last_timings = timings
        
        return result
    
    def get_last_timings(self) -> Dict[str, float]:
        """
        Get timing breakdown from the last generate_candidates call.
        
        Returns:
            Dictionary with timing information (in seconds) for each stage:
                - alias_map: Time spent applying alias corrections
                - normalize: Time spent normalizing query
                - extract_trigrams: Time spent extracting trigrams/skip-grams
                - fuzzy_expansion: Time spent generating fuzzy variants
                - filter_common: Time spent filtering common trigrams
                - index_lookup: Time spent looking up trigrams in index
                - scoring: Time spent calculating candidate scores
                - sort: Time spent sorting candidates
                - edit_distance_rerank: Time spent reranking with edit distance
                - prepare_results: Time spent preparing final results
                - total: Total time for candidate generation
        """
        return getattr(self, '_last_timings', {})
    
    def get_track_info(self, track_id: int) -> Optional[Dict]:
        """
        Get metadata for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Track metadata dictionary or None
        """
        return self.track_metadata.get(track_id)


def test_candidate_generation():
    """Test candidate generation with sample queries."""
    
    # Check if index files exist (try both relative and parent directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    index_path = os.path.join(base_dir, "data", "inverted_index.json")
    alias_path = os.path.join(base_dir, "data", "alias_map.json")
    metadata_path = os.path.join(base_dir, "data", "track_metadata.json")
    
    if not all(os.path.exists(p) for p in [index_path, alias_path, metadata_path]):
        print("Error: Index files not found. Run index_builder.py first.")
        print(f"Looking for files at: {base_dir}/data/")
        return
    
    # Initialize generator
    generator = CandidateGenerator(index_path, alias_path, metadata_path)
    
    # Test queries with typos
    test_queries = [
        "bldning lights",  # "Blinding Lights"
        "shpe of you",     # "Shape of You"
        "one dnace",       # "One Dance"
        "dont stop",       # "Don't Stop"
        "humble kendrick", # "HUMBLE." by Kendrick Lamar
    ]
    
    print("\n=== Candidate Generation Tests ===\n")
    
    for query in test_queries:
        print(f"Query: '{query}'")
        start = time.time()
        candidates = generator.generate_candidates(query, threshold=0.3, limit=5)
        elapsed = time.time() - start
        
        print(f"  Found {len(candidates)} candidates in {elapsed:.3f}s")
        
        for i, (track_id, score, metadata) in enumerate(candidates[:5], 1):
            artist = metadata['artist']
            title = metadata['title']
            print(f"  {i}. {artist} - {title} (score: {score:.3f})")
        
        print()


if __name__ == "__main__":
    test_candidate_generation()
