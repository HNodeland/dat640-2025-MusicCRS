"""
User interface layer for R7 autocorrect.

This module provides the UX for autocorrect suggestions:
1. "Did you mean..." display for ambiguous queries
2. High-confidence auto-fix for clear corrections
3. Natural language selection ("add the first two")

Task: 7.3 - UX: 'Did you mean…', high-confidence auto-fix, NL selection (2 points)

Usage:
    ui = AutocorrectUI(candidate_generator, ml_ranker)
    result = ui.autocorrect("bldning lights", current_playlist=[...])
    
    if result.auto_fixed:
        print(f"Added: {result.track}")
    else:
        print("Did you mean:")
        for suggestion in result.suggestions:
            print(f"  - {suggestion.artist} - {suggestion.title}")
"""

from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AutocorrectSuggestion:
    """Single autocorrect suggestion."""
    track_id: int
    track_uri: str
    artist: str
    title: str
    confidence: float


@dataclass
class AutocorrectResult:
    """Result of autocorrect operation."""
    query: str
    auto_fixed: bool
    track: Optional[AutocorrectSuggestion]  # If auto-fixed
    suggestions: List[AutocorrectSuggestion]  # If not auto-fixed
    latency_ms: float
    query_trigrams: Optional[List[str]] = None  # Debug: trigrams extracted from query
    matched_trigrams: Optional[List[str]] = None  # Debug: trigrams that matched in index
    candidates_found: int = 0  # Debug: number of candidates before ranking
    # Detailed timing breakdown
    gen_timings: Optional[dict] = None  # From candidate_generator.get_last_timings()
    rank_timings: Optional[dict] = None  # From ml_ranker.get_last_timings()


class AutocorrectUI:
    """User interface for autocorrect suggestions."""
    
    def __init__(self, candidate_generator, ml_ranker, high_confidence_threshold: float = 0.9):
        """
        Initialize autocorrect UI.
        
        Args:
            candidate_generator: CandidateGenerator instance
            ml_ranker: MLRanker instance
            high_confidence_threshold: Confidence threshold for auto-fix (default: 0.9)
        """
        self.generator = candidate_generator
        self.ranker = ml_ranker
        self.high_confidence_threshold = high_confidence_threshold
    
    def autocorrect(
        self, 
        query: str,
        current_playlist: Optional[List[str]] = None,
        max_suggestions: int = 5
    ) -> AutocorrectResult:
        """
        Perform autocorrect on a query.
        
        Steps:
        1. Generate candidates (trigram matching)
        2. Rank candidates (ML model)
        3. If top candidate has confidence ≥ threshold: auto-fix
        4. Else: return suggestions for "Did you mean..."
        
        Args:
            query: User's search query (potentially misspelled)
            current_playlist: List of track URIs in current playlist (for context)
            max_suggestions: Maximum number of suggestions to show
            
        Returns:
            AutocorrectResult with either auto-fix or suggestions
        """
        import time
        from normalizer import normalize, extract_trigrams_set
        start = time.time()
        
        # Extract query trigrams for debug info
        query_normalized = normalize(self.generator.apply_aliases(query))
        query_trigrams_set = extract_trigrams_set(query_normalized, normalize_first=False)
        query_trigrams_list = sorted(list(query_trigrams_set))
        
        # Step 1: Generate candidates
        candidates = self.generator.generate_candidates(
            query, 
            threshold=0.15,  # Permissive threshold - ML ranker will promote popular tracks
            limit=150,  # Increase limit to catch correct matches
            max_trigram_frequency=100000  # Allow common trigrams (e.g., "anc", "nce" in "dance")
        )
        
        # Find which trigrams matched in the index
        matched_trigrams = []
        for trigram in query_trigrams_list:
            if trigram in self.generator.inverted_index:
                matched_trigrams.append(trigram)
        
        if not candidates:
            # No candidates found
            elapsed = (time.time() - start) * 1000
            gen_timings = self.generator.get_last_timings()
            return AutocorrectResult(
                query=query,
                auto_fixed=False,
                track=None,
                suggestions=[],
                latency_ms=elapsed,
                query_trigrams=query_trigrams_list,
                matched_trigrams=matched_trigrams,
                candidates_found=0,
                gen_timings=gen_timings,
                rank_timings={}
            )
        
        # Step 2: Rank candidates
        ranked = self.ranker.rank_candidates(
            candidates,
            query=query,
            current_playlist=current_playlist,
            limit=max_suggestions
        )
        
        # Capture timing details
        gen_timings = self.generator.get_last_timings()
        rank_timings = self.ranker.get_last_timings()
        
        # Step 3: Convert to suggestions
        suggestions = []
        for track_id, score, metadata in ranked:
            suggestion = AutocorrectSuggestion(
                track_id=track_id,
                track_uri=metadata['uri'],
                artist=metadata['artist'],
                title=metadata['title'],
                confidence=score
            )
            suggestions.append(suggestion)
        
        # Step 4: Check if we should auto-fix
        auto_fixed = False
        top_track = None
        
        if suggestions and suggestions[0].confidence >= self.high_confidence_threshold:
            auto_fixed = True
            top_track = suggestions[0]
        
        elapsed = (time.time() - start) * 1000
        
        return AutocorrectResult(
            query=query,
            auto_fixed=auto_fixed,
            track=top_track,
            suggestions=suggestions if not auto_fixed else [],
            latency_ms=elapsed,
            query_trigrams=query_trigrams_list,
            matched_trigrams=matched_trigrams,
            candidates_found=len(candidates),
            gen_timings=gen_timings,
            rank_timings=rank_timings
        )
    
    def format_suggestions_html(self, result: AutocorrectResult) -> str:
        """
        Format autocorrect result as HTML for display.
        
        Args:
            result: AutocorrectResult from autocorrect()
            
        Returns:
            HTML string for display in chatbot
        """
        if result.auto_fixed:
            track = result.track
            return (
                f"✓ Auto-corrected to: <b>{track.artist} – {track.title}</b> "
                f"(confidence: {track.confidence:.2f})"
            )
        
        if not result.suggestions:
            return f"No matches found for: <b>{result.query}</b>"
        
        # Build "Did you mean..." list
        html = f"Did you mean:<br/><ol>"
        for i, suggestion in enumerate(result.suggestions, 1):
            html += (
                f"<li>{suggestion.artist} – {suggestion.title} "
                f"<span style='opacity:0.7'>(confidence: {suggestion.confidence:.2f})</span></li>"
            )
        html += "</ol>"
        html += "Type a number to select, or use natural language like 'add the first two'."
        
        return html
    
    def parse_selection(self, text: str, suggestions: List[AutocorrectSuggestion]) -> List[AutocorrectSuggestion]:
        """
        Parse natural language selection from user.
        
        TODO (Task 7.3.3): Implement NL selection parsing
        
        Supported patterns:
        - Numbers: "1", "3"
        - Ordinals: "first", "second", "third"
        - Ranges: "first two", "top 3", "1-3"
        - All: "all", "everything"
        
        Args:
            text: User's selection text
            suggestions: List of suggestions to select from
            
        Returns:
            List of selected suggestions
        """
        import re
        
        text_lower = text.lower().strip()
        
        # Pattern 1: Single number
        if text_lower.isdigit():
            idx = int(text_lower) - 1
            if 0 <= idx < len(suggestions):
                return [suggestions[idx]]
            return []
        
        # Pattern 2: Ordinals
        ordinal_map = {
            'first': 0, '1st': 0,
            'second': 1, '2nd': 1,
            'third': 2, '3rd': 2,
            'fourth': 3, '4th': 3,
            'fifth': 4, '5th': 4,
        }
        
        for ordinal, idx in ordinal_map.items():
            if ordinal in text_lower:
                if idx < len(suggestions):
                    return [suggestions[idx]]
                return []
        
        # Pattern 3: "first N" or "top N"
        match = re.search(r'(?:first|top)\s+(\d+)', text_lower)
        if match:
            n = int(match.group(1))
            return suggestions[:n]
        
        match = re.search(r'first\s+(two|three|four|five)', text_lower)
        if match:
            word_map = {'two': 2, 'three': 3, 'four': 4, 'five': 5}
            n = word_map.get(match.group(1), 1)
            return suggestions[:n]
        
        # Pattern 4: "all"
        if 'all' in text_lower or 'everything' in text_lower:
            return suggestions
        
        # Pattern 5: Range "1-3"
        match = re.search(r'(\d+)\s*-\s*(\d+)', text_lower)
        if match:
            start = int(match.group(1)) - 1
            end = int(match.group(2))
            return suggestions[start:end]
        
        # Default: empty
        return []


def test_ui():
    """Test autocorrect UI."""
    import os
    import sys
    
    # Add parent dir to path
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(parent_dir, 'src'))
    
    from candidate_generator import CandidateGenerator
    from ml_ranker import MLRanker
    
    print("=== Autocorrect UI Test ===\n")
    
    # Initialize components
    data_dir = os.path.join(parent_dir, 'data')
    index_path = os.path.join(data_dir, 'inverted_index.json')
    alias_path = os.path.join(data_dir, 'alias_map.json')
    metadata_path = os.path.join(data_dir, 'track_metadata.json')
    model_path = os.path.join(data_dir, 'ranker_model.pkl')
    
    if not os.path.exists(index_path):
        print("Error: Index files not found. Run index_builder.py first.")
        return
    
    print("Loading candidate generator...")
    generator = CandidateGenerator(index_path, alias_path, metadata_path)
    
    print("Loading ML ranker...")
    ranker = MLRanker(metadata_path, model_path=model_path)  # Use trained ML model
    
    print("Initializing autocorrect UI...\n")
    ui = AutocorrectUI(generator, ranker, high_confidence_threshold=0.5)
    
    # Test queries
    test_queries = [
        "shpe of you",
        "one dnace drake",
        "huble kendrick",
        "city of blnding lights",
    ]
    
    for query in test_queries:
        print(f"Query: '{query}'")
        result = ui.autocorrect(query, max_suggestions=5)
        
        if result.auto_fixed:
            print(f"  ✓ Auto-fixed to: {result.track.artist} - {result.track.title}")
            print(f"    Confidence: {result.track.confidence:.3f}")
        elif result.suggestions:
            print(f"  Did you mean:")
            for i, sug in enumerate(result.suggestions, 1):
                print(f"    {i}. {sug.artist} - {sug.title} (confidence: {sug.confidence:.3f})")
        else:
            print(f"  No matches found")
        
        print(f"  Latency: {result.latency_ms:.1f}ms\n")
    
    # Test selection parsing
    print("=== Testing Selection Parsing ===\n")
    mock_suggestions = [
        AutocorrectSuggestion(1, "uri1", "Artist A", "Track 1", 0.9),
        AutocorrectSuggestion(2, "uri2", "Artist B", "Track 2", 0.8),
        AutocorrectSuggestion(3, "uri3", "Artist C", "Track 3", 0.7),
        AutocorrectSuggestion(4, "uri4", "Artist D", "Track 4", 0.6),
        AutocorrectSuggestion(5, "uri5", "Artist E", "Track 5", 0.5),
    ]
    
    test_selections = [
        "1",
        "first",
        "first two",
        "top 3",
        "1-3",
        "all",
    ]
    
    for selection in test_selections:
        selected = ui.parse_selection(selection, mock_suggestions)
        print(f"Selection: '{selection}' -> {len(selected)} tracks")
        for sug in selected:
            print(f"  - {sug.artist} - {sug.title}")
        print()


if __name__ == "__main__":
    test_ui()
