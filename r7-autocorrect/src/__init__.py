"""
R7 Autocorrect Package

Context-aware auto-correct for song and artist queries.

Main Components:
    - normalizer: Text normalization and trigram extraction
    - index_builder: Offline inverted index construction
    - candidate_generator: Fast candidate lookup at runtime
    - ml_ranker: ML-based ranking of candidates
    - autocorrect_ui: User interface for suggestions

Quick Start:
    from r7_autocorrect import AutocorrectPipeline
    
    pipeline = AutocorrectPipeline(
        index_path="data/inverted_index.json",
        alias_path="data/alias_map.json",
        metadata_path="data/track_metadata.json"
    )
    
    result = pipeline.autocorrect("bldning lights")
"""

__version__ = "0.1.0"
__author__ = "DAT640 Group Project"

from .normalizer import normalize, extract_trigrams, calculate_trigram_overlap
from .candidate_generator import CandidateGenerator
from .ml_ranker import MLRanker
from .autocorrect_ui import AutocorrectUI, AutocorrectResult, AutocorrectSuggestion

__all__ = [
    "normalize",
    "extract_trigrams",
    "calculate_trigram_overlap",
    "CandidateGenerator",
    "MLRanker",
    "AutocorrectUI",
    "AutocorrectResult",
    "AutocorrectSuggestion",
]
