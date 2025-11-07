"""R7 Autocorrect Integration for MusicCRS - Clean rebuild."""

import os
import sys
import threading
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    """Result of autocorrect."""
    corrected: bool
    track_uri: Optional[str]
    artist: Optional[str]
    title: Optional[str]
    suggestions: List[Tuple[str, str, str, float]]
    confidence: float
    latency_ms: float = 0.0
    query_trigrams: Optional[List[str]] = None
    matched_trigrams: Optional[List[str]] = None
    candidates_found: int = 0
    # Detailed timing breakdown
    timing_details: Optional[dict] = None  # Contains gen_timings and rank_timings


class AutocorrectIntegration:
    """Integrates R7 autocorrect with MusicCRS (Singleton pattern)."""
    
    # Class-level shared state (singleton)
    _shared_ui = None
    _shared_initialized = False
    _shared_enabled = True
    _loading_callback = None  # Callback to notify frontend during loading
    _init_lock = threading.Lock()  # Prevent concurrent initialization
    
    def __init__(self, enabled: bool = True, loading_callback=None):
        """Initialize (shares R7 components across all instances)."""
        self.enabled = AutocorrectIntegration._shared_enabled
        
        # Store loading callback for first initialization
        if loading_callback and not AutocorrectIntegration._loading_callback:
            AutocorrectIntegration._loading_callback = loading_callback
        
        # Initialize shared components on first call
        if enabled and not AutocorrectIntegration._shared_initialized:
            self._initialize()
    
    @property
    def _ui(self):
        """Access shared UI instance."""
        return AutocorrectIntegration._shared_ui
    
    @property
    def _initialized(self):
        """Check if shared instance is initialized."""
        return AutocorrectIntegration._shared_initialized
    
    def _initialize(self):
        """Lazy initialization of R7 components (shared across all instances)."""
        with AutocorrectIntegration._init_lock:
            # Double-check after acquiring lock
            if AutocorrectIntegration._shared_initialized:
                return
            
            # Mark as initializing to prevent re-entry
            AutocorrectIntegration._shared_initialized = True
        
        try:
            # Notify frontend that loading is starting
            if AutocorrectIntegration._loading_callback:
                AutocorrectIntegration._loading_callback(
                    "🔄 <b>Loading R7 autocorrect system...</b><br/>"
                    "<span style='opacity:0.7'>This may take 20-30 seconds on first startup. "
                    "Loading inverted index with 26,000+ n-grams...</span>"
                )
            
            # Add R7 directory to path
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            r7_src = os.path.join(project_root, 'r7-autocorrect', 'src')
            
            if r7_src not in sys.path:
                sys.path.insert(0, r7_src)
            
            # Import R7 components
            from candidate_generator import CandidateGenerator
            from ml_ranker import MLRanker
            from autocorrect_ui import AutocorrectUI
            
            # Initialize components
            data_dir = os.path.join(project_root, 'r7-autocorrect', 'data')
            index_path = os.path.join(data_dir, 'inverted_index.json')
            alias_path = os.path.join(data_dir, 'alias_map.json')
            metadata_path = os.path.join(data_dir, 'track_metadata.json')
            model_path = os.path.join(data_dir, 'ranker_model.pkl')
            
            if not os.path.exists(index_path):
                print("Warning: R7 index not found. Autocorrect disabled.")
                if AutocorrectIntegration._loading_callback:
                    AutocorrectIntegration._loading_callback(
                        "⚠️ <b>R7 autocorrect disabled:</b> Index not found."
                    )
                AutocorrectIntegration._shared_enabled = False
                AutocorrectIntegration._shared_initialized = True
                return
            
            print("Loading R7 autocorrect system...")
            
            # Load components (this is the slow part - loading 513MB index)
            generator = CandidateGenerator(index_path, alias_path, metadata_path)
            ranker = MLRanker(metadata_path, model_path=model_path)
            
            # Auto-fix confidence threshold (0-1 scale, normalized with sigmoid)
            # 
            # THE TRADE-OFF:
            # Lower threshold (0.70-0.75): More auto-fixes, but may surprise users
            #   - Faster UX (fewer confirmations)
            #   - Risk: "hello" might add wrong version
            # 
            # Higher threshold (0.85-0.90): More confirmations, safer
            #   - Slower UX (more disambiguation)
            #   - Benefit: User always sees options for ambiguous queries
            #
            # CURRENT SETTING: 0.85 (conservative)
            # This means "hello" will show suggestions (including Adele as option)
            # But clear matches like "starboy" or typos like "shpe of you" still auto-fix
            # 
            # Confidence is displayed as percentage to users (0.85 = 85%)
            #
            # NOTE: The ML model has a known bias - it values trigram overlap heavily.
            # This means obscure exact matches (e.g., "Нюша - Hello") can score higher
            # than popular partial matches (e.g., "Adele - Hello"). We use a moderate
            # popularity boost to mitigate this, but it's not perfect. The proper fix
            # would be to retrain with more popularity-weighted training data.
            AutocorrectIntegration._shared_ui = AutocorrectUI(generator, ranker, high_confidence_threshold=0.85)
            AutocorrectIntegration._shared_initialized = True
            print("R7 autocorrect system loaded.")
            
            # Notify frontend that loading is complete
            if AutocorrectIntegration._loading_callback:
                AutocorrectIntegration._loading_callback(
                    "✅ <b>R7 autocorrect system ready!</b><br/>"
                    "<span style='opacity:0.7'>Fuzzy track search with typo correction is now available.</span>"
                )
            
        except Exception as e:
            print(f"Warning: Could not initialize R7 autocorrect: {e}")
            if AutocorrectIntegration._loading_callback:
                AutocorrectIntegration._loading_callback(
                    f"⚠️ <b>R7 autocorrect failed to load:</b> {str(e)}"
                )
            AutocorrectIntegration._shared_enabled = False
            AutocorrectIntegration._shared_initialized = True
    
    def correct_track_query(
        self, 
        query: str,
        current_playlist: Optional[List[str]] = None,
        auto_fix_threshold: float = 0.5
    ) -> Optional[CorrectionResult]:
        """Attempt to correct a track query."""
        if not self.enabled:
            return None
        
        self._initialize()
        
        if not self._ui:
            return None
        
        try:
            result = self._ui.autocorrect(
                query,
                current_playlist=current_playlist,
                max_suggestions=5
            )
            
            if not result or (not result.auto_fixed and not result.suggestions):
                return None
            
            if result.auto_fixed and result.track:
                return CorrectionResult(
                    corrected=True,
                    track_uri=result.track.track_uri,
                    artist=result.track.artist,
                    title=result.track.title,
                    suggestions=[],
                    confidence=result.track.confidence,
                    latency_ms=result.latency_ms,
                    query_trigrams=result.query_trigrams,
                    matched_trigrams=result.matched_trigrams,
                    candidates_found=result.candidates_found,
                    timing_details={
                        'gen_timings': result.gen_timings or {},
                        'rank_timings': result.rank_timings or {}
                    }
                )
            
            suggestions = [
                (sug.track_uri, sug.artist, sug.title, sug.confidence)
                for sug in result.suggestions
            ]
            
            return CorrectionResult(
                corrected=False,
                track_uri=None,
                artist=None,
                title=None,
                suggestions=suggestions,
                confidence=result.suggestions[0].confidence if result.suggestions else 0.0,
                latency_ms=result.latency_ms,
                query_trigrams=result.query_trigrams,
                matched_trigrams=result.matched_trigrams,
                candidates_found=result.candidates_found,
                timing_details={
                    'gen_timings': result.gen_timings or {},
                    'rank_timings': result.rank_timings or {}
                }
            )
            
        except Exception as e:
            print(f"R7 autocorrect error: {e}")
            return None
    
    def format_correction_html(self, result: CorrectionResult) -> str:
        """Format correction result as HTML."""
        if result.corrected:
            return (
                f"Auto-corrected to: <b>{result.artist} - {result.title}</b> "
                f"<span style='opacity:0.7'>(confidence: {result.confidence:.2f})</span>"
            )
        
        if result.suggestions:
            html = "Did you mean:<br/><ol>"
            for uri, artist, title, conf in result.suggestions:
                html += f"<li>{artist} - {title} <span style='opacity:0.7'>(conf: {conf:.2f})</span></li>"
            html += "</ol>"
            return html
        
        return "No corrections found."
