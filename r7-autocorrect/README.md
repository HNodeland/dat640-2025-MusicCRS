# R7: Context-Aware Auto-Correct for Songs & Artists

## Overview

This module implements intelligent auto-correction for typo-ridden song and artist queries using:
1. **Trigram-based inverted index** (26,000+ n-grams) for fast candidate generation with fuzzy matching
2. **ML-based ranking** using pairwise RankSVM combining edit distance, token/trigram overlap, popularity, and playlist context
3. **Smart UX** with high-confidence auto-fixes (≥85%) and "Did you mean..." suggestions
4. **Singleton pattern** for efficient memory usage (513MB index loaded once, shared across all sessions)

## Architecture

```
User Query: "shpe of yu" (typo)
     ↓
[AutocorrectIntegration] → Singleton wrapper with lazy loading
     ↓
[Normalizer] → lowercase, strip punctuation → "shpe of yu"
     ↓
[Alias Map] → Check known corrections (e.g., "weekend" → "The Weeknd")
     ↓
[Trigram Extraction] → ["shp", "hpe", "pe ", "e o", " of", "of ", "f y", " yu"]
     ↓
[Fuzzy Expansion] → Keyboard proximity variants (optional)
     - "shp" → "ahp", "dhp", "sgp", "sjp" (nearby keys)
     - Skip-grams for transpositions: "spe", "she"
     ↓
[Inverted Index Lookup] → Query 26,000+ trigram → track_id mappings
     ↓
Candidates: ~200-500 tracks with trigram overlap scores
     ↓
[Edit Distance Reranking] → Levenshtein distance on top 200 candidates
     ↓
[ML Ranker - RankSVM] → Score using 5 features:
     1. Normalized edit distance (0-1)
     2. Token overlap (Jaccard similarity)
     3. Trigram overlap (from index)
     4. Popularity score (log-scaled from MPD)
     5. Playlist-context boost (co-occurrence with current tracks)
     ↓
Top 5: [("Shape of You", 0.95), ("Shake It Off", 0.42), ...]
     ↓
[UX Layer] → Auto-add if confidence ≥ 0.85, else show "Did you mean..."
```

**Performance:**
- Candidate generation: ~150-200ms (fuzzy enabled)
- ML ranking: ~30-40ms
- Total latency: <300ms (well under 1 second target)

## Key Components

### Core Modules

#### 7.1: Candidate Generation (4 points) ✅
- **normalizer.py**: Text normalization (lowercase, punctuation, brackets, accents)
  - `normalize()`: Clean query text
  - `extract_trigrams()`: Generate character trigrams
  - `extract_trigrams_set()`: Fast set-based trigram extraction
  - `extract_bigrams()`: Generate character bigrams for additional context

- **index_builder.py**: Offline index construction from MPD dataset
  - Builds trigram → track_id inverted index (26,000+ unique trigrams)
  - Extracts track metadata (artist, title, album, popularity)
  - Generates alias map for common corrections
  - Output: `inverted_index.json` (513MB), `track_metadata.json`, `alias_map.json`

- **candidate_generator.py**: Runtime candidate lookup with fuzzy matching
  - Trigram-based inverted index lookup (O(k) where k = # unique trigrams)
  - Fuzzy trigram expansion using keyboard proximity (QWERTY layout)
  - Skip-gram generation for transposition/deletion tolerance
  - Edit distance reranking on top 200 candidates
  - Latency: ~150-200ms with fuzzy enabled, ~50-100ms without

#### 7.2: ML Ranker (3 points) ✅
- **ml_ranker.py**: Pairwise ranking model (RankSVM)
- **Features (5 total)**:
  1. **Edit distance**: Normalized Levenshtein distance (0-1)
  2. **Token overlap**: Jaccard similarity on word tokens
  3. **Trigram overlap**: Character trigram Jaccard (from index)
  4. **Popularity score**: Log-scaled play count from MPD (0-1)
  5. **Playlist-context boost**: Co-occurrence score with current playlist tracks
- **Model**: Trained with pairwise ranking on ~7,000 query-track pairs
- **Fallback**: Heuristic scoring if model unavailable
- **Training**: `train_ranker_pairwise.py` + `generate_training_data.py`
- **Latency**: ~30-40ms for top 200 candidates

#### 7.3: UX Layer (2 points) ✅
- **autocorrect_ui.py**: User interface logic
  - High-confidence auto-fix: threshold = 0.85 (85% confidence)
  - "Did you mean..." display: Shows top 5 suggestions with confidence scores
  - Natural language selection support: "add the first one", "add 1 and 3"
  - Detailed timing breakdown for performance analysis

- **autocorrect_integration.py**: Integration with MusicCRS
  - **Singleton pattern**: Shared index across all user sessions (memory efficient)
  - **Lazy loading**: Loads 513MB index only on first autocorrect query
  - **Thread-safe initialization**: Uses threading.Lock() to prevent concurrent loads
  - **Loading callback**: Notifies frontend during 20-30 second initial load
  - **Graceful degradation**: Disables if index not found

### Integration Flow
```python
# In MusicCRS
autocorrect = AutocorrectIntegration(enabled=True, loading_callback=notify_fn)
result = autocorrect.correct_track_query("shpe of yu", current_playlist=["Starboy"])

if result.corrected:
    # Auto-fixed with high confidence (≥85%)
    add_track(result.track_uri, result.artist, result.title)
else:
    # Show suggestions to user
    display_suggestions(result.suggestions)  # [(uri, artist, title, confidence), ...]
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Build inverted index (run once)
python src/index_builder.py --input ../data/mpd.sqlite --output data/

```

## Usage Example

```python
from src.candidate_generator import CandidateGenerator
from src.ml_ranker import MLRanker
from src.autocorrect_ui import AutocorrectUI

# Initialize components
generator = CandidateGenerator("data/inverted_index.json", "data/alias_map.json")
ranker = MLRanker("data/track_metadata.json", model_path="data/ranker_model.pkl")
ui = AutocorrectUI(generator, ranker)

# Handle typo query
result = ui.autocorrect("bldning lights", current_playlist=["Shape of You", "Perfect"])

if result.auto_fixed:
    print(f"Added: {result.track}")
else:
    print("Did you mean:")
    for i, suggestion in enumerate(result.suggestions, 1):
        print(f"  {i}. {suggestion.track} - {suggestion.artist} ({suggestion.confidence:.2f})")
```
