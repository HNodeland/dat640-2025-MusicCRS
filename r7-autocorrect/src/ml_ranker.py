"""
ML-based ranking for R7 autocorrect candidates.

This module implements a machine learning ranker that scores candidates using
multiple features: edit distance, token overlap, popularity, and playlist context.

Task: 7.2 - ML ranker combining edit distance, overlap, popularity, context (3 points)

Features:
    1. Edit distance (Levenshtein) - character-level similarity
    2. Token overlap (Jaccard) - word-level similarity
    3. Trigram overlap - character trigram similarity (from index)
    4. Popularity score - from Million Playlist Dataset
    5. Playlist-context score - boost tracks that co-occur with current playlist

Model: Pairwise ranking (RankSVM) - learns to order candidates by relevance

Usage:
    ranker = MLRanker("data/track_metadata.json", model_path="data/ranker_model.pkl")
    top_tracks = ranker.rank_candidates(candidates, query="blinding lights", 
                                        current_playlist=["Shape of You"])
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import Levenshtein
import numpy as np

import json
import pickle
import os
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from Levenshtein import distance as levenshtein_distance
except ImportError:
    print("Warning: python-Levenshtein not installed. Using fallback edit distance.")
    def levenshtein_distance(s1, s2):
        """Fallback edit distance implementation."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

from normalizer import normalize, extract_trigrams_set, extract_bigrams


class MLRanker:
    """ML-based ranker for autocorrect candidates."""
    
    def __init__(self, metadata_path: str, model_path: Optional[str] = None):
        """
        Initialize ML ranker.
        
        Args:
            metadata_path: Path to track_metadata.json
            model_path: Path to trained model (DEPRECATED - using heuristic instead)
        """
        self.metadata_path = metadata_path
        self.model_path = model_path
        
        print(f"Loading track metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            metadata_raw = json.load(f)
            self.track_metadata = {int(k): v for k, v in metadata_raw.items()}
        
        # Try to load ML model (pairwise ranking)
        self.model = None
        self.scaler = None
        self.model_type = 'heuristic'
        
        if model_path and Path(model_path).exists():
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.model_type = model_data.get('model_type', 'binary_classification')
                    
                    if self.model is not None and self.scaler is not None:
                        print(f"Loaded ML ranking model: {self.model_type}")
                    else:
                        print("Model file incomplete, using heuristic")
                        self.model = None
            except Exception as e:
                print(f"Failed to load model: {e}, using heuristic")
                self.model = None
        else:
            print("No model file found, using optimized heuristic scoring")
    
    def extract_features(
        self, 
        query: str, 
        track_id: int,
        trigram_score: float,
        current_playlist: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Extract features for ranking.
        
        Features (5 total - simplified, no bigrams):
        1. Edit distance (normalized by max length)
        2. Token overlap (Jaccard similarity at word level)
        3. Trigram overlap (character trigram Jaccard)
        4. Popularity score (log-normalized)
        5. Playlist-context score (co-occurrence with current playlist)
        
        Args:
            query: User's search query
            track_id: Candidate track ID
            trigram_score: Trigram overlap score from candidate generation
            current_playlist: List of track URIs currently in playlist
            
        Returns:
            Dictionary of feature name → value
        """
        metadata = self.track_metadata[track_id]
        
        # Normalize query and track text
        query_norm = normalize(query)
        track_text = f"{metadata['artist']} {metadata['title']}"
        track_norm = normalize(track_text)
        
        # Feature 1: Normalized edit distance
        edit_dist = levenshtein_distance(query_norm, track_norm)
        max_len = max(len(query_norm), len(track_norm))
        edit_distance_norm = 1.0 - (edit_dist / max_len if max_len > 0 else 0)
        
        # Feature 2: Token overlap (word-level Jaccard)
        query_tokens = set(query_norm.split())
        track_tokens = set(track_norm.split())
        
        if query_tokens and track_tokens:
            token_intersection = len(query_tokens & track_tokens)
            token_union = len(query_tokens | track_tokens)
            token_overlap = token_intersection / token_union if token_union > 0 else 0.0
        else:
            token_overlap = 0.0
        
        # Feature 3: Trigram overlap (USE the score from candidate generator!)
        # The candidate generator already computed this efficiently using the inverted index
        trigram_overlap = trigram_score
        
        # Feature 4: Popularity (log-normalized - MUST match training!)
        popularity = metadata.get('popularity', 0)
        # Use log scale: log(1 + pop) / log(1 + max_pop)
        # Assume max popularity ~30,000
        popularity_norm = np.log(1 + popularity) / np.log(1 + 30000)
        
        # Store raw popularity for hybrid ranking
        raw_popularity = popularity
        
        # Feature 5: Playlist-context score (co-occurrence)
        context_score = 0.0
        if current_playlist and len(current_playlist) > 0:
            # TODO: Implement co-occurrence matrix lookup
            # For now, just check if same artist is in playlist
            track_artist_norm = normalize(metadata['artist'])
            for playlist_uri in current_playlist[:10]:  # Check first 10 tracks
                # Extract track ID from URI
                try:
                    playlist_track_id = int(playlist_uri.split(':')[-1]) if ':' in playlist_uri else None
                    if playlist_track_id and playlist_track_id in self.track_metadata:
                        playlist_artist_norm = normalize(self.track_metadata[playlist_track_id]['artist'])
                        if track_artist_norm == playlist_artist_norm:
                            context_score += 0.1  # Boost if same artist
                except:
                    pass
            context_score = min(1.0, context_score)
        
        return {
            'edit_distance': edit_distance_norm,
            'token_overlap': token_overlap,
            'trigram_overlap': trigram_overlap,
            'popularity': popularity_norm,
            'context_score': context_score,
            'raw_popularity': raw_popularity,  # For hybrid ranking
        }
    
    def rank_candidates(
        self, 
        candidates: List[Tuple[int, float, Dict]], 
        query: str,
        current_playlist: Optional[List[str]] = None,
        limit: int = 5
    ) -> List[Tuple[int, float, Dict]]:
        """
        Rank candidates using ML model or heuristic scoring.
        
        Args:
            candidates: List of (track_id, trigram_score, metadata) from generator
            query: User's search query
            current_playlist: List of track URIs in current playlist
            limit: Number of top results to return
            
        Returns:
            List of (track_id, final_score, metadata) tuples, sorted by score descending
        """
        import time
        
        if not candidates:
            return []
        
        timings = {}
        start_time = time.time()
        
        # Feature extraction phase
        t0 = time.time()
        ranked = []
        
        for track_id, trigram_score, metadata in candidates:
            # Extract features
            features = self.extract_features(query, track_id, trigram_score, current_playlist)
            
            # Score using model or heuristic
            if self.model and self.scaler and self.model_type == 'pairwise_ranking':
                # Use pairwise ranking model (RankSVM)
                # Model predicts relevance score directly
                feature_vector = np.array([
                    features['edit_distance'],
                    features['token_overlap'],
                    features['trigram_overlap'],
                    features['popularity'],
                    features['context_score']
                ]).reshape(1, -1)
                
                # Scale features
                feature_vector_scaled = self.scaler.transform(feature_vector)
                
                # Get ranking score (decision function = relevance)
                ml_score = self.model.decision_function(feature_vector_scaled)[0]
                
                # POPULARITY ADJUSTMENT (post-training hack to fix training data bias)
                # 
                # THE PROBLEM:
                # - Training data has high trigram overlap → correct (by design)
                # - But real queries have: obscure exact match vs popular partial match
                # - Example: "hello" → "Нюша - Hello" (pop=1, exact) beats "Adele - Hello" (pop=17k, partial)
                # 
                # THE TRADE-OFF:
                # Option 1: Accept it - users can select from suggestions
                # Option 2: Boost popularity (current) - helps mainstream tracks win
                # Option 3: Retrain model with more popularity weight in training
                #
                # Currently using Option 2 with moderate boost.
                # To retrain (Option 3): Increase popularity feature weight or
                # generate training data with more popularity-based examples.
                #
                raw_pop = features['raw_popularity']
                
                # Moderate popularity boost (compromise between model and user expectations)
                if raw_pop > 10000:  # Mega-hits (top 0.1%)
                    popularity_boost = np.log1p(raw_pop) / 3.0
                elif raw_pop > 2000:  # Very popular (top 1%)
                    popularity_boost = np.log1p(raw_pop) / 5.0
                elif raw_pop > 500:  # Popular (top 5%)
                    popularity_boost = np.log1p(raw_pop) / 10.0
                else:
                    popularity_boost = 0.0
                
                raw_score = ml_score + popularity_boost
                
                # NORMALIZE to 0-1 using sigmoid for user-friendly confidence percentages
                # Sigmoid: 1 / (1 + exp(-x))
                # Maps raw scores (typically -5 to +20) to probabilities (0 to 1)
                # Shift by 5 so score=5 → 50% confidence
                shifted_score = raw_score - 5.0
                final_score = 1.0 / (1.0 + np.exp(-shifted_score))
                
                # Store raw score for debugging
                features['raw_ml_score'] = raw_score
                    
            elif self.model and self.model_type == 'binary_classification':
                # Legacy binary classification model
                feature_vector = np.array([
                    features['edit_distance'],
                    features['token_overlap'],
                    features['trigram_overlap'],
                    features['popularity'],
                    features['context_score']
                ]).reshape(1, -1)
                
                final_score = self.model.predict_proba(feature_vector)[0, 1]
            else:
                # OPTIMIZED HEURISTIC: Based on what actually works
                # Trigram overlap is KING (from inverted index)
                # Then exact matches matter
                # Popularity breaks ties
                final_score = (
                    0.50 * features['trigram_overlap'] +      # Most important!
                    0.20 * features['edit_distance'] +        # Exact/close matches
                    0.15 * features['token_overlap'] +        # Word-level matches
                    0.10 * features['popularity'] +           # Popularity breaks ties
                    0.05 * features['context_score']          # Playlist context (minor)
                )
            
            ranked.append((track_id, final_score, metadata))
        
        timings['feature_extraction'] = time.time() - t0
        
        # Sort by final score (descending)
        t0 = time.time()
        ranked.sort(key=lambda x: x[1], reverse=True)
        timings['sort'] = time.time() - t0
        
        timings['total'] = time.time() - start_time
        
        # Store timings for retrieval
        self._last_timings = timings
        
        return ranked[:limit]
    
    def get_last_timings(self) -> Dict[str, float]:
        """
        Get timing breakdown from the last rank_candidates call.
        
        Returns:
            Dictionary with timing information (in seconds) for each stage:
                - feature_extraction: Time spent extracting features and scoring
                - sort: Time spent sorting candidates by score
                - total: Total time for ranking
        """
        return getattr(self, '_last_timings', {})
    
    def train_model(self, training_data: List[Tuple[Dict, bool]]):
        """
        Train ML ranking model.
        
        TODO (Task 7.2.5): Implement model training
        
        Args:
            training_data: List of (features, is_correct) pairs
        """
        from sklearn.linear_model import LogisticRegression
        
        X = []
        y = []
        
        for features, is_correct in training_data:
            feature_vector = [
                features['edit_distance'],
                features['token_overlap'],
                features['popularity'],
                features['context_score'],
                features['trigram_score']
            ]
            X.append(feature_vector)
            y.append(1 if is_correct else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Train logistic regression
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X, y)
        
        print(f"Trained model with {len(training_data)} examples")
        print(f"  Training accuracy: {self.model.score(X, y):.3f}")
    
    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        print(f"Model saved to {path}")


def test_ranker():
    """Test ML ranker with sample data."""
    
    metadata_path = "data/track_metadata.json"
    
    if not os.path.exists(metadata_path):
        print("Error: Metadata file not found. Run index_builder.py first.")
        return
    
    ranker = MLRanker(metadata_path)
    
    # Sample candidates (track_id, trigram_score, metadata)
    # These would come from candidate_generator in practice
    print("\n=== ML Ranker Test ===\n")
    print("Note: This test requires actual track data from the index.")
    print("Run index_builder.py and candidate_generator.py first to generate test data.")


if __name__ == "__main__":
    test_ranker()
