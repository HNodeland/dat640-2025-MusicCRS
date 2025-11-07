"""
Inverted index builder for R7 autocorrect.

This module builds a pre-computed n-gram inverted index from the Million Playlist
Dataset. The index maps character trigrams to track IDs for fast fuzzy matching.

Task: 7.1.3 - Implement build_inverted_index() function

Architecture:
    Offline (build time):
        1. Load tracks from MPD SQLite database
        2. For each track: normalize(title + artist) → extract trigrams
        3. Build inverted index: trigram → [track_id_1, track_id_2, ...]
        4. Serialize to JSON for fast loading

    Index structure:
        {
            "hel": [1, 45, 892, ...],    # tracks containing "hel" trigram
            "ell": [1, 203, 445, ...],
            "llo": [1, 999, ...],
            ...
        }

Usage:
    python index_builder.py --input ../data/mpd.sqlite --output data/
"""

import sqlite3
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Set
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from normalizer import normalize, extract_trigrams, extract_bigrams


class IndexBuilder:
    """Build inverted index from track database."""
    
    def __init__(self, db_path: str, use_bigrams: bool = True):
        """
        Initialize index builder.
        
        Args:
            db_path: Path to MPD SQLite database
            use_bigrams: Whether to include bigrams in addition to trigrams
        """
        self.db_path = db_path
        self.use_bigrams = use_bigrams
        self.inverted_index: Dict[str, List[int]] = defaultdict(list)
        self.track_metadata: Dict[int, Dict] = {}
        self.track_id_counter = 0
    
    def load_tracks(self) -> List[Tuple[str, str, str, int]]:
        """
        Load tracks from database.
        
        Returns:
            List of (uri, artist, title, popularity) tuples
        """
        print(f"Loading tracks from {self.db_path}...")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = """
            SELECT track_uri, artist, title, COALESCE(popularity, 0) as popularity
            FROM tracks
            WHERE artist IS NOT NULL AND title IS NOT NULL
            ORDER BY popularity DESC
        """
        
        cursor = conn.execute(query)
        tracks = [(row['track_uri'], row['artist'], row['title'], row['popularity']) 
                  for row in cursor.fetchall()]
        
        conn.close()
        print(f"Loaded {len(tracks):,} tracks")
        return tracks
    
    def build_index(self, tracks: List[Tuple[str, str, str, int]]) -> None:
        """
        Build inverted index from tracks.
        
        For each track:
        1. Create combined text: "artist title"
        2. Normalize text
        3. Extract bigrams and trigrams
        4. Add n-gram → track_id mappings
        
        Args:
            tracks: List of (uri, artist, title, popularity) tuples
        """
        print(f"Building inverted index (bigrams: {self.use_bigrams}, trigrams: True)...")
        
        for uri, artist, title, popularity in tqdm(tracks, desc="Indexing"):
            track_id = self.track_id_counter
            self.track_id_counter += 1
            
            # Store metadata
            self.track_metadata[track_id] = {
                'uri': uri,
                'artist': artist,
                'title': title,
                'popularity': popularity,
                'normalized': normalize(f"{artist} {title}")
            }
            
            # Extract n-grams from normalized text
            combined_text = f"{artist} {title}"
            ngrams = []
            
            # Add trigrams
            ngrams.extend(extract_trigrams(combined_text, normalize_first=True))
            
            # Add bigrams if enabled
            if self.use_bigrams:
                ngrams.extend(extract_bigrams(combined_text, normalize_first=True))
            
            # Add to inverted index
            for ngram in set(ngrams):  # Use set to avoid duplicate entries per track
                self.inverted_index[ngram].append(track_id)
        
        print(f"Built index with {len(self.inverted_index):,} unique n-grams")
        print(f"  - Bigrams: {'included' if self.use_bigrams else 'excluded'}")
        print(f"Indexed {len(self.track_metadata):,} tracks")
    
    def calculate_statistics(self) -> Dict:
        """
        Calculate index statistics.
        
        Returns:
            Dictionary of statistics
        """
        ngram_counts = [len(track_ids) for track_ids in self.inverted_index.values()]
        
        return {
            'total_ngrams': len(self.inverted_index),
            'total_tracks': len(self.track_metadata),
            'avg_tracks_per_ngram': sum(ngram_counts) / len(ngram_counts) if ngram_counts else 0,
            'max_tracks_per_ngram': max(ngram_counts) if ngram_counts else 0,
            'min_tracks_per_ngram': min(ngram_counts) if ngram_counts else 0,
        }
    
    def save(self, output_dir: str) -> None:
        """
        Save inverted index and metadata to JSON files.
        
        Args:
            output_dir: Directory to save files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save inverted index
        index_path = os.path.join(output_dir, 'inverted_index.json')
        print(f"Saving inverted index to {index_path}...")
        with open(index_path, 'w') as f:
            json.dump(dict(self.inverted_index), f)
        print(f"Index saved ({os.path.getsize(index_path) / 1024 / 1024:.1f} MB)")
        
        # Save track metadata
        metadata_path = os.path.join(output_dir, 'track_metadata.json')
        print(f"Saving track metadata to {metadata_path}...")
        with open(metadata_path, 'w') as f:
            json.dump({str(k): v for k, v in self.track_metadata.items()}, f)
        print(f"Metadata saved ({os.path.getsize(metadata_path) / 1024 / 1024:.1f} MB)")
        
        # Save statistics
        stats = self.calculate_statistics()
        stats_path = os.path.join(output_dir, 'index_stats.json')
        print(f"Saving statistics to {stats_path}...")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print("\n=== Index Statistics ===")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value:,}")


def create_default_alias_map(output_dir: str) -> None:
    """
    Create a default alias map with common corrections.
    
    Args:
        output_dir: Directory to save alias map
    """
    alias_map = {
        # Common artist misspellings
        "weekend": "the weeknd",
        "weeknd": "the weeknd",
        "drake": "drake",
        "adel": "adele",
        "beyonce": "beyoncé",
        "taylor": "taylor swift",
        "ed sheeran": "ed sheeran",
        "bieber": "justin bieber",
        "ariana": "ariana grande",
        "billie": "billie eilish",
        
        # Common title patterns
        "dont": "don't",
        "cant": "can't",
        "wont": "won't",
        "im": "i'm",
        "youre": "you're",
        "theyre": "they're",
    }
    
    alias_path = os.path.join(output_dir, 'alias_map.json')
    print(f"Creating default alias map at {alias_path}...")
    with open(alias_path, 'w') as f:
        json.dump(alias_map, f, indent=2)
    print(f"Alias map created with {len(alias_map)} entries")


def main():
    """Main entry point for index builder."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Build inverted index for autocorrect')
    parser.add_argument('--input', default='../data/mpd.sqlite', help='Path to MPD SQLite database')
    parser.add_argument('--output', default='data/', help='Output directory for index files')
    parser.add_argument('--limit', type=int, help='Limit number of tracks (for testing)')
    parser.add_argument('--bigrams', action='store_true', default=True, help='Include bigrams (default: True)')
    parser.add_argument('--no-bigrams', dest='bigrams', action='store_false', help='Exclude bigrams')
    
    args = parser.parse_args()
    
    # Check if input database exists
    if not os.path.exists(args.input):
        print(f"Error: Database not found at {args.input}")
        print("Please ensure MPD SQLite database is available.")
        print("Expected location: ../data/mpd.sqlite")
        sys.exit(1)
    
    # Build index
    print(f"Building index with bigrams: {args.bigrams}")
    builder = IndexBuilder(args.input, use_bigrams=args.bigrams)
    tracks = builder.load_tracks()
    
    if args.limit:
        print(f"Limiting to first {args.limit} tracks for testing")
        tracks = tracks[:args.limit]
    
    builder.build_index(tracks)
    builder.save(args.output)
    
    # Create default alias map
    create_default_alias_map(args.output)
    
    print("\n✓ Index building complete!")
    print(f"  Index: {args.output}/inverted_index.json")
    print(f"  Metadata: {args.output}/track_metadata.json")
    print(f"  Aliases: {args.output}/alias_map.json")


if __name__ == "__main__":
    main()
