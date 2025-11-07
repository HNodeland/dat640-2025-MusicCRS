"""
Generate high-quality training data for ML ranker using Ollama.

This script generates diverse, realistic training data by:
1. Using Ollama to simulate realistic user search queries (typos, abbreviations, partial names)
2. Stratified sampling across popularity ranges (not just popular tracks)
3. Windowed sampling from real playlists to capture co-occurrence patterns
4. Generating both positive and hard negative examples

Task: 7.2.5 - Generate improved training data for ML ranker

Key improvements over old approach:
- Realistic queries via Ollama (not just random typos)
- Balanced sampling across popularity ranges
- Playlist context from real user data
- Larger, more diverse dataset (5000+ examples)
- Hard negatives (similar tracks that are wrong)

Usage:
    python generate_training_data.py --samples 5000 --output ../data/training_data.json
    
    # Then train with:
    python train_ranker.py --data ../data/training_data.json --output ../data/ranker_model.pkl
"""

import argparse
import json
import random
import sys
import os
import re
import string
import time
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from candidate_generator import CandidateGenerator
from normalizer import normalize


def call_ollama(prompt: str, model: str = "llama3.3:70b", timeout: int = 60) -> Optional[str]:
    """
    Call Ollama API to generate text (uses UiS server with API key).
    
    Args:
        prompt: The prompt to send to Ollama
        model: The model to use (default: llama3.3:70b)
        timeout: Timeout in seconds (default: 60)
        
    Returns:
        Generated text or None if failed
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    ollama_host = os.getenv("OLLAMA_HOST", "https://ollama.ux.uis.no")
    ollama_api_key = os.getenv("OLLAMA_API_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL", model)
    
    if not ollama_api_key:
        print("Warning: OLLAMA_API_KEY not found in .env file")
        return None
    
    try:
        import ollama
        client = ollama.Client(
            host=ollama_host,
            headers={"Authorization": f"Bearer {ollama_api_key}"}
        )
        
        # Set a timeout for the request
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Ollama request timed out")
        
        # Set alarm for timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout)
        
        try:
            response = client.generate(
                model=ollama_model,
                prompt=prompt,
                options={"stream": False, "temperature": 0.7, "max_tokens": 150}
            )
            result = response.get("response", "").strip()
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel alarm
        
        return result
    except TimeoutError as e:
        print(f"  ⏱️  Timeout after {timeout}s")
        return None
    except Exception as e:
        print(f"  ❌ Ollama error: {e}")
        return None


def generate_realistic_queries_batch(
    track_texts: List[str], 
    num_variants: int = 2,
    checkpoint_file: Optional[str] = None
) -> Dict[str, List[str]]:
    """
    Use Ollama to generate realistic search queries for MULTIPLE tracks in ONE call.
    This drastically reduces API calls (1 call for 10 tracks instead of 10 calls).
    
    Supports checkpointing to resume after interruptions.
    
    Args:
        track_texts: List of "Artist - Title" strings
        num_variants: Number of query variants per track
        checkpoint_file: Path to save/load progress (optional)
        
    Returns:
        Dict mapping track_text -> list of generated queries
    """
    import datetime
    
    # Batch size: 10 tracks per API call
    batch_size = 10
    all_results = {}
    
    # Load checkpoint if exists
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  📂 Loading checkpoint from {checkpoint_file}...")
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data.get('queries', {})
                print(f"  ✓ Resumed with {len(all_results)} tracks already processed")
        except Exception as e:
            print(f"  ⚠️  Could not load checkpoint: {e}")
    
    total_batches = (len(track_texts) + batch_size - 1) // batch_size
    completed_batches = len(all_results) // batch_size
    
    print(f"  📊 Progress: {completed_batches}/{total_batches} batches completed")
    
    start_time = time.time()
    
    for batch_num, i in enumerate(range(0, len(track_texts), batch_size), start=1):
        batch = track_texts[i:i+batch_size]
        
        # Skip if all tracks in batch already processed
        if all(track in all_results for track in batch):
            continue
        
        # Progress indicator
        elapsed = time.time() - start_time
        batches_done = batch_num - 1
        if batches_done > 0:
            avg_time_per_batch = elapsed / batches_done
            eta_seconds = avg_time_per_batch * (total_batches - batches_done)
            eta_minutes = eta_seconds / 60
            print(f"  🔄 Batch {batch_num}/{total_batches} | Elapsed: {elapsed/60:.1f}m | ETA: {eta_minutes:.1f}m")
        else:
            print(f"  🔄 Batch {batch_num}/{total_batches} | Starting...")
        
        # Create batch prompt
        tracks_list = "\n".join([f"{idx+1}. {track}" for idx, track in enumerate(batch)])
        
        prompt = f"""Generate {num_variants} realistic search queries for EACH of these songs. Users make typos, abbreviations, or partial searches.

Songs:
{tracks_list}

For EACH song, generate {num_variants} realistic queries (one per line). Format:
1a) <query1>
1b) <query2>
2a) <query1>
2b) <query2>
...

Examples for "The Weeknd - Blinding Lights":
1a) bliding lights
1b) weeknd blinding

Generate queries now:"""

        try:
            response = call_ollama(prompt)
            
            if response:
                # Parse response
                lines = [line.strip() for line in response.split('\n') if line.strip()]
                
                # Map queries back to tracks
                current_track_idx = 0
                current_queries = []
                
                for line in lines:
                    # Match patterns like "1a)", "2b)", etc.
                    match = re.match(r'^(\d+)[a-z]\)\s*(.+)$', line, re.IGNORECASE)
                    if match:
                        track_num = int(match.group(1)) - 1
                        query = match.group(2).strip().strip('"').strip("'")
                        
                        if track_num != current_track_idx:
                            # Save previous track's queries
                            if current_queries and current_track_idx < len(batch):
                                all_results[batch[current_track_idx]] = current_queries[:num_variants]
                            current_track_idx = track_num
                            current_queries = []
                        
                        if query:
                            current_queries.append(query)
                
                # Save last track's queries
                if current_queries and current_track_idx < len(batch):
                    all_results[batch[current_track_idx]] = current_queries[:num_variants]
        except Exception as e:
            print(f"  ⚠️  Batch {batch_num} error: {e}")
        
        # Fallback for tracks that didn't get queries
        for track in batch:
            if track not in all_results or not all_results[track]:
                all_results[track] = generate_fallback_queries(track, num_variants)
        
        # Save checkpoint every 5 batches
        if checkpoint_file and batch_num % 5 == 0:
            try:
                with open(checkpoint_file, 'w') as f:
                    json.dump({
                        'queries': all_results,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'progress': f"{batch_num}/{total_batches}"
                    }, f, indent=2)
                print(f"  💾 Checkpoint saved ({len(all_results)} tracks)")
            except Exception as e:
                print(f"  ⚠️  Could not save checkpoint: {e}")
        
        # Rate limiting: small delay between batches
        if i + batch_size < len(track_texts):
            time.sleep(0.5)  # 500ms between batches
    
    # Final save
    if checkpoint_file:
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    'queries': all_results,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'progress': 'complete'
                }, f, indent=2)
            print(f"  ✓ Final checkpoint saved")
        except Exception as e:
            print(f"  ⚠️  Could not save final checkpoint: {e}")
    
    return all_results


def generate_realistic_queries(track_text: str, num_variants: int = 3) -> List[str]:
    """
    Use Ollama to generate realistic search queries for a track.
    
    DEPRECATED: Use generate_realistic_queries_batch() for better efficiency.
    This function is kept for single-track fallback.
    
    Generates queries with:
    - Common misspellings
    - Abbreviations
    - Partial names (just artist or title)
    - Phonetic variations
    
    Args:
        track_text: "Artist - Title" format
        num_variants: Number of query variants to generate
        
    Returns:
        List of generated query strings
    """
    prompt = f"""Generate {num_variants} realistic search queries a user might type when looking for this song: "{track_text}"

Include variations like:
- Common misspellings (typing errors, missing letters)
- Abbreviations (shortened words)
- Partial queries (just artist or just title)
- Phonetic spellings (how it sounds)

Output ONLY the queries, one per line, no explanations.

Examples for "The Weeknd - Blinding Lights":
bliding lights
weeknd blinding
blnding lights

Now generate {num_variants} queries for: {track_text}"""

    response = call_ollama(prompt)
    
    if response:
        # Parse response lines
        queries = [line.strip() for line in response.split('\n') if line.strip()]
        # Clean up any extra formatting
        queries = [q.strip('- ').strip('"').strip("'").strip(')').strip(string.digits + '.)') for q in queries]
        # Filter out the original track text or very long queries
        queries = [q for q in queries if q.lower() != track_text.lower() and len(q) < 100 and len(q) > 2]
        return queries[:num_variants]
    
    # Fallback: generate simple typos if Ollama fails
    print("  Ollama failed, using fallback typo generation")
    return generate_fallback_queries(track_text, num_variants)


def generate_typo(text: str, typo_rate: float = 0.2) -> str:
    """
    Generate a typo-ridden version of text.
    
    Typo types:
    - Character deletion (25%)
    - Character substitution (25%)
    - Character transposition (25%)
    - Character insertion (25%)
    
    Args:
        text: Original text
        typo_rate: Probability of introducing a typo per character
        
    Returns:
        Text with typos
    """
    if not text or len(text) < 3:
        return text
    
    chars = list(text)
    n = len(chars)
    
    # Number of typos to introduce (at least 1, max 3)
    num_typos = random.randint(1, min(3, max(1, int(n * typo_rate))))
    
    for _ in range(num_typos):
        pos = random.randint(0, n - 1)
        typo_type = random.choice(['delete', 'substitute', 'transpose', 'insert'])
        
        if typo_type == 'delete' and n > 3:
            # Delete a character
            chars.pop(pos)
            n -= 1
        
        elif typo_type == 'substitute' and chars[pos].isalpha():
            # Substitute with nearby key
            nearby = 'abcdefghijklmnopqrstuvwxyz'
            chars[pos] = random.choice(nearby)
        
        elif typo_type == 'transpose' and pos < n - 1:
            # Swap adjacent characters
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        elif typo_type == 'insert' and chars[pos].isalpha():
            # Insert random character
            nearby = 'abcdefghijklmnopqrstuvwxyz'
            chars.insert(pos, random.choice(nearby))
            n += 1
    
    return ''.join(chars)


def generate_fallback_queries(track_text: str, num_variants: int = 3) -> List[str]:
    """
    Fallback query generation if Ollama is unavailable.
    Generates simple character-level typos.
    """
    normalized = normalize(track_text)
    queries = []
    
    for _ in range(num_variants * 2):  # Generate extra, filter later
        query = generate_typo(normalized, typo_rate=0.15)
        if query != normalized and query not in queries:
            queries.append(query)
        if len(queries) >= num_variants:
            break
    
    return queries


def load_playlist_data(data_dir: str, max_playlists: int = 1000) -> List[List[str]]:
    """
    Load playlist data for context-aware sampling.
    
    Args:
        data_dir: Path to MPD data directory
        max_playlists: Maximum number of playlists to load
        
    Returns:
        List of playlists, where each playlist is a list of track URIs
    """
    print(f"Loading playlist data from {data_dir}...")
    
    import glob
    
    playlists = []
    slice_files = glob.glob(os.path.join(data_dir, "mpd.slice.*.json"))
    
    # Load first few slice files
    for slice_file in sorted(slice_files)[:5]:  # First 5 slices = 5000 playlists
        try:
            with open(slice_file, 'r') as f:
                data = json.load(f)
                for playlist in data.get('playlists', []):
                    track_uris = [track['track_uri'] for track in playlist.get('tracks', [])]
                    if len(track_uris) >= 5:  # Only playlists with 5+ tracks
                        playlists.append(track_uris)
                    
                    if len(playlists) >= max_playlists:
                        break
            
            if len(playlists) >= max_playlists:
                break
        except Exception as e:
            print(f"  Error loading {slice_file}: {e}")
            continue
    
    print(f"  Loaded {len(playlists)} playlists")
    return playlists


def stratified_track_sampling(
    metadata: Dict,
    num_samples: int,
    popularity_bins: List[Tuple[int, int]] = [(0, 10), (10, 100), (100, 1000), (1000, 50000)]
) -> List[Tuple[int, Dict]]:
    """
    Sample tracks stratified by popularity to avoid bias.
    
    Args:
        metadata: Track metadata dict
        num_samples: Total number of tracks to sample
        popularity_bins: List of (min_pop, max_pop) tuples
        
    Returns:
        List of (track_id, metadata) tuples
    """
    print("Stratified sampling of tracks...")
    
    # Group tracks by popularity bin
    bins = defaultdict(list)
    for track_id, meta in metadata.items():
        popularity = meta.get('popularity', 0)
        for bin_idx, (min_pop, max_pop) in enumerate(popularity_bins):
            if min_pop <= popularity < max_pop:
                bins[bin_idx].append((int(track_id), meta))
                break
    
    # Print bin statistics
    for bin_idx, (min_pop, max_pop) in enumerate(popularity_bins):
        print(f"  Bin {bin_idx} ({min_pop}-{max_pop}): {len(bins[bin_idx]):,} tracks")
    
    # Sample proportionally from each bin
    samples_per_bin = num_samples // len(popularity_bins)
    sampled_tracks = []
    
    for bin_idx in range(len(popularity_bins)):
        bin_tracks = bins[bin_idx]
        if len(bin_tracks) > samples_per_bin:
            sampled = random.sample(bin_tracks, samples_per_bin)
        else:
            sampled = bin_tracks
        sampled_tracks.extend(sampled)
    
    # Shuffle
    random.shuffle(sampled_tracks)
    
    print(f"  Sampled {len(sampled_tracks)} tracks across {len(popularity_bins)} popularity bins")
    return sampled_tracks[:num_samples]


def extract_playlist_windows(
    playlists: List[List[str]],
    window_size: int = 10
) -> Dict[str, List[str]]:
    """
    Extract co-occurrence context from playlists using sliding windows.
    
    Args:
        playlists: List of playlists (each is list of track URIs)
        window_size: Size of sliding window
        
    Returns:
        Dict mapping track_uri -> list of co-occurring track URIs
    """
    print("Extracting playlist co-occurrence windows...")
    
    cooccurrence = defaultdict(set)
    
    for playlist in playlists:
        # Slide window over playlist
        for i in range(len(playlist)):
            track_uri = playlist[i]
            # Get context window (tracks before and after)
            window_start = max(0, i - window_size // 2)
            window_end = min(len(playlist), i + window_size // 2 + 1)
            context_tracks = playlist[window_start:window_end]
            
            # Add co-occurring tracks
            for context_uri in context_tracks:
                if context_uri != track_uri:
                    cooccurrence[track_uri].add(context_uri)
    
    # Convert sets to lists
    cooccurrence_dict = {k: list(v) for k, v in cooccurrence.items()}
    
    print(f"  Built co-occurrence map for {len(cooccurrence_dict):,} tracks")
    return cooccurrence_dict


def generate_training_examples(
    generator: CandidateGenerator,
    sampled_tracks: List[Tuple[int, Dict]],
    cooccurrence_map: Dict[str, List[str]],
    num_queries_per_track: int = 2,
    use_ollama: bool = True
) -> List[Dict]:
    """
    Generate training examples with features and labels.
    
    For each sampled track:
    1. Generate realistic queries (via Ollama BATCHED or fallback)
    2. Get candidates from generator
    3. Extract features for all candidates
    4. Label: original track = positive (1), others = negative (0)
    5. Add playlist context if available
    
    Args:
        generator: CandidateGenerator instance
        sampled_tracks: List of (track_id, metadata) tuples
        cooccurrence_map: Track URI -> co-occurring track URIs
        num_queries_per_track: Number of query variants per track
        use_ollama: Whether to use Ollama for query generation
        
    Returns:
        List of training example dicts with features and labels
    """
    print(f"\nGenerating training examples...")
    print(f"  Tracks: {len(sampled_tracks):,}")
    print(f"  Queries per track: {num_queries_per_track}")
    print(f"  Using Ollama: {use_ollama}")
    if use_ollama:
        print(f"  → Batching enabled: ~{len(sampled_tracks) // 10} API calls instead of {len(sampled_tracks):,}")
    print()
    
    training_examples = []
    successful_tracks = 0
    
    # BATCH QUERY GENERATION (much more efficient!)
    if use_ollama:
        print("Generating queries with Ollama (batched for efficiency)...")
        track_texts = [f"{meta['artist']} - {meta['title']}" for _, meta in sampled_tracks]
        
        # Use checkpoint file for resumability
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "query_checkpoint.json")
        
        queries_map = generate_realistic_queries_batch(
            track_texts, 
            num_queries_per_track,
            checkpoint_file=checkpoint_path
        )
        print(f"  ✓ Generated queries for {len(queries_map)}/{len(track_texts)} tracks via Ollama")
    else:
        queries_map = {}
    
    # Generate training examples
    print("\nProcessing candidates...")
    for idx, (track_id, metadata) in enumerate(sampled_tracks):
        if idx > 0 and idx % 100 == 0:
            print(f"  Progress: {idx}/{len(sampled_tracks)} tracks, {len(training_examples)} examples generated")
        
        track_text = f"{metadata['artist']} - {metadata['title']}"
        track_uri = metadata['uri']
        
        # Get playlist context if available
        playlist_context = None
        if track_uri in cooccurrence_map:
            context_uris = cooccurrence_map[track_uri]
            if context_uris:
                # Sample up to 10 context tracks
                playlist_context = random.sample(context_uris, min(10, len(context_uris)))
        
        # Get query variants (from batch or fallback)
        if use_ollama and track_text in queries_map:
            queries = queries_map[track_text]
        elif use_ollama:
            # Ollama batch missed this track, use fallback
            queries = generate_fallback_queries(track_text, num_queries_per_track)
        else:
            queries = generate_fallback_queries(track_text, num_queries_per_track)
        
        if not queries:
            continue
        
        # For each query variant
        for query in queries:
            # Get candidates
            try:
                candidates = generator.generate_candidates(
                    query,
                    threshold=0.15,  # Lower threshold to get more candidates
                    limit=30  # Get more candidates for hard negatives
                )
            except Exception as e:
                continue
            
            if not candidates:
                continue
            
            # Check if original track is in candidates
            original_found = any(cand_id == track_id for cand_id, _, _ in candidates)
            
            if not original_found:
                # Original not found - skip this query
                continue
            
            # Create training examples for all candidates
            for cand_id, trigram_score, cand_meta in candidates:
                example = {
                    'query': query,
                    'track_id': cand_id,
                    'original_track_id': track_id,
                    'trigram_score': trigram_score,
                    'is_correct': (cand_id == track_id),
                    'playlist_context': playlist_context,
                    # Store metadata for feature extraction later
                    'metadata': cand_meta
                }
                training_examples.append(example)
            
            successful_tracks += 1
            break  # Move to next track after successful query
    
    print(f"\n✓ Generated {len(training_examples)} training examples from {successful_tracks} tracks")
    
    # Print label distribution
    num_positive = sum(1 for ex in training_examples if ex['is_correct'])
    num_negative = len(training_examples) - num_positive
    print(f"  Positive examples: {num_positive}")
    print(f"  Negative examples: {num_negative}")
    print(f"  Ratio: 1:{num_negative/max(num_positive, 1):.1f}")
    
    return training_examples


def main():
    """Main training data generation pipeline."""
    parser = argparse.ArgumentParser(description='Generate training data for ML ranker')
    parser.add_argument('--samples', type=int, default=5000, 
                        help='Number of tracks to sample (default: 5000, recommended for 2M+ track database)')
    parser.add_argument('--queries-per-track', type=int, default=2,
                        help='Number of query variants per track (default: 2)')
    parser.add_argument('--use-ollama', action='store_true', default=False,
                        help='Use Ollama API for realistic query generation (BATCHED: ~500 API calls for 5000 tracks)')
    parser.add_argument('--no-ollama', action='store_true',
                        help='Disable Ollama, use fallback typo generation (faster, less realistic)')
    parser.add_argument('--output', default='../data/training_data.json',
                        help='Output file path (default: ../data/training_data.json)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    if args.no_ollama:
        args.use_ollama = False
    
    random.seed(args.seed)
    
    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(base_dir, "data")
    mpd_data_dir = os.path.join(os.path.dirname(base_dir), "data", "data")
    
    index_path = os.path.join(data_dir, "inverted_index.json")
    alias_path = os.path.join(data_dir, "alias_map.json")
    metadata_path = os.path.join(data_dir, "track_metadata.json")
    output_path = os.path.join(base_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    print("=" * 70)
    print("TRAINING DATA GENERATION - ML RANKER")
    print("=" * 70)
    print()
    print(f"Configuration:")
    print(f"  Tracks to sample: {args.samples:,}")
    print(f"  Queries per track: {args.queries_per_track}")
    print(f"  Use Ollama: {args.use_ollama}")
    if not args.use_ollama:
        print(f"  → Using fallback typo generation (faster, less realistic)")
    else:
        print(f"  → Using Ollama (slower, more realistic)")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {output_path}")
    print()
    
    # Load components
    print("Loading candidate generator...")
    generator = CandidateGenerator(index_path, alias_path, metadata_path)
    
    print("\nLoading track metadata...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load playlist data for context
    playlists = []
    if os.path.exists(mpd_data_dir):
        playlists = load_playlist_data(mpd_data_dir, max_playlists=1000)
    else:
        print(f"Warning: Playlist data directory not found: {mpd_data_dir}")
        print("  Continuing without playlist context...")
    
    # Extract co-occurrence patterns
    cooccurrence_map = {}
    if playlists:
        cooccurrence_map = extract_playlist_windows(playlists, window_size=10)
    
    # Stratified sampling of tracks
    sampled_tracks = stratified_track_sampling(
        metadata,
        num_samples=args.samples,
        popularity_bins=[(0, 10), (10, 100), (100, 1000), (1000, 50000)]
    )
    
    # Generate training examples
    training_examples = generate_training_examples(
        generator,
        sampled_tracks,
        cooccurrence_map,
        num_queries_per_track=args.queries_per_track,
        use_ollama=args.use_ollama
    )
    
    if not training_examples:
        print("\nError: No training examples generated. Exiting.")
        return
    
    # Save to JSON
    print(f"\nSaving training data to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump({
            'examples': training_examples,
            'metadata': {
                'num_examples': len(training_examples),
                'num_positive': sum(1 for ex in training_examples if ex['is_correct']),
                'num_negative': sum(1 for ex in training_examples if not ex['is_correct']),
                'num_tracks': len(sampled_tracks),
                'queries_per_track': args.queries_per_track,
                'used_ollama': args.use_ollama,
                'seed': args.seed,
                'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
            }
        }, f, indent=2)
    
    file_size = os.path.getsize(output_path)
    print(f"✓ Training data saved! Size: {file_size:,} bytes ({file_size/(1024*1024):.1f} MB)")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print(f"1. Review the generated data: less {output_path}")
    print(f"2. Train the model:")
    print(f"   python train_ranker.py --data {output_path} --output ../data/ranker_model.pkl")
    print()


if __name__ == "__main__":
    main()
