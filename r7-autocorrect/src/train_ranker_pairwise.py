#!/usr/bin/env python3
"""
Train a learning-to-rank model using pairwise ranking approach.

This script implements pairwise ranking (RankNet-style) where the model learns
to predict relative relevance between pairs of candidates, not absolute correctness.

Key differences from binary classification:
1. Generates PAIRS of candidates for each query
2. Model learns to predict which of two candidates is MORE relevant
3. Uses pairwise hinge loss or RankSVM
4. Much better for ranking tasks!

Usage:
    python train_ranker_pairwise.py training_data.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Feature names (should match ml_ranker.py)
FEATURE_NAMES = [
    'edit_distance',
    'token_overlap', 
    'trigram_overlap',
    'popularity',
    'context_score'
]


def load_training_data(json_path):
    """Load training data from JSON file."""
    logging.info(f"Loading training data from {json_path}...")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    examples = data['examples']
    metadata = data.get('metadata', {})
    
    logging.info(f"Loaded {len(examples)} training examples")
    logging.info(f"Metadata: {metadata}")
    
    return examples, metadata


def extract_features(example):
    """
    Extract feature vector from training example.
    
    Computes features on-the-fly from the raw data:
    - edit_distance: Normalized Levenshtein distance
    - token_overlap: Jaccard similarity of words
    - trigram_overlap: From training data (already computed)
    - popularity: Log-normalized popularity
    - context_score: Always 0 (no playlist context in training data)
    """
    import Levenshtein
    
    query = example['query'].lower()
    metadata = example['metadata']
    track_name = metadata['normalized'].lower()
    artist_name = metadata['artist'].lower()
    full_track = f"{track_name} {artist_name}"
    popularity = metadata['popularity']
    trigram_score = example['trigram_score']
    
    # 1. Edit distance (normalized)
    max_len = max(len(query), len(full_track), 1)
    edit_dist = Levenshtein.distance(query, full_track)
    edit_distance_norm = 1.0 - (edit_dist / max_len)
    
    # 2. Token overlap (Jaccard)
    query_tokens = set(query.split())
    track_tokens = set(full_track.split())
    if query_tokens and track_tokens:
        intersection = len(query_tokens & track_tokens)
        union = len(query_tokens | track_tokens)
        token_overlap = intersection / union if union > 0 else 0.0
    else:
        token_overlap = 0.0
    
    # 3. Trigram overlap (from training data)
    trigram_overlap = trigram_score
    
    # 4. Popularity (log-normalized)
    popularity_norm = np.log1p(popularity) / np.log1p(100000)  # Max ~100k
    
    # 5. Context score (no playlist context in training data)
    context_score = 0.0
    
    features = [
        edit_distance_norm,
        token_overlap,
        trigram_overlap,
        popularity_norm,
        context_score
    ]
    return np.array(features, dtype=np.float32)


def create_pairwise_examples(examples):
    """
    Convert training examples into pairwise ranking examples.
    
    For each query:
    1. Group all candidates by query
    2. For each positive (correct) candidate, create pairs with negative candidates
    3. Pair format: (positive_features - negative_features, label=1)
    
    This teaches the model: "positive should rank HIGHER than negative"
    
    Returns:
        X: Feature difference vectors (positive - negative)
        y: Labels (always 1, meaning first should rank higher)
        queries: Query text for each pair (for debugging)
    """
    logging.info("Creating pairwise ranking examples...")
    
    # Group examples by query
    query_groups = defaultdict(lambda: {'positive': [], 'negative': []})
    
    for ex in examples:
        query = ex['query']
        if ex['is_correct']:
            query_groups[query]['positive'].append(ex)
        else:
            query_groups[query]['negative'].append(ex)
    
    # Create pairs
    X_pairs = []
    y_pairs = []
    query_pairs = []
    
    pair_count = 0
    query_count = 0
    
    for query, group in query_groups.items():
        positives = group['positive']
        negatives = group['negative']
        
        if not positives or not negatives:
            continue  # Need both positive and negative examples
        
        query_count += 1
        
        # Create pairs: each positive with multiple negatives
        for pos_ex in positives:
            pos_features = extract_features(pos_ex)
            
            # Sample negatives to avoid too many pairs
            # Use at most 10 negative examples per positive
            sampled_negatives = negatives
            if len(negatives) > 10:
                sampled_negatives = np.random.choice(negatives, size=10, replace=False)
            
            for neg_ex in sampled_negatives:
                neg_features = extract_features(neg_ex)
                
                # Feature difference: positive - negative
                # Label=1: positive should rank higher (positive features are better)
                feature_diff_pos = pos_features - neg_features
                X_pairs.append(feature_diff_pos)
                y_pairs.append(1)
                query_pairs.append(query)
                
                # Also add reverse pair for balance
                # Label=0: negative should rank lower (negative features are worse)
                feature_diff_neg = neg_features - pos_features
                X_pairs.append(feature_diff_neg)
                y_pairs.append(0)
                query_pairs.append(query)
                
                pair_count += 2
    
    X = np.array(X_pairs, dtype=np.float32)
    y = np.array(y_pairs, dtype=np.int32)
    
    logging.info(f"Created {pair_count} pairwise examples from {query_count} queries")
    logging.info(f"Average pairs per query: {pair_count / query_count:.1f}")
    
    return X, y, query_pairs


def train_ranking_model(X_train, y_train, X_val, y_val):
    """
    Train a pairwise ranking model using LinearSVC.
    
    LinearSVC with hinge loss is equivalent to RankSVM for pairwise ranking.
    The model learns to predict: should candidate A rank higher than B?
    """
    logging.info("Training pairwise ranking model...")
    
    # Standardize features (important for SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train LinearSVC (RankSVM)
    # C controls regularization (higher = less regularization)
    model = LinearSVC(
        C=1.0,
        loss='hinge',  # Hinge loss for ranking
        max_iter=5000,
        random_state=42,
        dual=True  # Use dual formulation for small feature space
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_scaled, y_train)
    val_acc = model.score(X_val_scaled, y_val)
    
    logging.info(f"Training pairwise accuracy: {train_acc:.4f}")
    logging.info(f"Validation pairwise accuracy: {val_acc:.4f}")
    logging.info("(Pairwise accuracy = how often model correctly orders pairs)")
    
    return model, scaler


def analyze_feature_importance(model, feature_names):
    """
    Analyze feature importance from model coefficients.
    
    For ranking, coefficient sign matters:
    - Positive: Feature should be HIGHER in relevant candidates
    - Negative: Feature should be LOWER in relevant candidates
    """
    logging.info("\nFeature importance analysis:")
    logging.info("=" * 60)
    
    coefs = model.coef_[0]
    
    # Sort by absolute coefficient magnitude
    importance = list(zip(feature_names, coefs))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    
    for name, coef in importance:
        direction = "↑" if coef > 0 else "↓"
        logging.info(f"  {name:20s}: {coef:8.4f} {direction}")
    
    logging.info("=" * 60)
    logging.info("↑ = Higher is better (positive coefficient)")
    logging.info("↓ = Lower is better (negative coefficient)")


def save_model(model, scaler, output_path):
    """Save trained model and scaler."""
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_names': FEATURE_NAMES,
        'model_type': 'pairwise_ranking',
        'version': 'r7_pairwise_v1'
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    file_size = Path(output_path).stat().st_size
    logging.info(f"Model saved to {output_path} ({file_size} bytes)")


def test_model_predictions(model, scaler, examples):
    """
    Test model on some real examples to verify ranking behavior.
    
    For each query with multiple candidates, show predicted ranking.
    """
    logging.info("\nTesting model predictions on sample queries...")
    logging.info("=" * 60)
    
    # Group examples by query
    query_groups = defaultdict(list)
    for ex in examples:
        query_groups[ex['query']].append(ex)
    
    # Test on first 5 queries with multiple candidates
    test_queries = [q for q, exs in query_groups.items() if len(exs) > 3][:5]
    
    for query in test_queries:
        candidates = query_groups[query]
        
        # Extract features and get predictions
        features_list = []
        for ex in candidates:
            features = extract_features(ex)
            features_list.append(features)
        
        X = np.array(features_list)
        X_scaled = scaler.transform(X)
        
        # Get decision function scores (raw SVM scores)
        scores = model.decision_function(X_scaled)
        
        # Sort by score (higher = more relevant)
        ranked_indices = np.argsort(-scores)
        
        logging.info(f"\nQuery: '{query}'")
        for i, idx in enumerate(ranked_indices[:5]):
            ex = candidates[idx]
            score = scores[idx]
            label = "✓" if ex['is_correct'] else "✗"
            track_str = f"{ex['metadata']['normalized']} - {ex['metadata']['artist']}"
            logging.info(f"  {i+1}. {label} {track_str[:50]:50s} (score: {score:6.3f})")
    
    logging.info("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python train_ranker_pairwise.py <training_data.json>")
        sys.exit(1)
    
    training_data_path = sys.argv[1]
    
    # Load training data
    examples, metadata = load_training_data(training_data_path)
    
    # Create pairwise ranking examples
    X, y, queries = create_pairwise_examples(examples)
    
    # Split into train/val
    # Group by query to avoid data leakage
    unique_queries = list(set(queries))
    train_queries, val_queries = train_test_split(
        unique_queries, 
        test_size=0.2, 
        random_state=42
    )
    
    train_queries_set = set(train_queries)
    val_queries_set = set(val_queries)
    
    train_mask = [q in train_queries_set for q in queries]
    val_mask = [q in val_queries_set for q in queries]
    
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    
    logging.info(f"Train pairs: {len(X_train)}, Val pairs: {len(X_val)}")
    
    # Train model
    model, scaler = train_ranking_model(X_train, y_train, X_val, y_val)
    
    # Analyze feature importance
    analyze_feature_importance(model, FEATURE_NAMES)
    
    # Test predictions
    test_model_predictions(model, scaler, examples)
    
    # Save model
    output_dir = Path(training_data_path).parent
    output_path = output_dir / 'ranker_model_pairwise.pkl'
    save_model(model, scaler, output_path)
    
    logging.info(f"\n✓ Training complete!")
    logging.info(f"✓ Model saved to: {output_path}")
    logging.info(f"✓ Model type: Pairwise Ranking (RankSVM)")
    logging.info(f"\nNext steps:")
    logging.info(f"1. Update ml_ranker.py to use pairwise ranking predictions")
    logging.info(f"2. Test with: python test_pairwise_ranker.py")
    logging.info(f"3. Deploy: cp {output_path} ../data/ranker_model.pkl")


if __name__ == '__main__':
    main()
