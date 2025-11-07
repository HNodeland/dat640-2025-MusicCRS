"""
Text normalization for R7 autocorrect.

This module provides text preprocessing functions to normalize song/artist names
before indexing and querying. Normalization ensures consistent matching even with
variations in capitalization, punctuation, and special characters.

Task: 7.1.5 - Implement normalize() function

Functions:
    normalize(text: str) -> str: Main normalization pipeline
    remove_brackets(text: str) -> str: Strip bracketed content
    remove_punctuation(text: str) -> str: Remove punctuation marks
    extract_trigrams(text: str) -> List[str]: Extract character trigrams
"""

import re
from typing import List, Set


def normalize(text: str) -> str:
    """
    Normalize text for matching.
    
    Applies the following transformations:
    1. Lowercase
    2. Remove content in brackets/parentheses (e.g., "Song (Remix)" → "Song")
    3. Remove punctuation except spaces
    4. Strip leading/trailing whitespace
    5. Collapse multiple spaces to single space
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text string
        
    Examples:
        >>> normalize("Blinding Lights (The Weeknd)")
        'blinding lights'
        
        >>> normalize("Don't Stop Believin'!")
        'dont stop believin'
        
        >>> normalize("Mr. Brightside")
        'mr brightside'
    """
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove bracketed content
    text = remove_brackets(text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Strip and collapse spaces
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    
    return text


def remove_brackets(text: str) -> str:
    """
    Remove content in brackets, parentheses, or square brackets.
    
    Args:
        text: Text potentially containing bracketed content
        
    Returns:
        Text with bracketed content removed
        
    Examples:
        >>> remove_brackets("Song (Remix)")
        'Song '
        
        >>> remove_brackets("Song [feat. Artist]")
        'Song '
        
        >>> remove_brackets("Song - (Live Version)")
        'Song - '
    """
    # Remove (...), [...], and {...}
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[[^\]]*\]', '', text)
    text = re.sub(r'\{[^}]*\}', '', text)
    return text


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation marks but keep spaces.
    
    Args:
        text: Text potentially containing punctuation
        
    Returns:
        Text with punctuation removed
        
    Examples:
        >>> remove_punctuation("Don't stop!")
        'Dont stop'
        
        >>> remove_punctuation("Mr. Smith")
        'Mr Smith'
    """
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', '', text, flags=re.IGNORECASE)
    return text


def extract_bigrams(text: str, normalize_first: bool = True) -> List[str]:
    """
    Extract character bigrams from text.
    
    Bigrams are 2-character sequences used for fuzzy matching.
    Example: "cat" → ["ca", "at"]
             "hello" → ["he", "el", "ll", "lo"]
    
    Args:
        text: Text to extract bigrams from
        normalize_first: Whether to normalize text before extraction
        
    Returns:
        List of bigram strings
    """
    if normalize_first:
        text = normalize(text)
    
    if len(text) < 2:
        return [text] if text else []
    
    bigrams = []
    for i in range(len(text) - 1):
        bigram = text[i:i+2]
        bigrams.append(bigram)
    
    return bigrams


def extract_trigrams(text: str, normalize_first: bool = True) -> List[str]:
    """
    Extract character trigrams from text.
    
    Trigrams are 3-character sequences used for fuzzy matching.
    Example: "cat" → ["cat"]
             "cats" → ["cat", "ats"]
             "hello" → ["hel", "ell", "llo"]
    
    Args:
        text: Text to extract trigrams from
        normalize_first: Whether to normalize text before extraction
        
    Returns:
        List of trigram strings
        
    Examples:
        >>> extract_trigrams("cat")
        ['cat']
        
        >>> extract_trigrams("hello")
        ['hel', 'ell', 'llo']
        
        >>> extract_trigrams("a b")
        ['a b']
    """
    if normalize_first:
        text = normalize(text)
    
    if len(text) < 3:
        return [text] if text else []
    
    trigrams = []
    for i in range(len(text) - 2):
        trigram = text[i:i+3]
        trigrams.append(trigram)
    
    return trigrams


def extract_ngrams(text: str, n: int = 3, normalize_first: bool = True) -> List[str]:
    """
    Extract character n-grams from text.
    
    Args:
        text: Text to extract n-grams from
        n: Size of n-grams (2 for bigrams, 3 for trigrams)
        normalize_first: Whether to normalize text before extraction
        
    Returns:
        List of n-gram strings
    """
    if normalize_first:
        text = normalize(text)
    
    if len(text) < n:
        return [text] if text else []
    
    ngrams = []
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        ngrams.append(ngram)
    
    return ngrams


def extract_trigrams_set(text: str, normalize_first: bool = True) -> Set[str]:
    """
    Extract unique character trigrams as a set.
    
    Same as extract_trigrams() but returns a set for faster overlap calculations.
    
    Args:
        text: Text to extract trigrams from
        normalize_first: Whether to normalize text before extraction
        
    Returns:
        Set of unique trigram strings
        
    Examples:
        >>> sorted(extract_trigrams_set("hello"))
        ['ell', 'hel', 'llo']
        
        >>> sorted(extract_trigrams_set("aaa"))
        ['aaa']
    """
    return set(extract_trigrams(text, normalize_first))


def calculate_trigram_overlap(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity between trigram sets of two texts.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    Where A and B are the trigram sets of text1 and text2.
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Jaccard similarity score (0.0 to 1.0)
        
    Examples:
        >>> calculate_trigram_overlap("hello", "hello")
        1.0
        
        >>> calculate_trigram_overlap("hello", "hallo")  # doctest: +ELLIPSIS
        0.5...
        
        >>> calculate_trigram_overlap("cat", "dog")
        0.0
    """
    trigrams1 = extract_trigrams_set(text1)
    trigrams2 = extract_trigrams_set(text2)
    
    if not trigrams1 or not trigrams2:
        return 0.0
    
    intersection = len(trigrams1 & trigrams2)
    union = len(trigrams1 | trigrams2)
    
    if union == 0:
        return 0.0
    
    return intersection / union


if __name__ == "__main__":
    # Quick tests
    print("=== Normalization Tests ===")
    
    test_cases = [
        "Blinding Lights",
        "Don't Stop Believin'!",
        "Mr. Brightside",
        "Shape of You (Remix)",
        "One Dance [feat. Drake]",
        "HUMBLE. - Kendrick Lamar",
    ]
    
    for text in test_cases:
        normalized = normalize(text)
        trigrams = extract_trigrams(text)
        print(f"{text:40s} → {normalized:30s} ({len(trigrams)} trigrams)")
    
    print("\n=== Trigram Overlap Tests ===")
    pairs = [
        ("blinding lights", "blinding lights"),
        ("blinding lights", "blinding sun"),
        ("hello", "hallo"),
        ("cat", "dog"),
    ]
    
    for text1, text2 in pairs:
        overlap = calculate_trigram_overlap(text1, text2)
        print(f"{text1:20s} vs {text2:20s} → {overlap:.3f}")
