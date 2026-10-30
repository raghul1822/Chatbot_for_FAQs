"""
Text Preprocessing Module for College Information FAQ Chatbot
CodeAlpha Artificial Intelligence Internship - Task 2

Provides a robust, explainable NLP cleaning pipeline:
  1. Unicode Normalization (NFKD)
  2. Lowercasing
  3. Contraction Expansion (can't -> can not)
  4. Academic Abbreviation Preservation (B.Tech, M.Tech, CGPA, 10+2)
  5. Tokenization (NLTK with resilient regex fallback)
  6. Negation-Preserving Stopword Filtering (preserves 'not', 'without', 'no')
  7. WordNet Lemmatization
  8. Input Validation
"""

from typing import List, Optional, Set
import re
import string
import unicodedata

# Resilient NLTK import with automated fallback downloads
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)

    try:
        _lemmatizer_check = WordNetLemmatizer()
        _lemmatizer_check.lemmatize("students")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)

    try:
        _ = word_tokenize("test tokenizer")
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)

    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False


# CRITICAL: Words that MUST NOT be discarded during stopword filtering.
# Dropping negations changes the polarity of inquiries, such as:
# "Can I take exams without attendance?" vs "Can I take exams?"
NEGATION_WORDS: Set[str] = {
    "not",
    "no",
    "never",
    "none",
    "neither",
    "nor",
    "cannot",
    "can't",
    "won't",
    "don't",
    "didn't",
    "without",
    "against",
    "n't",
}

# Standard English contractions expansion dictionary
CONTRACTION_MAP = {
    "can't": "can not",
    "won't": "will not",
    "don't": "do not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    "i'm": "i am",
    "you're": "you are",
    "they're": "they are",
    "we're": "we are",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "where's": "where is",
    "how's": "how is",
}

# Academic abbreviation mappings to normalize degree terminology
ACADEMIC_TERM_NORMALIZATION = {
    r"\bb\.?\s*tech\b": "btech",
    r"\bm\.?\s*tech\b": "mtech",
    r"\bb\.?\s*e\b": "be",
    r"\bm\.?\s*b\.?\s*a\b": "mba",
    r"\bm\.?\s*c\.?\s*a\b": "mca",
    r"\bph\.?\s*d\b": "phd",
    r"\bc\.?\s*g\.?\s*p\.?\s*a\b": "cgpa",
    r"\bg\.?\s*p\.?\s*a\b": "gpa",
    r"\b10\s*\+\s*2\b": "10plus2",
}


class TextPreprocessor:
    """
    Reusable, modular NLP text preprocessor for the College Information FAQ Chatbot.
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        preserve_negations: bool = True,
    ) -> None:
        """
        Initializes the preprocessor pipeline.

        Args:
            remove_stopwords: Whether to filter non-informative high-frequency words.
            lemmatize: Whether to reduce morphological word forms to root lemmas.
            preserve_negations: Whether to protect negation tokens from being dropped.
        """
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.preserve_negations = preserve_negations

        # Initialize Lemmatizer
        if _NLTK_AVAILABLE:
            try:
                self.lemmatizer = WordNetLemmatizer()
            except Exception:
                self.lemmatizer = None
        else:
            self.lemmatizer = None

        # Build Stopword Dictionary
        if _NLTK_AVAILABLE:
            try:
                base_stops = set(stopwords.words("english"))
            except Exception:
                base_stops = self._fallback_stopwords()
        else:
            base_stops = self._fallback_stopwords()

        if self.preserve_negations:
            # Exclude negations so "not", "without", "no" remain intact
            self.stop_words = base_stops - NEGATION_WORDS
        else:
            self.stop_words = base_stops

    @staticmethod
    def _fallback_stopwords() -> Set[str]:
        """Built-in basic English stopwords fallback."""
        return {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
            "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
            "to", "was", "were", "will", "with"
        }

    @staticmethod
    def expand_contractions(text: str) -> str:
        """Expands conversational contractions (e.g. can't -> can not)."""
        lower = text.lower()
        for contraction, expanded in CONTRACTION_MAP.items():
            lower = re.sub(rf"\b{re.escape(contraction)}\b", expanded, lower)
        return lower

    @staticmethod
    def normalize_academic_terms(text: str) -> str:
        """Normalizes academic abbreviations like B.Tech -> btech, 10+2 -> 10plus2."""
        lowered = text.lower()
        for pattern, replacement in ACADEMIC_TERM_NORMALIZATION.items():
            lowered = re.sub(pattern, replacement, lowered)
        return lowered

    @staticmethod
    def normalize_text(text: Optional[str]) -> str:
        """
        Converts text to NFKD unicode format, lowercases, and compresses redundant spaces.
        """
        if text is None:
            return ""
        normalized = unicodedata.normalize("NFKD", str(text))
        lowered = normalized.lower()
        cleaned = re.sub(r"\s+", " ", lowered).strip()
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text into word tokens using NLTK or regex fallback."""
        if not text:
            return []
        if _NLTK_AVAILABLE:
            try:
                return word_tokenize(text)
            except Exception:
                pass
        # Fallback alphanumeric token extraction
        return re.findall(r"\b[a-zA-Z0-9']+\b", text)

    def filter_and_lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Strips edge punctuation, filters stopwords (protecting negations),
        and applies dual-pass (verb then noun) WordNet lemmatization.
        """
        processed: List[str] = []
        for token in tokens:
            clean_token = token.strip(string.punctuation).lower()
            if not clean_token:
                continue

            # Skip single-letter symbols except important identifiers
            if len(clean_token) == 1 and clean_token not in {"a", "i"}:
                continue

            # Stopword filtering
            if self.remove_stopwords and clean_token in self.stop_words:
                continue

            # Lemmatization
            if self.lemmatize and self.lemmatizer is not None:
                try:
                    lemma_v = self.lemmatizer.lemmatize(clean_token, pos="v")
                    clean_token = self.lemmatizer.lemmatize(lemma_v, pos="n")
                except Exception:
                    pass

            processed.append(clean_token)

        return processed

    def clean_text(self, text: Optional[str]) -> str:
        """
        End-to-end preprocessing pipeline for an input string.
        """
        if text is None:
            return ""
        norm = self.normalize_text(text)
        if not norm:
            return ""

        expanded = self.expand_contractions(norm)
        academic_normalized = self.normalize_academic_terms(expanded)
        tokens = self.tokenize(academic_normalized)
        processed = self.filter_and_lemmatize(tokens)
        return " ".join(processed)

    def is_valid_input(self, text: Optional[str]) -> bool:
        """
        Validates whether input contains meaningful alphanumeric content.
        Rejects None, empty string, whitespace-only, or pure punctuation queries.
        """
        if text is None:
            return False
        stripped = text.strip()
        if not stripped:
            return False
        # Must contain at least one alphanumeric character
        if not re.search(r"[a-zA-Z0-9]", stripped):
            return False
        return True

