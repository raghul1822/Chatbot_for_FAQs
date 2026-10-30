"""
Core Chatbot Matching Engine for College Information FAQ System
CodeAlpha Artificial Intelligence Internship - Task 2

Implements:
  - Dataset ingestion and schema validation from CSV or JSON
  - TF-IDF vectorization with n-gram features and sublinear term scaling
  - Cosine similarity computation
  - Confidence threshold decision gating
  - Fallback messaging for unsupported / out-of-domain queries
  - Top-3 candidate FAQ recommendations
  - Term-level explainability (XAI)
  - Primary function: get_response(user_question)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import TextPreprocessor


REQUIRED_COLUMNS = ("question", "answer")

DEFAULT_FALLBACK_MESSAGE = (
    "I am sorry, but I could not find a sufficiently relevant answer in the College FAQ "
    "knowledgebase. Please try rephrasing your question with specific academic keywords "
    "(e.g., 'admissions', 'B.Tech eligibility', 'library timings', 'hostel fees'). "
    "For official administrative support, please contact admissions@apextech.edu."
)

EMPTY_INPUT_MESSAGE = (
    "Please enter a valid question. Empty messages, whitespace, or pure punctuation "
    "cannot be processed by the FAQ matcher."
)


class DatasetValidationError(ValueError):
    """Raised when an FAQ dataset fails schema or data integrity validation."""
    pass


class CollegeFAQChatbot:
    """
    Intelligent FAQ Chatbot Engine for College Information.

    Workflow:
      1. Loads and validates dataset (CSV or JSON).
      2. Cleans FAQ corpus using TextPreprocessor.
      3. Fits TF-IDF vectorizer (unigrams + bigrams, sublinear TF scaling, L2 norm).
      4. Vectorizes user questions into the fitted space.
      5. Calculates Cosine Similarity: cos(theta) = (q . d) / (||q|| * ||d||).
      6. Applies confidence threshold to return verified answers or fallback.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path] = "data/faqs.csv",
        similarity_threshold: float = 0.35,
        top_k: int = 3,
        fallback_message: str = DEFAULT_FALLBACK_MESSAGE,
        include_keywords: bool = True,
    ) -> None:
        """
        Initializes and fits the FAQ Chatbot.

        Args:
            dataset_path: Path to dataset CSV or JSON file.
            similarity_threshold: Minimum cosine similarity score (0.0 to 1.0) to accept a match.
            top_k: Number of candidate suggestions to retrieve.
            fallback_message: Message returned when confidence is below threshold.
            include_keywords: Whether to append keywords to question text for indexing.
        """
        self.dataset_path = Path(dataset_path)
        self.similarity_threshold = float(similarity_threshold)
        self.top_k = int(top_k)
        self.fallback_message = fallback_message
        self.include_keywords = include_keywords

        # Initialize NLP preprocessor
        self.preprocessor = TextPreprocessor()

        # Initialize TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            norm="l2",
            lowercase=False,  # Lowercasing is handled explicitly in TextPreprocessor
        )

        # Load and fit data
        self.df: pd.DataFrame = pd.DataFrame()
        self.records: List[Dict[str, Any]] = []
        self.cleaned_corpus: List[str] = []
        self.tfidf_matrix = None
        self.is_fitted: bool = False

        self.load_and_fit()

    def load_dataset(self) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Loads and validates the FAQ dataset from CSV or JSON.

        Raises:
            FileNotFoundError: If dataset path is invalid.
            DatasetValidationError: If required columns or records are invalid.
        """
        if not self.dataset_path.is_file():
            raise FileNotFoundError(f"Dataset file not found at: {self.dataset_path.resolve()}")

        suffix = self.dataset_path.suffix.lower()

        if suffix == ".csv":
            try:
                df_raw = pd.read_csv(self.dataset_path, encoding="utf-8")
            except Exception as exc:
                raise DatasetValidationError(f"Could not read CSV file: {exc}") from exc

            # Verify required columns exist
            for col in REQUIRED_COLUMNS:
                if col not in df_raw.columns:
                    raise DatasetValidationError(
                        f"Missing mandatory column '{col}' in FAQ dataset. Required columns: {REQUIRED_COLUMNS}"
                    )

            if len(df_raw) == 0:
                raise DatasetValidationError("FAQ dataset CSV cannot be empty.")

            records = []
            seen_questions = set()

            for idx, row in df_raw.iterrows():
                q = str(row["question"]).strip() if pd.notna(row["question"]) else ""
                a = str(row["answer"]).strip() if pd.notna(row["answer"]) else ""

                if not q or not a:
                    raise DatasetValidationError(f"Row {idx} contains empty question or answer field.")

                # Clean and check duplicates
                q_norm = q.lower()
                if q_norm in seen_questions:
                    # Duplicate question detected
                    pass
                seen_questions.add(q_norm)

                record_id = str(row.get("faq_id", f"faq_{idx + 1:03d}")).strip()
                cat = str(row.get("category", "General")).strip() if pd.notna(row.get("category")) else "General"
                kw = str(row.get("keywords", "")).strip() if pd.notna(row.get("keywords")) else ""

                records.append({
                    "faq_id": record_id,
                    "category": cat,
                    "question": q,
                    "answer": a,
                    "keywords": kw,
                })

        elif suffix == ".json":
            import json
            try:
                with open(self.dataset_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
            except Exception as exc:
                raise DatasetValidationError(f"Could not read JSON file: {exc}") from exc

            if isinstance(raw_data, dict) and "faqs" in raw_data:
                raw_data = raw_data["faqs"]

            if not isinstance(raw_data, list) or len(raw_data) == 0:
                raise DatasetValidationError("JSON FAQ dataset must contain a non-empty list of FAQs.")

            records = []
            for idx, item in enumerate(raw_data, start=1):
                if not isinstance(item, dict):
                    raise DatasetValidationError(f"Item at index {idx} is not a valid JSON dictionary.")
                for col in REQUIRED_COLUMNS:
                    if col not in item or not str(item[col]).strip():
                        raise DatasetValidationError(f"Item at index {idx} missing required field '{col}'.")

                record_id = str(item.get("faq_id", item.get("id", f"faq_{idx:03d}"))).strip()
                cat = str(item.get("category", "General")).strip()
                q = str(item["question"]).strip()
                a = str(item["answer"]).strip()
                kw = str(item.get("keywords", ""))
                if isinstance(item.get("keywords"), list):
                    kw = " ".join(item["keywords"])

                records.append({
                    "faq_id": record_id,
                    "category": cat,
                    "question": q,
                    "answer": a,
                    "keywords": kw,
                })
        else:
            raise DatasetValidationError(f"Unsupported dataset format '{suffix}'. Supported formats: .csv, .json")

        df = pd.DataFrame(records)
        return df, records

    def load_and_fit(self) -> "CollegeFAQChatbot":
        """Loads the dataset and fits the TF-IDF vector space."""
        self.df, self.records = self.load_dataset()

        self.cleaned_corpus = []
        for item in self.records:
            corpus_text = item["question"]
            if self.include_keywords and item.get("keywords"):
                corpus_text = f"{corpus_text} {item['keywords']}"

            cleaned = self.preprocessor.clean_text(corpus_text)
            if not cleaned.strip():
                cleaned = self.preprocessor.normalize_text(corpus_text)
            self.cleaned_corpus.append(cleaned)

        self.tfidf_matrix = self.vectorizer.fit_transform(self.cleaned_corpus)
        self.is_fitted = True
        return self

    def set_threshold(self, threshold: float) -> None:
        """Dynamically tunes the cosine similarity cutoff threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be a float between 0.0 and 1.0.")
        self.similarity_threshold = float(threshold)

    def get_categories(self) -> List[str]:
        """Returns the list of unique categories in the knowledge base."""
        return sorted(list(self.df["category"].unique()))

    def get_faqs_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Returns all FAQs under a given category."""
        return [r for r in self.records if r["category"] == category]

    def explain_match(self, query: str, matched_faq_id: str) -> Dict[str, Any]:
        """
        Provides term-level explainability (XAI) detailing overlapping TF-IDF features.
        """
        if not self.is_fitted:
            return {}

        record_idx = next((i for i, r in enumerate(self.records) if r["faq_id"] == matched_faq_id), None)
        if record_idx is None:
            return {}

        cleaned_query = self.preprocessor.clean_text(query)
        q_vec = self.vectorizer.transform([cleaned_query]).toarray()[0]
        faq_vec = self.tfidf_matrix[record_idx].toarray()[0]

        feature_names = self.vectorizer.get_feature_names_out()
        product = q_vec * faq_vec
        overlapping_indices = np.where(product > 0)[0]

        contributions = []
        for idx in overlapping_indices:
            contributions.append({
                "term": feature_names[idx],
                "query_weight": round(float(q_vec[idx]), 4),
                "faq_weight": round(float(faq_vec[idx]), 4),
                "contribution": round(float(product[idx]), 4),
            })
        contributions.sort(key=lambda x: x["contribution"], reverse=True)

        return {
            "matched_faq_id": matched_faq_id,
            "matched_question": self.records[record_idx]["question"],
            "overlapping_features": contributions,
            "total_dot_product": round(float(np.sum(product)), 4),
        }

    def ask(
        self,
        user_question: str,
        threshold: Optional[float] = None,
        return_debug: bool = False,
    ) -> Dict[str, Any]:
        """
        Processes a user question and returns the answer, similarity score, and metadata.

        Args:
            user_question: Incoming text string from user.
            threshold: Optional override for the similarity threshold.
            return_debug: Whether to attach detailed XAI debugging metadata.

        Returns:
            Dictionary with response details: answer, confidence_score, status,
            matched_question, suggestions, and optional debug info.
        """
        active_threshold = float(threshold) if threshold is not None else self.similarity_threshold

        # Step 1: Validate input
        if not self.preprocessor.is_valid_input(user_question):
            return {
                "answer": EMPTY_INPUT_MESSAGE,
                "is_fallback": True,
                "confidence_score": 0.0,
                "similarity_threshold": active_threshold,
                "matched_faq_id": None,
                "matched_question": None,
                "category": None,
                "status": "empty_input",
                "suggestions": [],
                "debug": {"reason": "empty_or_whitespace"} if return_debug else None,
            }

        # Step 2: Clean input
        cleaned_query = self.preprocessor.clean_text(user_question)
        if not cleaned_query.strip():
            return {
                "answer": self.fallback_message,
                "is_fallback": True,
                "confidence_score": 0.0,
                "similarity_threshold": active_threshold,
                "matched_faq_id": None,
                "matched_question": None,
                "category": None,
                "status": "unrecognized_tokens",
                "suggestions": [],
                "debug": {"reason": "no_valid_tokens_after_cleanup"} if return_debug else None,
            }

        # Step 3: Vectorize user question
        query_vector = self.vectorizer.transform([cleaned_query])

        # Step 4: Handle out-of-vocabulary query
        if query_vector.nnz == 0:
            return {
                "answer": self.fallback_message,
                "is_fallback": True,
                "confidence_score": 0.0,
                "similarity_threshold": active_threshold,
                "matched_faq_id": None,
                "matched_question": None,
                "category": None,
                "status": "out_of_vocabulary_fallback",
                "suggestions": [],
                "debug": {"reason": "zero_vocabulary_overlap"} if return_debug else None,
            }

        # Step 5: Compute cosine similarity against all FAQ question vectors
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Step 6: Rank candidates
        ranked_indices = np.argsort(similarities)[::-1]
        best_idx = ranked_indices[0]
        best_score = round(float(similarities[best_idx]), 4)
        best_record = self.records[best_idx]

        # Step 7: Check threshold
        is_fallback = best_score < active_threshold

        if not is_fallback:
            response_answer = best_record["answer"]
            status = "matched"
        else:
            response_answer = self.fallback_message
            status = "low_confidence_fallback"

        # Step 8: Assemble alternative suggestions
        suggestions = []
        for idx in ranked_indices[: self.top_k]:
            score_val = round(float(similarities[idx]), 4)
            if score_val >= 0.15:
                suggestions.append({
                    "faq_id": self.records[idx]["faq_id"],
                    "category": self.records[idx]["category"],
                    "question": self.records[idx]["question"],
                    "score": score_val,
                })

        response: Dict[str, Any] = {
            "answer": response_answer,
            "is_fallback": is_fallback,
            "confidence_score": best_score,
            "similarity_threshold": active_threshold,
            "matched_faq_id": best_record["faq_id"],
            "matched_question": best_record["question"],
            "category": best_record["category"],
            "status": status,
            "suggestions": suggestions,
        }

        if return_debug:
            response["debug"] = {
                "cleaned_query": cleaned_query,
                "explainability": self.explain_match(user_question, best_record["faq_id"]),
                "all_candidates": suggestions,
            }

        return response


# Global singleton instance for standalone function calls
_GLOBAL_BOT: Optional[CollegeFAQChatbot] = None


def get_response(
    user_question: str,
    dataset_path: str = "data/faqs.csv",
    threshold: float = 0.35,
    return_debug: bool = False,
) -> Dict[str, Any]:
    """
    Standard reusable function interface: get_response(user_question).

    Args:
        user_question: Natural language question from the user.
        dataset_path: Path to the FAQ CSV dataset.
        threshold: Minimum similarity score threshold (default: 0.35).
        return_debug: Whether to include developer debug/explainability data.

    Returns:
        Dictionary containing answer, confidence_score, status, and matched question.
    """
    global _GLOBAL_BOT
    if _GLOBAL_BOT is None:
        _GLOBAL_BOT = CollegeFAQChatbot(dataset_path=dataset_path, similarity_threshold=threshold)
    return _GLOBAL_BOT.ask(user_question, threshold=threshold, return_debug=return_debug)

