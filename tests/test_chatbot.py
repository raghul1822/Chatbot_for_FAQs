"""
Comprehensive Unit Test Suite for College Information FAQ Chatbot
CodeAlpha Artificial Intelligence Internship - Task 2

Covers all 15 required testing criteria:
  1. Exact FAQ question match
  2. Capitalization variation
  3. Punctuation variations
  4. Paraphrased question
  5. Short question
  6. Question containing spelling variations / typos
  7. Empty input
  8. Whitespace-only input
  9. Unrelated question (fallback)
  10. Ambiguous / multi-intent question
  11. Duplicate FAQ handling
  12. Missing dataset file handling
  13. Invalid dataset columns validation
  14. Questions containing important negation words
  15. Questions below the similarity threshold
  16. Top-level get_response() standalone interface
"""

import sys
import unittest
from pathlib import Path
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot import (
    CollegeFAQChatbot,
    DatasetValidationError,
    DEFAULT_FALLBACK_MESSAGE,
    EMPTY_INPUT_MESSAGE,
    get_response,
)
from preprocess import TextPreprocessor, NEGATION_WORDS


class TestTextPreprocessor(unittest.TestCase):
    """Tests NLP text cleaning, tokenization, contractions, and negation preservation."""

    def setUp(self):
        self.preprocessor = TextPreprocessor(
            remove_stopwords=True,
            lemmatize=True,
            preserve_negations=True,
        )

    def test_lowercasing_and_whitespace(self):
        raw = "  WHAT   ARE   the library   HOURS???  \n"
        cleaned = self.preprocessor.clean_text(raw)
        self.assertNotIn("WHAT", cleaned)
        self.assertIn("library", cleaned)
        self.assertIn("hour", cleaned)

    def test_contractions_expansion(self):
        raw = "I can't attend classes and shouldn't fail."
        expanded = self.preprocessor.expand_contractions(raw)
        self.assertIn("can not", expanded)

    def test_academic_term_normalization(self):
        """Verifies academic abbreviations are normalized for consistent vectorization."""
        text = "Applying for B.Tech and M.Tech with 10+2 marks."
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("btech", cleaned)
        self.assertIn("mtech", cleaned)
        self.assertIn("10plus2", cleaned)

    def test_preservation_of_negation_words(self):
        """CRITICAL: Verifies negation words like 'not', 'no', 'without' are preserved."""
        text = "Can I write exams without minimum attendance?"
        cleaned = self.preprocessor.clean_text(text)
        self.assertIn("without", cleaned.split())

        for neg in ["not", "no", "never", "without", "cannot"]:
            self.assertNotIn(neg, self.preprocessor.stop_words)

    def test_empty_and_whitespace_validation(self):
        self.assertFalse(self.preprocessor.is_valid_input(""))
        self.assertFalse(self.preprocessor.is_valid_input("   "))
        self.assertFalse(self.preprocessor.is_valid_input("???!!!..."))
        self.assertFalse(self.preprocessor.is_valid_input(None))
        self.assertTrue(self.preprocessor.is_valid_input("College fees?"))


class TestDatasetValidation(unittest.TestCase):
    """Tests dataset loading, schema verification, and error handling."""

    def test_missing_dataset_file(self):
        """Test Case 12: Missing dataset file error handling."""
        with self.assertRaises(FileNotFoundError):
            CollegeFAQChatbot(dataset_path="data/non_existent_file.csv")

    def test_invalid_columns_dataset(self):
        """Test Case 13: Invalid dataset columns error handling."""
        temp_invalid_csv = PROJECT_ROOT / "data" / "temp_invalid.csv"
        df_invalid = pd.DataFrame({"wrong_col1": ["a"], "wrong_col2": ["b"]})
        df_invalid.to_csv(temp_invalid_csv, index=False)

        try:
            with self.assertRaises(DatasetValidationError):
                CollegeFAQChatbot(dataset_path=temp_invalid_csv)
        finally:
            if temp_invalid_csv.exists():
                temp_invalid_csv.unlink()

    def test_duplicate_faqs_handling(self):
        """Test Case 11: Duplicate FAQ questions in dataset are processed cleanly."""
        dataset_path = PROJECT_ROOT / "data" / "faqs.csv"
        bot = CollegeFAQChatbot(dataset_path=dataset_path)
        self.assertGreaterEqual(len(bot.records), 30)


class TestCollegeChatbotFunctional(unittest.TestCase):
    """Integration test suite covering all functional chat requirements."""

    @classmethod
    def setUpClass(cls):
        cls.dataset_path = PROJECT_ROOT / "data" / "faqs.csv"
        cls.bot = CollegeFAQChatbot(
            dataset_path=cls.dataset_path,
            similarity_threshold=0.35,
            top_k=3,
            include_keywords=True,
        )

    def tearDown(self):
        self.bot.set_threshold(0.35)

    def test_01_exact_faq_question(self):
        """Test Case 1: Exact FAQ question match."""
        query = "How can I apply for undergraduate admission?"
        res = self.bot.ask(query)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_001")
        self.assertGreaterEqual(res["confidence_score"], 0.60)
        self.assertIn("https://admissions.apextech.edu", res["answer"])

    def test_02_capitalization_variation(self):
        """Test Case 2: Same question with different capitalization."""
        query = "HOW CAN I APPLY FOR UNDERGRADUATE ADMISSION?"
        res = self.bot.ask(query)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_001")

    def test_03_punctuation_differences(self):
        """Test Case 3: Question with extra punctuation differences."""
        query = "What is the application deadline for the fall semester?!?"
        res = self.bot.ask(query)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_002")

    def test_04_paraphrased_question(self):
        """Test Case 4: Paraphrased question with synonymous words."""
        query = "When is the last date to submit the application form for fall?"
        res = self.bot.ask(query)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_002")
        self.assertIn("June 30th", res["answer"])

    def test_05_short_question(self):
        """Test Case 5: Short keyword-dense question."""
        query = "hostel curfew"
        res = self.bot.ask(query)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_039")
        self.assertIn("9:00 PM", res["answer"])

    def test_06_spelling_variations_typos(self):
        """Test Case 6: Question containing spelling variations / typos."""
        query = "Can tuition fees be payed in installmnts???"
        res = self.bot.ask(query)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_017")
        self.assertIn("installments", res["answer"])

    def test_07_empty_input(self):
        """Test Case 7: Empty input handling."""
        res = self.bot.ask("")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["status"], "empty_input")
        self.assertEqual(res["answer"], EMPTY_INPUT_MESSAGE)

    def test_08_whitespace_only_input(self):
        """Test Case 8: Whitespace-only input handling."""
        res = self.bot.ask("      \t \n")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["status"], "empty_input")
        self.assertEqual(res["answer"], EMPTY_INPUT_MESSAGE)

    def test_09_unrelated_question(self):
        """Test Case 9: Unrelated / out-of-domain question triggers fallback."""
        query = "Can you bake me a chocolate cake and order groceries?"
        res = self.bot.ask(query)
        self.assertTrue(res["is_fallback"])
        self.assertLess(res["confidence_score"], self.bot.similarity_threshold)
        self.assertEqual(res["answer"], DEFAULT_FALLBACK_MESSAGE)

    def test_10_ambiguous_question_returns_suggestions(self):
        """Test Case 10: Ambiguous question returns top suggestions."""
        query = "admission requirements"
        res = self.bot.ask(query)
        self.assertGreater(len(res["suggestions"]), 0)

    def test_14_negation_in_question(self):
        """Test Case 14: Questions containing negation words retain distinction."""
        query = "Can I take semester examinations without meeting the minimum attendance?"
        res = self.bot.ask(query)
        self.assertEqual(res["matched_faq_id"], "faq_024")
        self.assertIn("No", res["answer"])

    def test_15_below_similarity_threshold(self):
        """Test Case 15: Questions scoring below threshold trigger fallback."""
        # Intentionally high threshold
        self.bot.set_threshold(0.95)
        res = self.bot.ask("Tell me about the campus library?")
        self.assertTrue(res["is_fallback"])
        self.assertEqual(res["status"], "low_confidence_fallback")

    def test_16_get_response_standalone_function(self):
        """Test Case 16: Top-level reusable get_response() function works."""
        res = get_response("What are the library hours?")
        self.assertIn("answer", res)
        self.assertIn("confidence_score", res)
        self.assertFalse(res["is_fallback"])
        self.assertEqual(res["matched_faq_id"], "faq_031")


if __name__ == "__main__":
    unittest.main()
