import unittest
from unittest.mock import MagicMock
from neural_data_analysis.analysis import NLPProcessor
import logging
import nltk
from nltk.corpus import stopwords, wordnet


# tests/test_nlp_processor.py


class TestNLPProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up class-level resources. Ensure NLTK data is downloaded.
        """
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet")

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger")
        except LookupError:
            nltk.download("averaged_perceptron_tagger")

    def setUp(self):
        """
        Initialize NLPProcessor instance before each test.
        """
        self.logger = MagicMock(spec=logging.Logger)
        self.nlp = NLPProcessor(logger=self.logger)

    def test_get_part_of_speech(self):
        """
        Test the POS tagging of various words.
        """
        # Test nouns
        self.assertEqual(self.nlp.get_part_of_speech("dog"), wordnet.NOUN)

        # Test verbs
        self.assertEqual(self.nlp.get_part_of_speech("running"), wordnet.VERB)

        # Test adjectives
        self.assertEqual(self.nlp.get_part_of_speech("quick"), wordnet.ADJ)

        # Test adverbs
        self.assertEqual(self.nlp.get_part_of_speech("quickly"), wordnet.ADV)

        # Test unknown word (should default to NOUN)
        self.assertEqual(self.nlp.get_part_of_speech("asdfghjkl"), wordnet.NOUN)

    def test_get_lemma(self):
        """
        Test lemmatization of words with different POS tags.
        """
        # Verb lemmatization
        self.assertEqual(self.nlp.get_lemma("running"), "run")

        # Noun lemmatization
        self.assertEqual(self.nlp.get_lemma("children"), "child")

        # Adjective lemmatization
        self.assertEqual(
            self.nlp.get_lemma("better"), "better"
        )  # WordNet lemmatizer may not change "better"

        # Adverb lemmatization
        self.assertEqual(self.nlp.get_lemma("quickly"), "quickly")  # No change expected

        # Word that cannot be lemmatized (should return original)
        self.assertEqual(self.nlp.get_lemma("asdfghjkl"), "asdfghjkl")

    def test_get_synonyms_wordnet(self):
        """
        Test synonym retrieval using WordNet.
        """
        synonyms_dog = self.nlp.get_synonyms_wordnet("dog")
        expected_synonyms_dog = ["domestic_dog", "Canis_familiaris"]
        # Since 'Canis_familiaris' is a synonym, but may vary based on WordNet version
        # We'll check for presence rather than exact match
        self.assertIn("domestic dog", synonyms_dog)
        # Note: WordNet returns 'canis_familiaris' as a synonym; transformed to 'canis familiaris'
        self.assertIn("canis familiaris", synonyms_dog)

        synonyms_run = self.nlp.get_synonyms_wordnet("run")
        self.assertIn("sprint", synonyms_run)
        self.assertIn("jog", synonyms_run)
        self.assertNotIn("run", synonyms_run)  # Original word should be excluded

        synonyms_nonexistent = self.nlp.get_synonyms_wordnet("asdfghjkl")
        self.assertEqual(synonyms_nonexistent, [])

    def test_get_stem(self):
        """
        Test stemming of words using PorterStemmer.
        """
        self.assertEqual(self.nlp.get_stem("running"), "run")
        self.assertEqual(self.nlp.get_stem("better"), "better")
        self.assertEqual(self.nlp.get_stem("cats"), "cat")
        self.assertEqual(self.nlp.get_stem("happiness"), "happi")
        self.assertEqual(self.nlp.get_stem("asdfghjkl"), "asdfghjk")

    def test_create_excluded_words(self):
        """
        Test creation of excluded words based on stopwords.
        """
        excluded_words = self.nlp.create_excluded_words()
        self.assertIsInstance(excluded_words, set)
        # Test presence of common stopwords
        self.assertIn("the", excluded_words)
        self.assertIn("is", excluded_words)
        self.assertIn("and", excluded_words)
        # Test absence of non-stopwords
        self.assertNotIn("dog", excluded_words)
        self.assertNotIn("run", excluded_words)

    def test_create_word_groups(self):
        """
        Test grouping of words into synonyms and different forms.
        """
        words = [
            "dogs",
            "dog",
            "cats",
            "cat",
            "running",
            "run",
            "children",
            "child",
            "men",
            "man",
        ]
        word_groups = self.nlp.create_word_groups(words)

        expected_groups = {
            "dog": {"dogs", "dog"},
            "cat": {"cats", "cat"},
            "run": {"running", "run"},
            "child": {"children", "child"},
            "man": {"men", "man"},
        }

        self.assertEqual(word_groups, expected_groups)

    def test_reduce_word_list_synonyms_manual(self):
        """
        Test reducing word list using manual synonym groups.
        """
        word_list = [
            "dogs",
            "dog",
            "cats",
            "cat",
            "running",
            "run",
            "children",
            "child",
            "men",
            "man",
        ]
        synonym_groups = [
            ["dogs", "dog"],
            ["cats", "cat"],
            ["running", "run"],
            ["children", "child"],
            ["men", "man"],
        ]
        reduced = self.nlp.reduce_word_list_synonyms(
            word_list, method="manual", synonym_groups=synonym_groups
        )
        # expected_reduced = ["dog", "cat", "run", "child", "man"]
        expected_reduced = ["dogs", "cats", "running", "children", "men"]
        self.assertEqual(reduced, expected_reduced)

    def test_reduce_word_list_synonyms_wordnet(self):
        """
        Test reducing word list using WordNet synonyms.
        """
        word_list = [
            "dog",
            "canine",
            "puppy",
            "run",
            "sprint",
            "jog",
            "child",
            "kid",
            "happy",
            "joyful",
        ]
        reduced = self.nlp.reduce_word_list_synonyms(word_list, method="wordnet")
        # Expected: Each group represented by the first word encountered
        # Assuming 'canine' and 'puppy' are synonyms of 'dog', 'sprint' and 'jog' of 'run',
        # 'kid' of 'child', 'joyful' of 'happy'
        expected_reduced = ["dog", "run", "child", "happy"]
        self.assertEqual(reduced, expected_reduced)

    def test_reduce_word_list_synonyms_manual_missing_groups(self):
        """
        Test that ValueError is raised when synonym_groups is missing for manual method.
        """
        word_list = ["dog", "cat"]
        with self.assertRaises(ValueError):
            self.nlp.reduce_word_list_synonyms(word_list, method="manual")

    def test_reduce_word_list_synonyms_unsupported_method(self):
        """
        Test that ValueError is raised for unsupported reduction methods.
        """
        word_list = ["dog", "cat"]
        with self.assertRaises(ValueError):
            self.nlp.reduce_word_list_synonyms(word_list, method="unsupported_method")

    def test_create_word_groups_with_punctuation(self):
        """
        Test that punctuation is removed and words are converted to lowercase.
        """
        words = [
            "Dogs!",
            "DOGS",
            "Cats?",
            "CATS.",
            "Running,",
            "RUNNING",
            "Children;",
            "CHILDREN",
        ]
        word_groups = self.nlp.create_word_groups(words)

        expected_groups = {
            "dog": {"dogs"},
            "cat": {"cats"},
            "run": {"running"},
            "child": {"children"},
        }

        # Due to duplicates in sets, 'dogs' appears once, etc.
        self.assertEqual(expected_groups, word_groups)

    def test_create_word_groups_with_synonyms(self):
        """
        Test that words are correctly grouped even if they are synonyms.
        """
        words = ["dog", "canine", "puppy"]
        word_groups = self.nlp.create_word_groups(words)

        # Assuming 'canine' and 'puppy' are lemmatized to 'dog' via synonyms
        expected_groups = {"dog": {"dog", "canine", "puppy"}}

        # Depending on implementation, synonyms may not be grouped unless explicitly handled
        self.assertEqual(word_groups, expected_groups)


if __name__ == "__main__":
    unittest.main()
