import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from functools import lru_cache
import string
from typing import Optional
import logging
import numpy as np
import re

# Download necessary NLTK resources
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")


class NLPProcessor:
    """
    A class encapsulating various Natural Language Processing (NLP) utilities,
    including stemming, lemmatization, POS tagging, synonym retrieval, word grouping,
    and exclusion of stopwords.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the NLPProcessor with stemmer, lemmatizer, and logger.

        Args:
            logger (logging.Logger, optional): Logger for outputting log information.
        """
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.logger = logger or self._setup_default_logger()
        self.logger.info("Initialized NLPProcessor.")

    @staticmethod
    def _setup_default_logger() -> logging.Logger:
        """
        Sets up a default logger if none is provided.

        Returns:
            logging.Logger: Configured logger.
        """
        logger = logging.getLogger("NLPProcessorLogger")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    @lru_cache(maxsize=None)
    def get_part_of_speech(self, word: str, verbose: bool = False) -> str:
        """
        Get the WordNet part of speech tag for a given word.

        Args:
            word (str): The word to get the POS tag for.
            verbose (bool): Whether to log debug information.

        Returns:
            str: The WordNet POS tag (e.g., NOUN, VERB).
        """
        try:
            tag = nltk.pos_tag([word])[0][1][0].upper()
            tag_dict = {
                "J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV,
            }
            pos = tag_dict.get(tag, wordnet.NOUN)
            if verbose:
                self.logger.debug(f"POS for '{word}': {pos}")
            return pos
        except IndexError:
            # Default to noun if POS tagging fails
            self.logger.warning(
                f"POS tagging failed for word '{word}'. Defaulting to NOUN."
            )
            return wordnet.NOUN

    def get_lemma(self, word: str, verbose: bool = False) -> str:
        """
        Lemmatize a word based on its part of speech.

        Args:
            word (str): The word to lemmatize.
            verbose (bool): Whether to log debug information.

        Returns:
            str: The lemmatized word.
        """
        try:
            pos = self.get_part_of_speech(word)
            lemma = self.lemmatizer.lemmatize(word, pos)
            if verbose:
                self.logger.debug(f"Lemmatized '{word}' to '{lemma}' with POS '{pos}'.")
            return lemma
        except Exception as e:
            # Log the exception and return the original word
            self.logger.warning(f"Lemmatization failed for word '{word}': {e}")
            return word

    def plural_to_singular(self, word: str) -> str:
        # 1. Custom dictionary of known plurals
        known_plurals = {
            "men": "man",
            "women": "woman",
            "children": "child",
            "people": "person",
            "feet": "foot",
            "teeth": "tooth",
            "geese": "goose",
            # Add more known exceptions here
        }
        if word in known_plurals:
            return known_plurals[word]

        # 2. Inflect
        p = inflect.engine()
        singular_inflect = p.singular_noun(word)
        if singular_inflect:  # singular_noun returns False if no singular found
            return singular_inflect

        # 3. NLTK WordNet Lemmatizer
        lemmatizer = WordNetLemmatizer()
        singular_lemma = lemmatizer.lemmatize(word, pos="n")
        if singular_lemma and singular_lemma != word:
            return singular_lemma

        # 4. Pattern's singularize (if pattern is available)
        if pattern_singularize is not None:
            pattern_sing = pattern_singularize(word)
            if pattern_sing and pattern_sing != word:
                return pattern_sing

        # 5. If all else fails, return the original word
        return word

    @lru_cache(maxsize=None)
    def get_synonyms_wordnet(self, word: str, verbose: bool = False) -> list[str]:
        """
        Get the synonyms of a word using WordNet.

        Args:
            word (str): The word to find synonyms for.
            verbose (bool): Whether to log debug information.

        Returns:
            list[str]: A list of synonyms.
        """
        synonyms = set()
        for syn in wordnet.synsets(word.lower()):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        synonym_list = sorted(synonyms)
        if verbose:
            self.logger.debug(f"Synonyms for '{word}': {synonym_list}")
        return synonym_list

    def get_stem(self, word: str, verbose: bool = False) -> str:
        """
        Stem a word using PorterStemmer.

        Args:
            word (str): The word to stem.
            verbose (bool): Whether to log debug information.

        Returns:
            str: The stemmed word.
        """
        stem = self.stemmer.stem(word)
        if verbose:
            self.logger.debug(f"Stemmed '{word}' to '{stem}'.")
        return stem

    def create_excluded_words(self, verbose: bool = False) -> set[str]:
        """
        Create a set of words to filter out based on English stopwords.

        Returns:
            set(str): Set of excluded words.
        """
        excluded_words = set(stopwords.words("english"))

        if verbose:
            self.logger.debug(f"Excluded words: {excluded_words}")
        return excluded_words

    def create_word_groups(
        self, words: list[str] | np.ndarray[str], verbose: bool = False
    ) -> dict[str, set[str]]:
        """
        Create groups of words that are synonyms or different forms (e.g., plural).

        Args:
            words (list[str]): List of unique words.
            verbose (bool): Whether to log debug information.

        Returns:
            dict[str, Set[str]]: Dictionary mapping representative words to their group members.

        """
        # Known plural forms and their singular equivalents
        known_plurals = {
            "men": "man",
            "women": "woman",
            "boys": "boy",
            "girls": "girl",
            "children": "child",
            "people": "person",
            "suits": "suit",
            "shirts": "shirt",
            "hands": "hand",
            "tables": "table",
            "arms": "arm",
            "flowers": "flower",
            "videos": "video",
            "stairs": "stair",
            "wrestlers": "wrestler",
            "woods": "wood",
        }

        # Create a translation table for removing punctuation from words
        remove_punc = str.maketrans("", "", string.punctuation)
        word_groups: dict[str, set[str]] = {}
        for w in words:
            word_lowercase = w.lower()

            # Remove trailing possessive markers: 's or '
            # This removes "'s" at the end of a word, e.g. "sally's" -> "sally"
            word_non_possessive = re.sub(r"'s$", "", word_lowercase)
            word_non_possessive = re.sub(r"'$", "", word_non_possessive)

            word_cleaned = word_non_possessive.translate(remove_punc)

            # Skip if empty after cleaning
            if not word_cleaned:
                continue

            # Get base form (lemma or known plural)
            # Note that get_lemma() from WordNetLemmatizer is used to get singular form
            base_word = known_plurals.get(word_cleaned, self.get_lemma(word_cleaned))

            if base_word not in word_groups:
                word_groups[base_word] = set()

            # Add the original word (present in the caption (as list of strings) and various forms)
            # Because the goal is to create a mapping from the captions (as list of strings) to the base word
            word_groups[base_word].add(w)
            word_groups[base_word].add(word_cleaned)

            if verbose:
                self.logger.debug(f"Grouped '{w}' under '{base_word}'.")

        word_groups = dict(sorted(word_groups.items()))
        if verbose:
            self.logger.info(f"Created word groups: {word_groups}")
        return word_groups

    def reduce_word_list_synonyms(
        self,
        word_list: list[str],
        method: str = "manual",
        synonym_groups: Optional[list[list[str]]] = None,
        verbose: bool = False,
    ) -> list[str]:
        """
        Reduce a list of words by grouping synonyms together.

        Args:
            word_list (list[str]): A list of words to be reduced.
            method (str, optional): The method to used for reducing the word list.
                Either "manual" or "wordnet". Defaults to "manual".
            synonym_groups (list[list[str]]):
                A list of synonym groups where each group is a list of synonyms.
                Required if method is "manual". Defaults to None.
            verbose (bool, optional): Whether to log debug information. Defaults

        Raises:
            ValueError: If method is "manual" but synonym_groups is not provided.
            ValueError: If the method specified is not supported.

        Returns:
            list[str]: A reduced list of words.

        """
        if method == "manual":
            if synonym_groups is None:
                raise ValueError(
                    "synonym_groups must be provided when method is 'manual'"
                )

            reduced_list = []
            synonym_dict = {}

            # Create a dictionary mapping each word to its primary synonym
            for group in synonym_groups:
                if not group:
                    continue  # Skip empty groups
                primary_word = group[0].lower()
                for word in group:
                    synonym_dict[word.lower()] = primary_word
                    if verbose:
                        self.logger.info(
                            f"Mapping '{word.lower()}' to '{primary_word}' in synonym_dict."
                        )

            for word in word_list:
                primary_word = synonym_dict.get(word.lower(), word.lower())
                if primary_word not in reduced_list:
                    reduced_list.append(primary_word)
                    if verbose:
                        self.logger.info(f"Added '{primary_word}' to reduced_list.")

            self.logger.info(f"Reduced word list using manual method: {reduced_list}")
            return reduced_list
        elif method == "wordnet":
            reduced_list = []
            visited_words = set()
            word_list_lower = [word.lower() for word in word_list]

            for word in word_list_lower:
                if word in visited_words:
                    continue
                synonyms = self.get_synonyms_wordnet(word)
                combined_words = [w for w in word_list_lower if w in synonyms]
                if combined_words:
                    representative = combined_words[0]
                    reduced_list.append(representative)
                    if verbose:
                        self.logger.info(
                            f"Added '{representative}' as representative of {combined_words}."
                        )
                visited_words.update(combined_words)

            self.logger.info(f"Reduced word list using WordNet method: {reduced_list}")
            return reduced_list

        else:
            raise ValueError(
                f"Method '{method}' is not supported. Choose 'manual' or 'wordnet'."
            )
