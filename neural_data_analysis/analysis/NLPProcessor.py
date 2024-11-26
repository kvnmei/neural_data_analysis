import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from functools import lru_cache
import string
from typing import Optional
import logging

# Download necessary NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")


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
    def get_part_of_speech(self, word: str) -> str:
        """
        Get the WordNet part of speech tag for a given word.

        Args:
            word (str): The word to get the POS tag for.

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
            self.logger.debug(f"POS for '{word}': {pos}")
            return pos
        except IndexError:
            # Default to noun if POS tagging fails
            self.logger.warning(
                f"POS tagging failed for word '{word}'. Defaulting to NOUN."
            )
            return wordnet.NOUN

    def get_lemma(self, word: str) -> str:
        """
        Lemmatize a word based on its part of speech.

        Args:
            word (str): The word to lemmatize.

        Returns:
            str: The lemmatized word.
        """
        try:
            pos = self.get_part_of_speech(word)
            lemma = self.lemmatizer.lemmatize(word, pos)
            self.logger.debug(f"Lemmatized '{word}' to '{lemma}' with POS '{pos}'.")
            return lemma
        except Exception as e:
            # Log the exception and return the original word
            self.logger.warning(f"Lemmatization failed for word '{word}': {e}")
            return word

    @lru_cache(maxsize=None)
    def get_synonyms_wordnet(self, word: str) -> list[str]:
        """
        Get the synonyms of a word using WordNet.

        Args:
            word (str): The word to find synonyms for.

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
        self.logger.debug(f"Synonyms for '{word}': {synonym_list}")
        return synonym_list

    def get_stem(self, word: str) -> str:
        """
        Stem a word using PorterStemmer.

        Args:
            word (str): The word to stem.

        Returns:
            str: The stemmed word.
        """
        stem = self.stemmer.stem(word)
        self.logger.debug(f"Stemmed '{word}' to '{stem}'.")
        return stem

    def create_excluded_words(self) -> set[str]:
        """
        Create a set of words to filter out based on English stopwords.

        Returns:
            set(str): Set of excluded words.
        """
        excluded_words = set(stopwords.words("english"))
        self.logger.debug(f"Excluded words: {excluded_words}")
        return excluded_words

    def create_word_groups(self, words: list[str]) -> dict[str, set[str]]:
        """
        Create groups of words that are synonyms or different forms (e.g., plural).

        Args:
            words (list[str]): List of unique words.

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

        # Remove punctuation from words
        cleaned_words = [
            word.lower() for word in words if word not in string.punctuation
        ]
        self.logger.debug(f"Cleaned words (no punctuation, lowercase): {cleaned_words}")

        word_groups: dict[str, set[str]] = {}
        for word in cleaned_words:
            # Get the base form of the word
            base_word = known_plurals.get(word, self.get_lemma(word))
            if base_word not in word_groups:
                word_groups[base_word] = set()
            word_groups[base_word].add(word)
            self.logger.debug(f"Grouped '{word}' under '{base_word}'.")

        self.logger.info(f"Created word groups: {word_groups}")
        return word_groups

    def reduce_word_list_synonyms(
        self,
        word_list: list[str],
        method: str = "manual",
        synonym_groups: Optional[list[list[str]]] = None,
    ) -> list[str]:
        """
        Reduce a list of words by grouping synonyms together.

        Args:
            word_list (list[str]): A list of words to be reduced.
            method (str, optional): The method to used for reducing the word list.
                Either "manual" or "wordnet". Defaults to "manual".
            synonym_groups (Optional[list[list[str]]], optional):
                A list of synonym groups where each group is a list of synonyms.
                Required if method is "manual". Defaults to None.

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
                    self.logger.debug(
                        f"Mapping '{word.lower()}' to '{primary_word}' in synonym_dict."
                    )

            for word in word_list:
                primary_word = synonym_dict.get(word.lower(), word.lower())
                if primary_word not in reduced_list:
                    reduced_list.append(primary_word)
                    self.logger.debug(f"Added '{primary_word}' to reduced_list.")

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
                    self.logger.debug(
                        f"Added '{representative}' as representative of {combined_words}."
                    )
                visited_words.update(combined_words)

            self.logger.info(f"Reduced word list using WordNet method: {reduced_list}")
            return reduced_list

        else:
            raise ValueError(
                f"Method '{method}' is not supported. Choose 'manual' or 'wordnet'."
            )
