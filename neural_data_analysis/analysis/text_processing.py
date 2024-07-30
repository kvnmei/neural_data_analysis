import nltk
from nltk.corpus import stopwords, wordnet
from nltk.data import find
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def download_resource(resource):
    try:
        # TODO: for some reason, the find function never finds the resource and skips to the nltk.download
        find(resource)
    except LookupError:
        nltk.download(resource, quiet=True)


download_resource("stopwords")
download_resource("wordnet")
download_resource("averaged_perceptron_tagger")


# Function to get the part of speech for lemmatization
def get_pos(word):
    """Get part of speech."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def create_excluded_words():
    # create a set of words to filter out

    excluded_words = set(stopwords.words("english"))
    # filter_words.update(
    #     [
    #     ]
    # )

    return excluded_words


def get_lemma(word):
    return lemmatizer.lemmatize(word, get_pos(word))


def get_synonyms_wordnet(word: str):
    """Get the synonyms of a word using WordNet.

    Parameters:
        word (str): The word to find synonyms for.

    Returns:
        synonym_list: A list of synonyms.
    """
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    synonym_list = list(synonyms)
    return synonym_list


def get_stem(word):
    return stemmer.stem(word)


def create_word_groups(words):
    # Known plural forms and their singular equivalents
    known_plurals = {
        "men": "man",
        "women": "woman",
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
    words = [word for word in words if word not in string.punctuation]

    word_groups = {}
    # group words by their stem
    for word in words:
        lemma = known_plurals.get(word, get_lemma(word))
        if lemma not in word_groups:
            word_groups[lemma] = set()
        word_groups[lemma].add(word)

    # combined words with same meaning (synonyms)
    # combined_groups = {}
    # for lemma, group in word_groups.items():
    #     # the representative word is arbitrarily the first word of the first lemma group
    #     representative = list(group)[0]
    #     combined_groups[representative] = set(group)
    #     # for every word in the word group, get its synonym
    #     for word in list(group):
    #         synonyms = get_synonyms_wordnet(word)
    #         # for every synonym, if its lemma is a key in word_groups, then save all the words in the word group to the representative word
    #         for synonym in synonyms:
    #             synonym_lemma = get_lemma(synonym)
    #             if synonym_lemma in word_groups:
    #                 combined_groups[representative].update(word_groups[synonym_lemma])

    print("Word groups created.")
    return word_groups


def reduce_word_list_synonyms(word_list, method: str = "manual", synonym_groups=None):
    """
    Given a list of words, reduce the list by grouping synonyms together.

    Parameters:
        word_list (list): A list of words.
        method (str): The method to use to reduce the list of words. Either "manual" or "wordnet".
        synonym_groups (list): A list of lists of synonyms.

    Returns:
        list: A reduced list of words.

    """
    if method == "manual":
        reduced_list = []
        synonym_dict = {}

        # Create a dictionary mapping each word to its primary synonym
        for group in synonym_groups:
            primary_word = group[0]
            for word in group:
                synonym_dict[word] = primary_word

        for word in word_list:
            primary_word = synonym_dict.get(word, word)
            if primary_word not in reduced_list:
                reduced_list.append(primary_word)

        return reduced_list
    elif method == "wordnet":
        reduced_list = []
        visited_words = set()
        for word in word_list:
            if word in visited_words:
                continue
            synonyms = get_synonyms_wordnet(word)
            combined_words = [w for w in word_list if w in synonyms]
            if combined_words:
                reduced_list.append(
                    combined_words[0]
                )  # You can choose how to combine, here we take the first
            visited_words.update(combined_words)
        return reduced_list

    else:
        raise ValueError("Method not supported")
