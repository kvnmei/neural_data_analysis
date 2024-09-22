from sklearn.preprocessing import MultiLabelBinarizer


def create_multilabel_binary_embeddings(data, labels):
    """
    This function creates multilabel binary embeddings from the data and labels.

    Every sample can have multiple labels, and the embeddings are created by creating a binary vector for each label.
    The binary vector is 1 if the label is present in the sample, and 0 otherwise.

    Parameters:
        data (list[list[str]]): A list of strings for each data sample, combined in a list.
        labels: the unique labels for the data samples.

    Returns:

    """
    mlb = MultiLabelBinarizer(classes=labels)
    binary_embeddings = mlb.fit_transform(data)
    return binary_embeddings
