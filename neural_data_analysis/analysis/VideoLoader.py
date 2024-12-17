import logging
from abc import ABC
from pathlib import Path
import polars as pl
import pandas as pd

import cv2
import h5py
import numpy as np
import torch
from numpy import ndarray
from torchvision import transforms
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer
from .embedder_utils import embedder_from_spec
from .NLPProcessor import NLPProcessor
from ..utils import add_default_repr, setup_default_logger
from ..statistics import shuffle_binary_array_by_group, generate_random_clusters

example_video_loader_config = {
    "video_path": "data/stimulus/short_downsampled.avi",
    "embedders_to_use": ["moten", "gist", "rgbhsvl", "face_regressors"],
    "word_labels": {
        "set_controls": True,
        "shuffled_controls": True,
        "n_shuffled_controls": 1,
        "shifted_controls": True,
        "n_shifted_controls": 1,
    },
}


@add_default_repr
class VideoLoader(ABC):
    def __init__(
        self,
        config: dict,
        logger: logging.Logger = None,
        embedder_configs: dict = None,
    ):
        """
        Initialize the VideoLoader class with the configuration parameters.

        Parameters:
            config (dict): dictionary of configuration parameters, typically saved and loaded as a yaml file.

        """

        self.logger = logger or setup_default_logger()
        self.logger.info("========== Initializing VideoLoader class ==========")
        self.config: dict = config

        # Load video frames
        self.frames = self.load_video_frames()
        self.frames_info = None
        self.frame_captions = None
        self.original_frame_index = np.arange(self.frames.shape[0])

        self.nlp_processor = None
        self.blip2_control_labels: list[str] = []
        self.blip2_words_df: pd.DataFrame = pd.DataFrame()

        # Load embeddings
        self.embedders_to_use = self.config.get("embedders_to_use")
        self.embedder_configs = embedder_configs
        self.embeddings: dict = self.create_frame_embeddings()

    def load_video_frames(
        self,
        video_path: str | None = None,
        reader: str = "opencv",
        limit: int | None = None,
    ) -> np.ndarray:
        """
        Loads the frames of a video as a NumPy array.

        Args:
            video_path (str): The path to the video file. If None, uses config["video_path"].
            reader (str): The video reader to use. Currently only "opencv" is supported.
            limit (int, optional): The maximum number of frames to load, if provided.

        Returns:
            frames (np.ndarray): The frames of the video as a NumPy array of shape (num_frames, height, width, channels).
        """
        if video_path is None:
            video_path = self.config.get(
                "video_path", "data/stimulus/short_downsampled.avi"
            )

        video_path_obj = Path(video_path)
        if not video_path_obj.exists():
            raise FileNotFoundError(f"Video path does not exist: {video_path}")

        self.logger.info(f"Loading video [{video_path}] using [{reader}]...")

        if reader.lower() == "opencv":
            # Attempt to open the video file
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Cannot open video file {video_path}")

            # Try to get the total frame count for progress bar
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count_str = frame_count if frame_count > 0 else "unknown"
            self.logger.info(f"Detected {frame_count_str} frames in the video.")

            frames = []
            frames_to_read = (
                min(frame_count, limit)
                if (limit is not None and frame_count > 0)
                else limit
            )

            # Setup progress bar with known or unknown total
            total_for_pbar = frame_count if frame_count > 0 else None

            with tqdm(
                total=total_for_pbar, desc="Loading frames", unit="frames"
            ) as pbar:
                idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        # End of video or read error
                        break
                    frames.append(frame)
                    idx += 1
                    pbar.update(1)
                    if frames_to_read is not None and idx >= frames_to_read:
                        # Reached the limit of frames to read
                        break

            cap.release()
        else:
            raise ValueError(
                f"Reader '{reader}' is not supported. Currently only 'opencv' is supported."
            )

        frames_np = np.array(frames)
        num_frames = frames_np.shape[0]

        self.logger.info(f"Loaded {num_frames} frames from the video.")
        self.logger.info(f"Frames shape: {frames_np.shape}")

        return frames_np

    def _tokenize_captions(
        self, captions: list[str]
    ) -> tuple[list[list[str]], list[str]]:
        """
        Tokenize captions into words.

        Args:
            captions (list[str]): List of captions.

        Returns:
            tuple[list[list[str]], list[str]]: Tuple containing list of tokenized captions and all words.
        """
        split_captions = [
            caption.split() for caption in captions
        ]  # Split by white space by default
        all_words = [word for caption in split_captions for word in caption]
        self.logger.info(f"Tokenized captions into {len(all_words)} non-unique words.")
        return split_captions, all_words

    def _filter_words(
        self,
        words: np.ndarray,
        counts: np.ndarray,
        word_groups: dict[str, set[str]],
        occurrence_minimum: int,
    ) -> list[str]:
        """
        Filter words based on occurrence minimum and excluded words.
        Default occurrence minimum is the median count.

        Args:
            words (np.ndarray): Array of unique words.
            counts (np.ndarray): Array of word counts.
            word_groups (dict[str, set[str]]): Dictionary of word groups.
            occurrence_minimum (Optional[int]): Minimum occurrence to include a word.

        Returns:
            list[str]: Filtered list of labels.
        """
        if self.config["word_labels"].get("use_saved_labels", False):
            saved_labels_path = self.config["word_labels"].get("saved_labels_path", "")
            if not Path(saved_labels_path).exists():
                self.logger.error(
                    f"Saved labels file '{saved_labels_path}' does not exist."
                )
                raise FileNotFoundError(
                    f"Saved labels file '{saved_labels_path}' does not exist."
                )

            try:
                saved_labels = np.loadtxt(
                    self.config["word_labels"]["saved_labels_path"],
                    delimiter=",",
                    dtype=str,
                )
                saved_labels = [label.lower() for label in saved_labels]
                filtered_word_groups = {
                    key: word_groups[key] for key in saved_labels if key in word_groups
                }
                self.logger.info(
                    f"Loaded {len(filtered_word_groups)} saved labels from '{saved_labels_path}'."
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to load saved labels from '{saved_labels_path}': {e}"
                )
                raise
        else:
            if occurrence_minimum is None:
                occurrence_minimum = int(np.median(counts))
                self.logger.info(
                    f"Set occurrence_minimum to median count: {occurrence_minimum}"
                )

            excluded_words = self.nlp_processor.create_excluded_words()
            self.logger.debug(f"Excluded words: {excluded_words}")

            filtered_words_counts = [
                (word, count)
                for word, count in zip(words, counts)
                if word.lower() not in excluded_words and count >= occurrence_minimum
            ]
            if filtered_words_counts:
                filtered_words, _ = zip(*filtered_words_counts)
                filtered_words = list(filtered_words)
            else:
                filtered_words = []

            self.logger.info(
                f"{len(filtered_words)} words remain after filtering with occurrence minimum [{occurrence_minimum}]."
            )

            filtered_word_groups = {
                key: word_groups[key] for key in filtered_words if key in word_groups
            }

        labels = list(filtered_word_groups.keys())
        self.logger.info(f"{len(labels)} unique words remain after grouping.")
        return labels

    def _map_captions_to_labels(
        self,
        split_captions: list[list[str]],
        word_groups: dict[str, set[str]],
    ) -> list[list[str]]:
        """
        Map each caption to its corresponding labels based on word groups.

        Args:
            split_captions (list[list[str]]): Tokenized captions.
            word_groups (dict[str, set[str]]): Dictionary of word groups.

        Returns:
            list[list[str]]: List of labels for each caption.
        """
        frame_word_labels = []
        for caption in split_captions:
            caption_labels = set()
            for word in caption:
                word_lower = word.lower()
                for representative, group in word_groups.items():
                    if word_lower in group:
                        caption_labels.add(representative)
                        break  # Stop searching once a match is found
            frame_word_labels.append(list(caption_labels))
        self.logger.debug(
            f"Mapped captions to labels for {len(frame_word_labels)} frames."
        )
        return frame_word_labels

    def _create_control_encodings(
        self, n_frames: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, list[str]]:
        """
        Create control encodings with predefined patterns.
        """
        self.logger.info(
            "Creating control encodings. E.g., half-half, random, all-zero, all-one, clustered."
        )
        control_labels = [
            "half_half1",
            "half_half2",
            "random1",
            "random2",
            "random3",
            "all_zero",
            "all_one",
            "clustered1",
            "clustered2",
            "clustered3",
        ]

        half_length = n_frames // 2
        half_half1 = np.zeros(n_frames, dtype=int)
        half_half1[:half_length] = 1

        half_half2 = np.zeros(n_frames, dtype=int)
        half_half2[half_length:] = 1

        random1 = rng.choice([0, 1], size=n_frames, p=[0.75, 0.25])
        random2 = rng.choice([0, 1], size=n_frames, p=[0.5, 0.5])
        random3 = rng.choice([0, 1], size=n_frames, p=[0.25, 0.75])

        all_zero = np.zeros(n_frames, dtype=int)
        all_one = np.ones(n_frames, dtype=int)

        clustered1 = generate_random_clusters(
            n_frames, num_clusters=100, max_cluster_size=100, rng=rng
        )
        clustered2 = generate_random_clusters(
            n_frames, num_clusters=10, max_cluster_size=1000, rng=rng
        )
        clustered3 = generate_random_clusters(
            n_frames, num_clusters=5, max_cluster_size=2000, rng=rng
        )

        control_encodings = np.vstack(
            [
                half_half1,
                half_half2,
                random1,
                random2,
                random3,
                all_zero,
                all_one,
                clustered1,
                clustered2,
                clustered3,
            ]
        ).T

        return control_encodings, control_labels

    def _create_shuffled_encodings(
        self,
        encodings: np.ndarray,
        labels: list[str],
        n_frames: int,
        n_shuffled_controls: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Create shuffled encodings by shuffling existing encodings.
        """
        shuffled_encodings = []
        shuffled_labels = []

        self.logger.info(
            f"Creating {n_shuffled_controls} shuffled controls for each of {len(labels)} labels."
        )
        for i, label in enumerate(tqdm(labels, desc=f"Creating shuffled controls")):
            word_encoding = encodings[:, i]
            random_seeds = rng.choice(
                np.arange(1, n_frames), size=n_shuffled_controls, replace=False
            )
            for seed in random_seeds:
                shuffled_encoding = shuffle_binary_array_by_group(
                    word_encoding, seed=seed
                )
                shuffled_encodings.append(shuffled_encoding.reshape(-1, 1))
                shuffled_labels.append(f"{label}_shuffle_{seed}")

        shuffled_encodings = np.hstack(shuffled_encodings)
        return shuffled_encodings, shuffled_labels

    def _create_shifted_encodings(
        self,
        encodings: np.ndarray,
        labels: list[str],
        n_frames: int,
        n_shifted_controls: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, list[str]]:
        """
        Create shifted encodings by shifting existing encodings.
        """
        shifted_encodings = []
        shifted_labels = []

        shift_range = np.arange(750, n_frames - 750)
        if len(shift_range) < n_shifted_controls:
            shift_values = shift_range
        else:
            shift_values = rng.choice(
                shift_range, size=n_shifted_controls, replace=False
            )

        self.logger.info(
            f"Creating {n_shifted_controls} shifted controls for each of {len(labels)} labels."
        )
        for i, label in enumerate(tqdm(labels, desc=f"Creating shifted controls")):
            word_encoding = encodings[:, i]
            for shift in shift_values:
                shifted_encoding = np.roll(word_encoding, shift)
                shifted_encodings.append(shifted_encoding.reshape(-1, 1))
                shifted_labels.append(f"{label}_shift_{shift}")

        shifted_encodings = np.hstack(shifted_encodings)
        return shifted_encodings, shifted_labels

    def create_multilabel_one_hot_encoding_from_captions(
        self,
        occurrence_minimum: int = None,
        set_controls: bool = True,
        shuffled_controls: bool = True,
        n_shuffled_controls: int = 10,
        shifted_controls: bool = True,
        n_shifted_controls: int = 10,
    ) -> np.ndarray:
        """
        Create multilabel binary word occurrence vectors based on the occurrence of words in image captions.
        Each image is represented by a binary vector indicating the presence of specific words in its caption.

        Parameters:
            occurrence_minimum (int, optional): The minimum number of occurrences of a word to be included in the embedding.
            set_controls (bool): Whether to add a nonsense control group to the embeddings.
            shuffled_controls (bool): Whether to add shuffled versions of the word label embeddings to the embeddings.
            n_shuffled_controls (int): The number of shuffled controls to add.
            shifted_controls (bool): Whether to add shifted embeddings by a certain offset of frames to the embeddings.
            n_shifted_controls (int): The number of shifted controls to add.

        Returns:
            np.ndarray: Multilabel binary embeddings of the frames.
        """
        self.logger.info(
            "----- Creating multilabel one-hot encoding for image captions -----"
        )

        # Initialize NLPProcessor
        self.nlp_processor = NLPProcessor(logger=self.logger)

        # Tokenize captions into words and count word occurrences
        split_captions, all_words = self._tokenize_captions(self.frame_captions)
        words, counts = np.unique(all_words, return_counts=True)
        self.logger.info(f"Found {len(words)} unique words in captions.")

        # Create word groups (e.g., synonyms, plural forms)
        word_groups = self.nlp_processor.create_word_groups(words=words)
        self.logger.info(
            "Created word groups for mapping caption words to representative base words."
        )

        # Map captions to list of representative words that will be compared to target word labels to decode
        frame_word_labels = self._map_captions_to_labels(split_captions, word_groups)

        # Filter words based on occurrence and excluded words, creating a set of word labels to decode
        if occurrence_minimum is None:
            # by default, set the occurrence minimum to be a fraction of the frames
            occurrence_minimum = len(self.frames) // 40
        labels = self._filter_words(
            words=words,
            counts=counts,
            word_groups=word_groups,
            occurrence_minimum=occurrence_minimum,
        )

        # Create multilabel one-hot encoding
        mlb = MultiLabelBinarizer(classes=labels)
        one_hot_matrix = mlb.fit_transform(frame_word_labels)
        self.logger.info(
            f"Created multilabel one-hot matrix with shape {one_hot_matrix.shape}."
        )

        # Add control encodings
        control_labels = []
        rng_seed = self.config.get("seed", None)
        rng = np.random.default_rng(rng_seed)
        n_frames = one_hot_matrix.shape[0]

        if set_controls:
            control_encodings, controls = self._create_control_encodings(
                n_frames=n_frames, rng=rng
            )
            one_hot_matrix = np.hstack((one_hot_matrix, control_encodings))
            control_labels.extend(controls)
            self.logger.info(f"Added {len(controls)} control encodings.")

        if shuffled_controls:
            shuffled_encodings, shuffled_labels = self._create_shuffled_encodings(
                encodings=one_hot_matrix[:, : len(labels)],
                labels=labels,
                n_frames=n_frames,
                n_shuffled_controls=n_shuffled_controls,
                rng=rng,
            )
            one_hot_matrix = np.hstack((one_hot_matrix, shuffled_encodings))
            control_labels.extend(shuffled_labels)
            self.logger.info(
                f"Added {len(shuffled_labels)} shuffled control encodings."
            )

        if shifted_controls:
            shifted_encodings, shifted_labels = self._create_shifted_encodings(
                one_hot_matrix[:, : len(labels)],
                labels,
                n_frames,
                n_shifted_controls,
                rng,
            )
            one_hot_matrix = np.hstack((one_hot_matrix, shifted_encodings))
            control_labels.extend(shifted_labels)
            self.logger.info(f"Added {len(shifted_labels)} shifted control encodings.")

        # Update labels with control labels
        labels.extend(control_labels)

        # Create DataFrame with word labels and their counts across frames
        frame_counts = one_hot_matrix.sum(axis=0)
        word_labels_df = pd.DataFrame({"word": labels, "count_frames": frame_counts})

        # Store attributes
        self.blip2_control_labels = control_labels
        self.blip2_words_df = word_labels_df
        self.logger.info("COMPLETED: creation of multilabel one-hot encodings.\n")

        return one_hot_matrix.astype(int)

    def _load_embeddings_from_file(
        self, file_path: Path
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Helper method to load embeddings from an HDF5 file."""
        try:
            with h5py.File(file_path, "r") as f:
                # If the file contains multiple datasets, return a dict, otherwise return tensor
                if len(f.keys()) > 1:
                    return {k: f[k][:] for k in f.keys()}
                return f["data"][:]
        except Exception as e:
            self.logger.error(
                f"Error loading precomputed embeddings from {file_path}: {e}"
            )
            raise

    def _save_embeddings_to_file(
        self, file_path: Path, embedding: torch.Tensor | dict[str, torch.Tensor]
    ) -> None:
        """Helper method to save embeddings to an HDF5 file."""
        try:
            with h5py.File(file_path, "w") as h5file:
                if isinstance(embedding, dict):
                    for key, value in embedding.items():
                        h5file.create_dataset(key, data=value)
                else:
                    h5file.create_dataset("data", data=embedding)
        except Exception as e:
            self.logger.error(f"Error saving embeddings to {file_path}: {e}")
            raise

    def _embed_frames(
        self,
        embedder_name: str,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Embeds frames using a specified embedder and returns the resulting embeddings.
        """
        if self.frames is None:
            self.logger.info("Frames not loaded, loading video frames now...")
            self.frames, self.frames_info = self.load_video_frames()

        images = torch.stack([transforms.ToTensor()(frame) for frame in self.frames])
        image_embedder = embedder_from_spec(
            embedder_name, embedder_configs=self.embedder_configs
        )
        return image_embedder.embed(images)

    def _process_blip2_embeddings(self, embedding: list[bytes | str]) -> ndarray:
        """
        Helper to process BLIP2 embeddings. Convert bytes to strings, then
        create multilabel encodings.
        """
        # Convert bytes to strings
        self.frame_captions = [
            (item.decode("utf-8").strip() if isinstance(item, bytes) else item.strip())
            for item in embedding
        ]
        self.logger.info("CONVERTED: [blip2] embeddings from bytes to strings.")

        return self.create_multilabel_one_hot_encoding_from_captions(
            occurrence_minimum=self.config["word_labels"]["occurrence_minimum"],
            set_controls=self.config["word_labels"]["set_controls"],
            shuffled_controls=self.config["word_labels"]["shuffled_controls"],
            n_shuffled_controls=self.config["word_labels"]["n_shuffled_controls"],
            shifted_controls=self.config["word_labels"]["shifted_controls"],
            n_shifted_controls=self.config["word_labels"]["n_shifted_controls"],
        )

    def create_frame_embeddings(
        self,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        """
        Create video frame embeddings using specified embedders in the config.

        Returns:
            embeddings (dict): A dictionary of embeddings, with embedder names as keys.
        """
        self.logger.info("----- Creating video frame embeddings -----")
        embeddings: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {}
        save_dir = Path("precomputed/")
        save_dir.mkdir(parents=True, exist_ok=True)

        for embedder_name in self.embedders_to_use:
            embedding_config = self.embedder_configs.get(embedder_name, {})
            embedding_name = embedding_config.get("embedding_name", embedder_name)
            embeddings_filename = f"video_frame_embeddings_{embedder_name}.h5"
            embeddings_path = save_dir / embeddings_filename

            self.logger.info(f"Creating embeddings for [{embedding_name}]...")
            # Load or create embeddings
            if embeddings_path.exists():
                self.logger.info(
                    f"Precomputed embeddings for [{embedding_name}] found."
                )
                self.logger.info(
                    "Loading precomputed embeddings for [{embedding_name}]"
                    f" from [{str(save_dir / embeddings_filename)}]"
                )
                embedding = self._load_embeddings_from_file(embeddings_path)
                self.logger.info(f"LOADED: embeddings for [{embedding_name}]")
            else:
                self.logger.info(
                    f"Precomputed embeddings for [{embedding_name}] not found. Creating now..."
                )
                embedding = self._embed_frames(embedder_name)
                self.logger.info(f"COMPLETED: Embedded images with [{embedder_name}].")
                self.logger.info(
                    f"Saving embedding to {str(save_dir / embeddings_filename)}..."
                )
                self._save_embeddings_to_file(embeddings_path, embedding)
                self.logger.info(f"SAVED: {embedding_name} image embeddings.")

            # Process BLIP2 case
            if embedding_name == "blip2":
                embedding = self._process_blip2_embeddings(embedding)

            embeddings[embedding_name] = embedding

        logging.info("COMPLETED: Created video frame embeddings.")
        return embeddings
