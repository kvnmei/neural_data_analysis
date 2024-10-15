"""
Plotting functions

Functions:
    plot_scatter_with_images
    plot_variance_explained
    plot_ecdf
    elbow_curve
    pairwise_distance_heatmap
    plot_tsne_projection


"""

import base64
import io
import os
from pathlib import Path

from typing import Union, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bokeh.io import output_file
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure, save, show
from bokeh.transform import factor_cmap
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap


# =======================================
# Function: plot_histogram
# ========================================
def plot_histogram(data: pd.DataFrame) -> None:
    """
    Given a list of values, plot a histogram of the count/percentage of the values.

    Returns:
        None
    """
    fig, ax = plt.subplots()
    sns.histplot(data, kde=False, ax=ax, stat="count")
    plt.show()


# =======================================
# Function: plot_scatter
# ========================================
def plot_scatter(
    data: pd.DataFrame, x_column: str, y_column: str, backend: str = "seaborn", **kwargs
) -> None:
    """
    Plots a scatter plot of the data with optional hue, palette, and annotations.

    Parameters:
        data (pd.DataFrame): The DataFrame containing the data to plot.
        x_column (str): Column name for the x-axis.
        y_column (str): Column name for the y-axis.
        backend (str): Plotting package to use. Options: "seaborn" or "matplotlib". Default is "seaborn".

    Optional kwargs:
        figsize (tuple, optional): Size of the figure. Default is (12, 8).
        title (str, optional): Title of the plot. Default is "Scatter Plot".
        xlabel (str, optional): Label for the x-axis. Default is the name of x_column.
        ylabel (str, optional): Label for the y-axis. Default is the name of y_column.
        xtick_labels (list, optional): Custom labels for the x-axis ticks.
        ytick_labels (list, optional): Custom labels for the y-axis ticks.
        add_diagonal (bool, optional): Whether to add a diagonal line y = x. Default is False.
        save_dir (str or Path, optional): Directory to save the plot. Default is "plots".
        save_filename (str, optional): File name to save the plot. Default is "scatter_plot.png".
        hue (str, optional): Column name to map plot aspects (e.g., color). Default is None.
        palette (str or list, optional): Palette for the hue mapping. Default is None.
        word_labels (str or list of str, optional): Column name or list containing labels to annotate points. Default is None.
        annotate (bool, optional): Whether to annotate points with labels. Requires 'word_labels'. Default is False.
        size (str or float, optional): Column name or value to control marker sizes. Default is None.
        sizes (tuple of (float, float), optional): Min and max size for mapping marker sizes when 'size' is a column. Default is (20, 200).

    Returns:
        None
    """
    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Extract optional parameters with defaults
    figsize: tuple = kwargs.get("figsize", (12, 8))
    title: str = kwargs.get("title", "Scatter Plot")
    xlabel: str = kwargs.get("xlabel", x_column)
    ylabel: str = kwargs.get("ylabel", y_column)
    xtick_labels: Optional[List[str]] = kwargs.get("xtick_labels", None)
    ytick_labels: Optional[List[str]] = kwargs.get("ytick_labels", None)
    add_diagonal: bool = kwargs.get("add_diagonal", False)
    save_dir: Path = Path(kwargs.get("save_dir", "plots"))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_filename: str = kwargs.get("save_filename", "scatter_plot.png")
    hue: Optional[str] = kwargs.get("hue", None)
    palette: Optional[Union[str, List[str]]] = kwargs.get("palette", None)
    word_labels: Optional[Union[str, List[str]]] = kwargs.get("word_labels", None)
    annotate: bool = kwargs.get("annotate", False)
    size: Optional[Union[str, float]] = kwargs.get("size", None)
    sizes: Tuple[float, float] = kwargs.get("sizes", (20, 200))

    # Validate backend
    if backend not in ["seaborn", "matplotlib"]:
        raise ValueError("Unsupported backend. Choose 'seaborn' or 'matplotlib'.")

    # Create the figure and axes explicitly
    fig, ax = plt.subplots(figsize=figsize)

    if backend == "seaborn":
        # Prepare scatterplot parameters
        scatter_kwargs = {
            "data": data,
            "x": x_column,
            "y": y_column,
            "ax": ax,
            "s": 100,  # Default marker size
            "alpha": 0.7,
            "edgecolor": "w",
        }

        # Add hue and palette if provided
        if hue:
            scatter_kwargs["hue"] = hue
            scatter_kwargs["palette"] = palette

        # Add size and sizes if provided
        if size:
            scatter_kwargs["size"] = size
            scatter_kwargs["sizes"] = sizes
        else:
            scatter_kwargs["s"] = 100  # Default marker size

        # Create the scatter plot
        scatter = sns.scatterplot(**scatter_kwargs)

    elif backend == "matplotlib":
        # Prepare scatterplot parameters
        scatter_kwargs = {
            "x": data[x_column],
            "y": data[y_column],
            "s": 100,  # Default marker size
            "alpha": 0.7,
            "edgecolors": "w",
            "c": "blue",  # Default color
            "marker": "o",
        }

        # Add hue and palette if provided
        if hue:
            if hue not in data.columns:
                raise ValueError(f"Hue column '{hue}' not found in data.")
            unique_values = data[hue].unique()
            if isinstance(palette, str):
                palette_colors = sns.color_palette(palette, len(unique_values))
            elif isinstance(palette, list):
                if len(palette) < len(unique_values):
                    raise ValueError("Not enough colors provided in palette.")
                palette_colors = palette
            else:
                palette_colors = sns.color_palette("deep", len(unique_values))

            color_mapping = {
                val: palette_colors[i] for i, val in enumerate(unique_values)
            }
            colors = data[hue].map(color_mapping)

            scatter_kwargs["c"] = colors

        # Add size if provided
        if size:
            if isinstance(size, str):
                if size not in data.columns:
                    raise ValueError(f"Size column '{size}' not found in data.")
                # Normalize sizes to the specified range
                size_values = data[size]
                size_norm = (size_values - size_values.min()) / (size_values.max() - size_values.min())
                scatter_kwargs["s"] = size_norm * (sizes[1] - sizes[0]) + sizes[0]
            elif isinstance(size, (int, float)):
                scatter_kwargs["s"] = size
            else:
                raise TypeError("Size must be a column name (str) or a numeric value (int or float).")
        else:
            scatter_kwargs["s"] = 100  # Default marker size

        # Create the scatter plot
        scatter = ax.scatter(**scatter_kwargs)

        # Create a legend if hue is used
        if hue:
            handles = []
            for val, color in color_mapping.items():
                handles.append(
                    plt.Line2D(
                        [], [], marker="o", linestyle="", color=color, label=str(val)
                    )
                )
            ax.legend(
                handles=handles, title=hue, bbox_to_anchor=(1.05, 1), loc="upper left"
            )

    # Set title and axis labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)

    # Set custom tick labels if provided
    if xtick_labels:
        ax.set_xticklabels(xtick_labels)
    if ytick_labels:
        ax.set_yticklabels(ytick_labels)

    # Add diagonal line if requested
    if add_diagonal:
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        min_val = min(x_min, y_min)
        max_val = max(x_max, y_max)
        ax.plot([min_val, max_val], [min_val, max_val], ls="--", c="red", label="y = x")
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.legend()

    # Annotate points with word labels if requested
    if annotate and word_labels is not None:
        if isinstance(word_labels, str):
            if word_labels not in data.columns:
                raise ValueError(
                    f"word_labels column '{word_labels}' not found in data."
                )
            labels = data[word_labels]
        elif isinstance(word_labels, list):
            if len(word_labels) != len(data):
                raise ValueError(
                    "Length of word_labels list must match number of data points."
                )
            labels = word_labels
        else:
            raise TypeError(
                "word_labels must be a column name (str) or a list of strings."
            )

        for i, label in enumerate(labels):
            ax.text(
                data.iloc[i][x_column] + 0.005 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                data.iloc[i][y_column] + 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label,
                horizontalalignment="left",
                size="medium",
                color="black",
                weight="normal",
            )

    # Adjust layout for better spacing
    plt.tight_layout()

    # Save the plot if save_filename is provided
    if save_filename:
        save_path = save_dir / save_filename
        fig.savefig(save_path)
        print(f"Plot successfully saved to {save_path}")

    # Display the plot
    plt.show()


def plot_scatter_with_images(
    data_points: np.ndarray,
    images: List[Image.Image],
    descriptors: dict = None,
    color_by: str = None,
    title: str = "Scatter plot with image hover",
    legend_title: str = None,
    save_dir: Path = Path("plots"),
    filename: str = "scatterplot_with_hover.html",
    show_plot: bool = False,
) -> None:
    """
    Create a 2-D bokeh scatterplot in HTML with hover tool that displays images.

    Args:
        data_points (np.ndarray): 2-D array of the points to plot
        images (list): images that correspond to number of data points
        descriptors (dict, optional): labels for each data point
        color_by (str, optional): which variable (key) in descriptors to color the points by
        title:
        legend_title:
        save_dir:
        filename:
        show_plot:

    Returns:
        None

    Example:
        plot_scatter_with_images()

    Note:
        Displays the plot or saves the plot to HTML file.
    """

    # Convert the PIL images to base64-encoded strings
    images_base64 = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        buffer = io.BytesIO()
        img.save(buffer, format="png")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_base64.append("data:image/png;base64," + img_str)

    if descriptors is None:
        descriptors = {}
    # Create a data source for the plot
    source_dict = {
        "x": np.array([p[0] for p in data_points]),
        "y": np.array([p[1] for p in data_points]),
        "thumbnail": images_base64,
        **descriptors,
    }
    source = ColumnDataSource(data=source_dict)

    # information to be displayed on hover
    descriptors_tooltips = [
        (f"{key}", f"@{key}")
        for (key, value) in source_dict.items()
        if key not in ["x", "y", "thumbnail"]
    ]
    tooltips = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ] + descriptors_tooltips

    # Create the plot
    plot = figure(
        title=title,
        x_axis_label="Dimension 1",
        y_axis_label="Dimension 2",
        tools="hover, pan, wheel_zoom, zoom_in, zoom_out, box_zoom, reset, save",
        tooltips=tooltips,
        width=1000,
        height=1000,
    )

    if not descriptors == {}:
        color_descriptor = descriptors[color_by]
        legend_categories = [str(i) for i in sorted(np.unique(color_descriptor))]
        color_map = factor_cmap(
            color_by,
            palette=Category20[len(legend_categories)],
            factors=legend_categories,
        )

        plot.circle(
            source=source,
            x="x",
            y="y",
            size=10,
            color=color_map,
            line_color=None,
            fill_alpha=0.7,
            legend_group=color_by,
        )
    else:
        plot.circle(
            source=source,
            x="x",
            y="y",
            size=10,
            line_color=None,
            fill_alpha=0.7,
        )

    plot.legend.title = legend_title
    # if you want to reorder the legend items
    # plot.legend[0].items

    # Add the hover tool with the thumbnail image as the tooltip
    plot.add_tools(
        HoverTool(
            tooltips="""
                <div>
                    <img
                        src="@thumbnail" height="100" width="100"
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
            """
        )
    )

    # configure the output file
    output_file(save_dir / filename)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save(plot)

    if show_plot:
        show(plot)


# =======================================
# Function: plot_strip_and_swarm
# ========================================


def plot_strip_and_swarm(
    data: pd.DataFrame,
    backend: str = "seaborn",
    **kwargs,
) -> None:
    """
    Plots a strip and swarm plot of the data.

    Parameters:
        data:
        backend:
        **kwargs:

    Optional kwargs:
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        xtick_labels (list, optional): Custom labels for the x-axis ticks.
        plot_all_xtick_labels (bool, optional): Whether to plot all the xtick labels.
        ytick_labels (list, optional): Custom labels for the y-axis ticks.
        save_dir (str or Path, optional): Directory to save the plot.
        save_filename (str, optional): File name to save the plot.

    Returns:

    """
    # TODO: Incomplete function
    # Set the default save directory if not provided
    save_dir = kwargs.get("save_dir", Path("plots"))
    save_dir = Path(save_dir)  # Ensure save_dir is a Path object
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the default save file name if not provided
    save_filename = kwargs.get("save_filename", "binary_matrix_heatmap.png")

    # Get xtick labels from kwargs if provided
    xtick_labels = kwargs.get("xtick_labels")
    ytick_labels = kwargs.get("ytick_labels")

    if backend.lower() == "seaborn":
        fig, ax = plt.subplots(figsize=(fig_width, fig_length))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="viridis",
            annot=True,
            xticklabels=xtick_labels,
            yticklabels=kwargs.get("ytick_labels"),
        )

        # Set title and axis labels from kwargs
        ax.set_title(kwargs.get("title", "Stripplot with Swarmplot"))
        ax.set_xlabel(kwargs.get("xlabel", "Columns"))
        ax.set_ylabel(kwargs.get("ylabel", "Rows"))

        plt.tight_layout()

        if save_filename:
            # Create the plots directory if it doesn't exist
            fig.savefig(save_dir / save_filename)

        plt.show()


# =======================================
# Function: plot_variance_explained
# ========================================
def plot_variance_explained(data_points, plot_name=None, save_plot=False):
    """
    Plots the variance explained per principal component
    and a cumulative variance explained per principal component
    on a line plot.

    Args:
        data_points (array): high-dimensional points (n_samples, n_features)
        plot_name (string): name to save the plots
        save_plot (bool): whether to save the plot

    Returns:
        variance_df (pd.DataFrame): DataFrame containing the variance explained per PC

    Example:
        video_data_loader = VideoDataLoader(cfg, device)
        plot_variance_explained(video_data_loader.frame_embeddings)
    """

    pca = PCA()
    pca.fit(data_points)
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)

    variance_dict = {
        "PC": np.arange(pca.n_components_) + 1,
        "variance_explained": explained_var,
        "cumulative_variance": cumulative_var,
    }
    variance_df = pd.DataFrame(variance_dict)

    n_components_50 = np.argmax(cumulative_var >= 0.5)
    n_components_90 = np.argmax(cumulative_var >= 0.90)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    sns.lineplot(x="PC", y="variance_explained", data=variance_df, ax=axes[0, 0])
    sns.lineplot(
        x="PC",
        y="cumulative_variance",
        data=variance_df,
        ax=axes[1, 0],
    )
    sns.lineplot(
        x="PC",
        y="variance_explained",
        data=variance_df,
        ax=axes[0, 1],
    )
    sns.lineplot(
        x="PC",
        y="cumulative_variance",
        data=variance_df,
        ax=axes[1, 1],
    )
    axes[0, 1].set(xscale="log", yscale="log")
    axes[1, 1].set(xscale="log", yscale="log")

    for i, ax in enumerate(fig.axes):
        ax.axvline(
            x=n_components_50,
            color="purple",
            linestyle="--",
            label="50% variance explained",
        )
        ax.axvline(
            x=n_components_90,
            color="green",
            linestyle="--",
            label="90% variance explained",
        )
        ax.legend()
        ax.set_xlabel("Principal Component", fontsize=16)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

    axes[0, 0].set_ylabel("Variance Explained", fontsize=16)
    axes[1, 0].set_ylabel("Cumulative Variance Explained", fontsize=16)
    axes[0, 1].set_ylabel("Variance Explained", fontsize=16)
    axes[1, 1].set_ylabel("Cumulative Variance Explained", fontsize=16)

    plt.suptitle(f"PCA Explained Variance \n{plot_name}", fontsize=20)
    plt.tight_layout()

    if save_plot:
        save_dir = "../plots/PCA_variance_explained"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(
            f"{save_dir}/{plot_name}_PC_variance_explained.png",
            dpi=300,
            bbox_inches="tight",
        )
    else:
        # plt.show()
        pass

    # print(f"Number of components for 99% variance: {pca.n_components_}")
    # print(f"Variance explained by 3 PCs: {sum(pca.explained_variance_ratio_[0:3])}")
    # print(f"Variance explained by 50 PCs: {sum(pca.explained_variance_ratio_[0:50])}")

    return variance_df


def plot_ecdf(data):
    plt.figure()
    if data.ndim == 1:
        _ = sns.ecdfplot(data)
        plt.show()
    elif data.ndim == 2:
        for i in range(data.shape[1]):
            _ = sns.ecdfplot(data[:, i], label=f"feature_{i+1}")
        plt.show()


# noinspection PyUnresolvedReferences
def elbow_curve(data, max_k=10, plot=True, seed=42):
    inertia = []
    for k in np.arange(max_k):
        kmeans = KMeans(n_clusters=k, random_state=seed).fit(data)
        inertia.append(kmeans.inertia_)
    if plot:
        plt.figure()
        plt.plot(np.arange(max_k), inertia, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.show()
    return inertia


# ========================================
# Function: plot_tsne_projections]
# ========================================
def plot_tsne_projections(
    projections: np.ndarray,
    backend: str = "matplotlib",
    **kwargs,
) -> None:
    """t-SNE 2-D Projections Plot

    Parameters:
        projections (np.ndarray): t-SNE projection matrix
        backend (str): which plotting package to use

    Optional kwargs:
        title (str): title of the plot
        suptitle (str): suptitle of the plot
        labels (list[str]): labels for each data point
        categories (list[str]): categories for each data point
        legend_title (str): title for the legend

    Returns:
        None
    """

    title = kwargs.get("title", "t-SNE Plot")
    suptitle = kwargs.get("suptitle", None)
    labels = kwargs.get("labels", [])
    categories = kwargs.get("categories", None)
    legend_title = kwargs.get("legend_title", "Legend")

    # check if input is 2-D array.
    if projections.shape[1] != 2:
        raise ValueError(f"Expected 2-D array. Got {projections.shape[1]} dimensions.")

    if backend == "matplotlib":
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(
            projections[:, 0],
            projections[:, 1],
            s=5,  # marker size
            c=categories,  # color
            label=labels,
            cmap="viridis",
            alpha=0.6,  # transparency
        )
        # plotting data point labels
        if labels:
            for i, (x, y) in enumerate(projections):
                plt.text(
                    x + 0.3, y + 0.3, labels[i], fontsize=9, ha="right", va="bottom"
                )  # Adjust text position and size as needed
        ax.set_title(title if title else "")
        fig.suptitle(suptitle if suptitle else "")
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.legend(title=legend_title, loc="best", bbox_to_anchor=(1, 0.5))
        plt.title(title, fontsize=16)
        plt.tight_layout()

    elif backend == "seaborn":
        data = pd.DataFrame(
            {"tsne_1": projections[:, 0], "tsne_2": projections[:, 1], "label": labels}
        )
        fig, ax = plt.subplots(figsize=(10, 6))
        # Create a scatter plot colored by category

        sns.scatterplot(
            data=data,
            x="tsne_1",
            y="tsne_2",
            hue="label",
            ax=ax,
            palette=sns.color_palette(
                "bright", len(np.unique(labels if labels else []))
            ),
            legend="full",
        )
        # legend1 = ax.legend(
        #     *scatter.legend_elements(),
        #     loc="upper left",
        #     bbox_to_anchor=(1.05, 1.0),
        # )
        # ax.add_artist(legend1)
        # plt.legend()
        ax.set_xlabel("t-SNE Dimension 1")
        ax.set_ylabel("t-SNE Dimension 2")
        ax.set_title(title if title else "")
        fig.suptitle(suptitle if suptitle else "")
        plt.subplots_adjust(right=0.7)
        # plt.tight_layout(rect=[0, 0, 0.75, 1])

    else:
        raise ValueError(
            f"Invalid backend: {backend}. Choose from 'seaborn' or 'matplotlib'."
        )

    return fig


# ========================================
# Function: plot_design_matrix
# ========================================
def plot_design_matrix(design_matrix):
    """
    This plots the design matrix that goes into a linear decoding/encoding model.

    Returns:

    """

    # TODO: write this function

    # also create a visualization of the entire design matrix
    plt.figure()
    plt.imshow(design_matrix, aspect="auto")
    plt.xlabel("Feature")
    plt.ylabel("Time (Frames)")
    plt.show()


# ========================================
# Function: create_rose_plot
# ========================================
def create_polar_plot_tuning_curve(
    df, neuron_id, custom_order=None, metric_to_plot="importance_avg_abs", **kwargs
) -> None:
    """Create a polar plot for a single neuron.

    Parameters:
        df (DataFrame): The DataFrame containing neuron data.
        neuron_id (int or str): The ID of the neuron to plot.
        custom_order (list): A custom order for the labels (optional).
        metric_to_plot (str): The metric to plot on the rose plot.

    Returns:
        None
    """
    # Set the default save directory if not provided
    save_dir = kwargs.get("save_dir", Path("plots"))
    save_dir = Path(save_dir)  # Ensure save_dir is a Path object
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the default save file name if not provided
    save_filename = kwargs.get(
        "save_filename", f"polar_plot_{neuron_id}_{metric_to_plot}.png"
    )

    # Subset the DataFrame for the given neuron
    neuron_df = df[df["cell_id"] == neuron_id]
    print(f"There are {len(neuron_df)} values for cell [{neuron_id}].")

    # Prepare the data for plotting
    labels = neuron_df["label"].unique()
    if custom_order:
        labels = [label for label in custom_order if label in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    if metric_to_plot == "importance_avg_abs":
        values = [
            neuron_df[neuron_df["label"] == label]["importance_avg_abs"].values[0]
            for label in labels
        ]
    elif metric_to_plot == "importance_avg":
        values = [
            neuron_df[neuron_df["label"] == label]["importance_avg"].values[0]
            for label in labels
        ]
    elif metric_to_plot == "importance_rank":
        highest_rank = df["importance_rank"].max()
        values = [
            highest_rank
            - neuron_df[neuron_df["label"] == label]["importance_rank"].values[0]
            for label in labels
        ]
    elif metric_to_plot == "importance_ratio":
        values = [
            neuron_df[neuron_df["label"] == label]["importance_ratio"].values[0]
            for label in labels
        ]
    else:
        raise ValueError(f"Invalid metric to plot: {metric_to_plot}")
    print(
        f"Plotting the polar plot for neuron [{neuron_id}] with metric [{metric_to_plot}]."
    )

    # Repeat the first value to close the circle
    angles += angles[:1]
    values += values[:1]

    # Create the rose plot
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    ax.plot(angles, values, "o-", linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    # Add labels to the plot
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=16, fontweight="regular")
    ax.tick_params(pad=20)
    # Add title
    ax.set_title(
        f"Polar Plot for Neuron {neuron_id}\nMetric: {metric_to_plot}", size=20, y=1.1
    )
    plt.tight_layout()

    # Save the plot to the specified directory with the specified file name
    plot_path = save_dir / save_filename
    fig.savefig(plot_path)

    # Show the plot
    plt.show()


# ========================================
# Function: plot_heatmap_pairwise_distance
# ========================================
def plot_heatmap_pairwise_distance(
    pw_dist_mat: np.ndarray,
    backend: str = "seaborn",
    show_plot: bool = True,
    **kwargs,
) -> None:
    """Heatmap for Pairwise Distances Matrix

    Parameters:
        pw_dist_mat (np.ndarray): Pairwise distance (square, symmetric) matrix.
        backend (str): Plotting package to use.
            Options: "seaborn" or "matplotlib"
        show_plot (bool): Whether to show the plot.

    Optional kwargs:
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        xtick_labels (list, optional): Custom labels for the x-axis ticks.
        ytick_labels (list, optional): Custom labels for the y-axis ticks.
        save_dir (str or Path, optional): Directory to save the plot.
        save_filename (str, optional): File name to save the plot.

    Returns:
        None
    """
    # Check if input is a square matrix
    if pw_dist_mat.shape[0] != pw_dist_mat.shape[1]:
        raise ValueError(f"Expected a square matrix. Got {pw_dist_mat.shape}.")

    # Set the default save directory if not provided
    save_dir = kwargs.get("save_dir", Path("plots"))
    save_dir = Path(save_dir)  # Ensure save_dir is a Path object
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the default save file name if not provided
    save_filename = kwargs.get("save_filename", "pairwise_distance_heatmap.png")

    # Get xtick and ytick labels from kwargs if provided
    xtick_labels = kwargs.get("xtick_labels")
    ytick_labels = kwargs.get("ytick_labels", xtick_labels)

    if backend.lower() == "seaborn":
        pair_dist_df = pd.DataFrame(pw_dist_mat)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            pair_dist_df,
            cmap="YlGnBu",
            ax=ax,
            xticklabels=xtick_labels,
            yticklabels=ytick_labels,
        )
        ax.set_title(kwargs.get("title", "Pairwise Distance Heatmap"))
        ax.set_xlabel(kwargs.get("xlabel", "Data Point"))
        ax.set_ylabel(kwargs.get("ylabel", "Data Point"))

        plt.tight_layout()

        if save_filename:
            # Create the plots directory if it doesn't exist
            fig.savefig(save_dir / save_filename)

        # Show the plot
        if show_plot:
            plt.show()

    else:
        raise ValueError(
            f"Invalid backend: {backend}. Choose from 'seaborn' or 'matplotlib'."
        )


# ========================================
# Function: plot_heatmap_binary_matrix
# ========================================
def plot_heatmap_binary_matrix(
    matrix: np.ndarray, backend: str = "seaborn", **kwargs
) -> None:
    """Plots a heatmap of a binary-valued matrix.

    Parameters:
        matrix (np.ndarray): The binary-valued matrix to be plotted.
        backend (str): Plotting package to use. Options: "seaborn" or "matplotlib".

    Optional kwargs:
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        xtick_labels (list, optional): Custom labels for the x-axis ticks.
        plot_all_xtick_labels (bool, optional): Whether to plot all the xtick labels.
        ytick_labels (list, optional): Custom labels for the y-axis ticks.
        save_dir (str or Path, optional): Directory to save the plot.
        save_filename (str, optional): File name to save the plot.

    Returns:
        None
    """
    # Set the default save directory if not provided
    save_dir = kwargs.get("save_dir", Path("plots"))
    save_dir = Path(save_dir)  # Ensure save_dir is a Path object
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the default save file name if not provided
    save_filename = kwargs.get("save_filename", "binary_matrix_heatmap.png")

    # Get xtick and ytick labels from kwargs if provided
    xtick_labels = kwargs.get("xtick_labels")
    ytick_labels = kwargs.get("ytick_labels")

    # Calculate the number of x labels either from xtick_labels or a default number
    if xtick_labels is not None:
        num_xlabels = len(xtick_labels)
    else:
        xtick_labels = np.arange(matrix.shape[1])
        num_xlabels = matrix.shape[1]

    if ytick_labels is not None:
        num_ylabels = len(ytick_labels)
    else:
        ytick_labels = np.arange(matrix.shape[0])
        num_ylabels = matrix.shape[0]

    max_width = 12.0  # Maximum allowable width in inches
    max_length = 10.0  # Maximum allowable length in inches

    # Dynamically calculate the figure size with a ceiling
    xlabels_width = (
        num_xlabels * 0.5 + 2.0
    )  # 0.5 inch per label, 2.0 inch for title and y-axis
    fig_width = min(
        max_width, xlabels_width
    )  # Width based on the number of columns, with a max of 12 inches
    ylabels_length = (
        num_ylabels * 0.5 + 2.0
    )  # 0.5 inch per label, 2.0 inch for title and x-axis
    fig_length = min(
        ylabels_length, max_length
    )  # Height based on the number of rows, with a max of 10 inches

    if backend.lower() == "seaborn":
        colors = ["lightblue", "darkblue"]
        cmap = ListedColormap(colors)
        fig, ax = plt.subplots(figsize=(fig_width, fig_length))

        heatmap = sns.heatmap(
            matrix,
            ax=ax,
            # xticklabels=xtick_labels,
            yticklabels=ytick_labels,
            cmap=cmap,
            cbar=False,
        )
        if kwargs.get("plot_all_xtick_labels", False):
            ax.set_xticklabels(xtick_labels)
        # else:
        #     ax.xaxis.set_major_locator(plt.MaxNLocator())

        # Set title and axis labels from kwargs
        ax.set_title(kwargs.get("title", "Binary Matrix Heatmap"))
        ax.set_xlabel(kwargs.get("xlabel", "Columns"))
        ax.set_ylabel(kwargs.get("ylabel", "Rows"))

        # # Extract the colormap from the heatmap
        # cmap = heatmap.collections[0].cmap
        # # Normalize the values for the colormap
        # norm = plt.Normalize(vmin=0, vmax=1)
        # color_0 = cmap(norm(0))
        # color_1 = cmap(norm(1))

        # Create custom legend
        # legend_elements = [
        #     Patch(facecolor=color_0, edgecolor="black", label="absent"),
        #     Patch(facecolor=color_1, edgecolor="black", label="present"),
        # ]

        legend_elements = [
            Patch(facecolor=colors[0], edgecolor="black", label="absent"),
            Patch(facecolor=colors[1], edgecolor="black", label="present"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", title="Legend")

        plt.tight_layout()

        if save_filename:
            # Create the plots directory if it doesn't exist
            fig.savefig(save_dir / save_filename)

        # Show the plot
        plt.show()

    else:
        raise ValueError(
            f"Invalid backend: {backend}. Choose from 'seaborn' or 'matplotlib'."
        )


# =======================================
# Function: plot_heatmap_matrix
# ========================================
def plot_heatmap_matrix(
    matrix: np.ndarray,
    backend: str = "seaborn",
    **kwargs,
) -> None:
    """
    Plots a heatmap of a matrix of values.

    Parameters:
        matrix (np.ndarray): The matrix to be plotted.
        backend (str): Plotting package to use. Options: "seaborn" or "matplotlib".

    Optional kwargs:
        title (str, optional): Title of the plot.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        xtick_labels (list, optional): Custom labels for the x-axis ticks.
        ytick_labels (list, optional): Custom labels for the y-axis ticks.
        save_dir (str or Path, optional): Directory to save the plot.
        save_filename (str, optional): File name to save the plot.

    Returns:
        None
    """
    # Set the default save directory if not provided
    save_dir = kwargs.get("save_dir", Path("plots"))
    save_dir = Path(save_dir)  # Ensure save_dir is a Path object
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set the default save file name if not provided
    save_filename = kwargs.get("save_filename", "matrix_heatmap.png")

    # Get xtick labels from kwargs if provided
    xtick_labels = kwargs.get("xtick_labels")

    # Calculate the number of x labels either from xtick_labels or a default number
    if xtick_labels is not None:
        num_xlabels = len(xtick_labels)
    else:
        num_xlabels = matrix.shape[1]

    # Determine the figure width based on the number of x labels
    fig_width = max(12.0, num_xlabels * 0.75)  # Min width of 12, and 0.75 per label
    fig_length = (
        matrix.shape[0] * 1.0 + 2.0
    )  # 1.0 per row, 2.0 for title and x/y labels

    if backend.lower() == "seaborn":
        try:
            fig, ax = plt.subplots(figsize=(fig_width, fig_length))
            sns.heatmap(
                matrix,
                ax=ax,
                cmap="viridis",
                annot=True,
                xticklabels=xtick_labels,
                yticklabels=kwargs.get("ytick_labels"),
            )

            # Set title and axis labels from kwargs
            ax.set_title(kwargs.get("title", "Binary Matrix Heatmap"))
            ax.set_xlabel(kwargs.get("xlabel", "Columns"))
            ax.set_ylabel(kwargs.get("ylabel", "Rows"))

            plt.tight_layout()

            if save_filename:
                # Create the plots directory if it doesn't exist
                fig.savefig(save_dir / save_filename)

            plt.show()
        except ValueError as e:
            print(e)

    else:
        raise ValueError(f"Invalid backend: {backend}. Choose from 'seaborn'.")
