import base64
import io
import os
from pathlib import Path
from typing import List

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
            color_by, palette=Category20[len(legend_categories)], factors=legend_categories
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


def plot_variance_explained(data_points, plot_name=None, save_plot=False):
    """
    Plots the variance explain per principal component
    and a cumulative variance explained per principal component

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


def pairwise_distance_heatmap(pw_dist_mat: np.ndarray, backend: str = "seaborn") -> None:
    """

    Args:
        pw_dist_mat (np.ndarray): pairwise distance matrix
        backend (str): which plotting package to use

    Returns:
        None
    """
    # make sure it's a pairwise distance matrix
    if pw_dist_mat.shape[0] != pw_dist_mat.shape[1]:
        print("Not a pairwise distance matrix. Calculating pairwise distances using Euclidean distance...")
        pw_dist_mat = pairwise_distances(pw_dist_mat, metric="euclidean")

    if backend == "seaborn":
        pair_dist_df = pd.DataFrame(pw_dist_mat)
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(pair_dist_df, cmap="YlGnBu", ax=ax)
        plt.show()

    save_dir = Path("../plots/pairwise_distance_matrices")
    save_dir.mkdir(exist_ok=True, parents=True)


def plot_tsne_projection(tsne_mat: np.ndarray, labels: np.ndarray, backend: str = "seaborn") -> None:
    """

    Args:
        tsne_mat (np.ndarray): t-SNE projection matrix
        labels (np.ndarray): labels for each data point
        backend (str): which plotting package to use

    Returns:
        None
    """

    df_plot = pd.DataFrame({
        "tsne_1": tsne_mat[:, 0],
        "tsne_2": tsne_mat[:, 1],
        "label": labels
    })

    if backend == "seaborn":
        fig, ax = plt.subplots()
        sns.scatterplot(
            df_plot,
            x="tsne_1",
            y="tsne_2",
            hue="label",
            ax=ax
        )
        # legend1 = ax.legend(
        #     *scatter.legend_elements(),
        #     loc="upper left",
        #     bbox_to_anchor=(1.05, 1.0),
        # )
        # ax.add_artist(legend1)
        # plt.legend()
        plt.xlabel("t-SNE Dimension 1")
        plt.ylabel("t-SNE Dimension 2")
        plt.title("t-SNE Plot Colored by Cluster Label")
        plt.subplots_adjust(right=0.7)
        # plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.show()

def plot_design_matrix(design_matrix):
    """
    This plots the design matrix that goes into a linear decoding/encoding model.

    Returns:

    """

    #TODO: write this function

    # also create a visualization of the entire design matrix
    plt.figure()
    plt.imshow(design_matrix, aspect="auto")
    plt.xlabel("Feature")
    plt.ylabel("Time (Frames)")
    plt.show()