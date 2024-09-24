"""

Plotting functions for model results.

Functions:

    model_performance_by_fold
    model_performance_by_brain_area
    model_performance_by_time
    plot_model_predictions
    plot_confusion_matrix
    plot_object_detection_result



"""

from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from neural_data_analysis.analysis.model_evaluation import evaluate_metric


# TODO: fix legend location


def model_performance_by_fold(
    df: pd.DataFrame,
) -> None:
    """
    Plots the model performance by fold.
    Model performance variable should be continuous and have multiple values.
    Fold variable should be categorical.

    Args:
        df (pd.DataFrame): data

    Returns:
        None
    """
    _results_to_plot = pd.melt(
        df,
        id_vars=["fold", "bin_center", "bin_size", "embedding"],
        value_vars=["r2_mean", "corr_mean"],
        var_name="score_type",
        value_name="score_value",
    )

    fig, ax = plt.subplots()
    sns.stripplot(
        data=_results_to_plot,
        x="fold",
        y="score_value",
        hue="score_type",
        legend="auto",
    )
    plt.legend()
    plt.show()


def model_performance_by_brain_area(
    df: pd.DataFrame,
    df_avg: pd.DataFrame = None,
    metrics: List[str] = ("r2_mean", "corr_mean"),
    backend="seaborn",  # "bokeh
) -> None:
    """
    Plots the model performance by brain area.
    Model performance variable should be continuous and have multiple values.
    Brain area variable should be categorical.

    Args:
        df (pd.DataFrame): data
        df_avg (pd.DataFrame): data averaged across iterations
        metrics (list[str]): model performance metrics to plot (should be in df)
        backend (str): what plotting package to use


    Returns:
        None
    """
    if isinstance(metrics, str):
        metrics = tuple(metrics)

    if backend == "seaborn":
        # PLOTS ALL METRICS ON SAME PLOT!!! (may not make sense if they use different scales
        plt.figure()
        sns.stripplot(
            data=df,
            x="brain_area",
            y="score_value",
            hue="score_type",
            legend="auto",
        )
        if df_avg is not None:
            sns.stripplot(
                data=df_avg,
                x="brain_area",
                y="score_value",
                hue="score_type",
                linewidth=2,
                # legend="auto",
                legend=False,
            )
        plt.legend()
        plt.show()


def model_performance_by_time(
    df: pd.DataFrame,
    df_avg: pd.DataFrame = None,
    metrics: List[str] = ("r2_mean", "corr_mean"),
    backend="seaborn",
) -> None:
    """

    Args:
        df: data
        df_avg: data averaged across iterations
        metrics: model performance metrics to plot (should be in df)
        backend: what plotting package to use

    Returns:

    """

    if backend == "seaborn":
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="bin_center",
            y="score_value",
            hue="score_type",
            ax=ax,
        )
        sns.lineplot(
            data=df,
            x="bin_center",
            y="score_value",
            hue="score_type",
            ax=ax,
            legend=False,
        )
        plt.legend()
        plt.show()


def plot_model_predictions(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    sample_index: np.ndarray = None,
    backend="matplotlib",
    block_size: int = 0,
    **kwargs,
) -> None:
    # TODO: finish this... include an option to plot the block_size

    """
    Ground truth vs predictions plots.
    Generally, this would be for a single experiment and single variable.

    Args:
        ground_truth (np.ndarray): (n_samples) ground truth values
        predictions (np.ndarray): (n_samples) predicted values
        sample_index (np.ndarray): (n_samples) indices for each sample
        backend (str): which plotting package to use
        block_size (int): size of sample blocks, optional

    Returns:
        None

    Examples:
        Plot the model predictions for one fold of a KFold CV experiment.
    """
    # create the x-axis labels
    if sample_index is None:
        x_range = np.arange(0, ground_truth.shape[0])
    else:
        x_range = sample_index

    _r2 = evaluate_metric(ground_truth, predictions, "r2")
    _corr = evaluate_metric(ground_truth, predictions, "corr")

    if backend == "matplotlib":
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 10))
        axes[0].scatter(x_range, ground_truth, marker=".", label="ground_truth")
        axes[0].scatter(
            x_range,
            predictions,
            marker=".",
            label="predictions",
        )

        if block_size > 0:
            block_dividers = np.arange(x_range[0], len(x_range), block_size)
            for div in block_dividers:
                axes[0].axvline(x=div, color="red", linestyle="--")

        axes[0].set_xlabel("index")
        axes[0].set_ylabel(f"value")
        axes[0].legend()

        # axes[0].set_title(f"R2: {_r2:.3f} | Corr: {_corr:.3f}")

        axes[1].scatter(ground_truth, predictions)
        axes[1].set_xlabel("ground truth")
        axes[1].set_ylabel("predictions")
        axes[1].set_title(f"R2: {_r2:.3f} | Corr: {_corr:.3f}")

        theta = np.polyfit(ground_truth, predictions, deg=1)
        best_fit = theta[0] * ground_truth + theta[1]
        axes[1].plot(ground_truth, best_fit, linestyle="--", color="red")

        plt.suptitle(f"Ground truth vs. Predictions")
        plt.tight_layout()
        plt.show()


def plot_confusion_matrix(
    confusion_mat: np.ndarray, labels: List[str] = None, backend: str = "sklearn"
) -> None:
    """

    Parameters:
        confusion_mat (np.ndarray): confusion matrix
        labels (list[str]): class labels
        backend (str): which plotting package to use

    Returns:
        None
    """
    if backend == "sklearn":
        _ = ConfusionMatrixDisplay(confusion_mat, display_labels=labels)
        _.plot()
        plt.tight_layout()
        plt.show()
    elif backend == "seaborn":
        plt.figure()
        sns.heatmap(confusion_mat, annot=True)
        plt.show()


def plot_object_detection_result(image, result, ax=None):
    fig, ax = plt.subplots()
    ax.imshow(image)
    # indicate the bounding box
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        ax.plot(
            [box[0], box[2], box[2], box[0], box[0]],
            [box[1], box[1], box[3], box[3], box[1]],
            label=label,
        )
    ax.legend()
    plt.show()


# TODO: this is for bokeh backend of model performance by brain area
# color_by = "score_type"
# legend_categories = sorted(np.unique(_results_to_plot["score_type"]))
# source = ColumnDataSource(_results_to_plot)
# color_map = factor_cmap(
#     color_by, palette=["blue", "orange"], factors=legend_categories
# )
# plot = figure(
#     title="Model performance metrics",
#     x_axis_label="Brain Area",
#     y_axis_label="Metric Score",
#     width=1000,
#     height=1000,
# )
# plot.circle(
#     source=source,
#     x="brain_area",
#     y="score_value",
#     color=color_map,
#     legend_group="score_type",
# )
#
# output_file("plots/test.html")
# show(plot)

# TODO: what is this?
# plot the metrics
# sns.scatterplot(
#     data=results.results,
#     x="bin_center",
#     y="r2_score_avg",
#     hue="embedding",
#     style="brain_area",
#     legend=False,
# )
