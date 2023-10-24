import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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
    metrics: list[str] = ("r2_mean", "corr_mean"),
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
    metrics: list[str] = ("r2_mean", "corr_mean"),
    backend="seaborn",
) -> None:
    """

    Args:
        df: data

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
        fig, ax = plt.subplots()
        ax.scatter(x_range, ground_truth, marker=".", label="ground_truth")
        ax.scatter(
            x_range,
            predictions,
            marker=".",
            label="predictions",
        )

        if block_size > 0:
            block_dividers = np.arange(x_range[0], len(x_range), block_size)
            for div in block_dividers:
                ax.axvline(x=div, color="red", linestyle="--")

        plt.xlabel("index")
        plt.ylabel(f"value")
        plt.suptitle(f"Ground truth vs. Predictions")
        plt.title(f"R2: {_r2:.3f} | Corr: {_corr:.3f}")
        plt.legend()
        plt.tight_layout()
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
