import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity

from neural_data_analysis.utils import average_across_iterations, reshape_into_2d


def process_results_multiple_regression(
    df: pd.DataFrame,
    metrics: list[str] = ("r2", "corr"),
    model_eval_input_cols: dict = None,
    by_feature: bool = True,
    avg_across_variables: bool = True,
) -> pd.DataFrame:
    """
    Given a dataframe with "ground_truth" and "predictions" columns,
    calculate the model performance for each sample or feature.
    Appends the metric scores to the input dataframe.

    Args:
        df (pd.DataFrame): dataframe containing the ground truth and predictions
        metrics (list): list of metrics to compute between predictions and ground truth
        model_eval_input_cols (dict): name of the column containing the ground truth and the predictions
        gt_col: name of the column containing the ground truth
        by_feature: if True, evaluate performance for each feature separately
            if False, evaluate performance for each sample separately

    Returns:
        None, appends the metrics to the input df
    """
    # calculate the model performance by computing a metric between the ground truth and the predictions
    # by default, will look for columns named "ground_truth" and "predictions"
    if model_eval_input_cols is None:
        model_eval_input_cols = {
            "ground_truth": "ground_truth",
            "predictions": "predictions",
        }

    # this will modify the dataframe in place
    append_model_scores(
        df,
        metrics=metrics,
        by_feature=by_feature,
        gt_col=model_eval_input_cols["ground_truth"],
        pred_col=model_eval_input_cols["predictions"],
    )

    # calculate the mean of the metric across the features or samples
    if avg_across_variables:
        for metric in metrics:
            if by_feature:
                df[f"{metric}_mean"] = np.mean(np.stack(df[metric]), axis=1)
            else:
                df[f"{metric}_mean"] = df[metric].apply(np.mean)
        metrics = [f"{metric}_mean" for metric in metrics]

    # average across the folds
    averaged_df = average_across_iterations(
        df,
        iter_var="fold",
        target_var=metrics,
    )

    return averaged_df


def append_model_scores(
    df: pd.DataFrame,
    metrics: list[str],
    by_feature: bool = True,
    gt_col: str = "ground_truth",
    pred_col: str = "predictions",
) -> None:
    """
    Given a dataframe with "ground_truth" and "predictions" columns,
    calculate the model performance for each sample or feature.
    Appends the metric scores to the input dataframe.

    Args:
        df: dataframe containing the ground truth and predictions
        metrics: list of metrics to compute between predictions and ground truth
        pred_col: name of the column containing the predictions
        gt_col: name of the column containing the ground truth
        by_feature: if True, evaluate performance for each feature separately
            if False, evaluate performance for each sample separately

    Returns:
        None, appends the metrics to the input df
    """
    if isinstance(metrics, str):
        metrics = tuple(metrics)

    scores = []
    for i in range(len(df)):
        row_score = evaluate_model_performance(
            df[gt_col][i], df[pred_col][i], metric=metrics, by_feature=by_feature
        )
        scores.append(row_score)
    for metric in metrics:
        df[metric] = [score[metric] for score in scores]


# noinspection GrazieInspection
def evaluate_model_performance(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    metric: list[str],
    by_feature=True,
) -> dict[str, np.ndarray]:
    """
    Args:
        ground_truth (np.ndarray): (n_samples, n_features)
        predictions (np.ndarray: (n_samples, n_features)
        metric (tuple[str]): metric to evaluate performance
        by_feature (bool): if True, evaluate performance for each feature separately
                        if False, evaluate performance for each sample separately

    Returns:
        scores (list): list of scores for each feature or sample
    """
    ground_truth = reshape_into_2d(ground_truth)
    predictions = reshape_into_2d(predictions)

    scores = {}
    for metric_name in metric:
        metric_score = []
        if by_feature:
            # noinspection GrazieInspection
            for feature_idx in range(ground_truth.shape[1]):
                feat_gt = ground_truth[:, feature_idx]
                feat_pred = predictions[:, feature_idx]
                score = evaluate_metric(feat_gt, feat_pred, metric_name)
                metric_score.append(score)

                # r2 = r2_score(ground_truth[:, feature_idx], predictions[:, feature_idx])
                # mse = mean_squared_error(ground_truth[:, feature_idx], predictions[:, feature_idx])
                # corr = pearsonr(ground_truth[:, feature_idx], predictions[:, feature_idx])[0]
                # # corr2 = np.corrcoef(ground_truth[:, feature_idx], predictions[:, feature_idx])[0,1]
                # cos_sim = cosine_similarity(
                #     ground_truth[:, feature_idx].reshape(1, -1), predictions[:, feature_idx].reshape(1, -1)
                # ).item()
                # cos_dist = distance.cosine(ground_truth[:, feature_idx], predictions[:, feature_idx])
                # euc_dist = distance.euclidean(ground_truth[:, feature_idx], predictions[:, feature_idx])
                # print(f"R2: {r2}, "
                #       f"MSE: {mse}, "
                #       f"corr: {corr}, "
                #       f"cosine similarity: {cos_sim}, "
                #       f"cosine distance: {cos_dist}, "
                #       f"euclidean distance: {euc_dist},")
                # scores.append([f"feature_{feature_idx}", r2, mse, corr, cos_sim, euc_dist])
        else:
            # noinspection GrazieInspection
            for i in range(ground_truth.shape[0]):
                sample_gt = ground_truth[i]
                sample_pred = predictions[i]
                score = evaluate_metric(sample_gt, sample_pred, metric_name)
                metric_score.append(score)

            # r2 = r2_score(ground_truth[i], predictions[i])
            # mse = mean_squared_error(ground_truth[i], predictions[i])
            # corr = pearsonr(ground_truth[i], predictions[i])[0]
            # # corr2 = np.corrcoef(ground_truth[i], predictions[i])[0,1]
            # cos_sim = cosine_similarity(
            #     ground_truth[i].reshape(1, -1), predictions[i].reshape(1, -1)
            # ).item()
            # cos_dist = distance.cosine(ground_truth[i], predictions[i])
            # euc_dist = distance.euclidean(ground_truth[i], predictions[i])
            # print(f"R2: {r2}, "
            #       f"MSE: {mse}, "
            #       f"corr: {corr}, "
            #       f"cosine similarity: {cos_sim}, "
            #       f"cosine distance: {cos_dist}, "
            #       f"euclidean distance: {euc_dist},")
            # scores.append([f"sample_{i}", r2, mse, corr, cos_sim, euc_dist])
        scores[metric_name] = np.array(metric_score)
    # scores_df = pd.DataFrame(
    #     scores,
    #     columns=[
    #         "index",
    #         "r2",
    #         "mse",
    #         "corr",
    #         "cos_sim",
    #         "euc_dist",
    #     ],
    # )

    return scores


def evaluate_metric(ground_truth: np.array, predictions: np.array, metric: str):
    """

    Args:
        ground_truth (np.array):
        predictions (np.array):
        metric (str):

    Returns:

    """
    if (metric == "correlation") or (metric == "corr"):
        score = pearsonr(ground_truth, predictions)[0]
    elif (metric == "r-squared") or (metric == "r2"):
        score = r2_score(ground_truth, predictions)
    elif (metric == "mean-squared-error") or (metric == "mse"):
        score = mean_squared_error(ground_truth, predictions)
    elif (metric == "cosine_similarity") or (metric == "cos_sim"):
        score = cosine_similarity(
            ground_truth.reshape(1, -1), predictions.reshape(1, -1)
        ).item()
    elif (metric == "accuracy") or (metric == "acc"):
        score = np.mean(ground_truth == predictions)
    else:
        raise ValueError(f"Metric {metric} not supported.")
    return score


def residual_analysis(y_val, preds):
    # TODO: implement this function

    return None