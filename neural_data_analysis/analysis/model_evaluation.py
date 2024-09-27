import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
import itertools
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
)

# from tabulate import tabulate
import logging
import warnings


def calculate_model_performance(
    df: pd.DataFrame,
    metrics=None,
    metrics_for_shuffled_data=None,
    model_eval_input_cols: dict = None,
    by_feature: bool = True,
    avg_across_variables: bool = True,
    shuffle_ground_truth: bool = False,
    columns_to_keep: list[str] = (
        "brain_area",
        "bin_center",
        "bin_size",
        "embedding",
    ),
    classes: list = None,
) -> pd.DataFrame:
    """
    Given a dataframe with "ground_truth" and "predictions" columns,
    calculate the model performance for each sample (row) or feature (column).
    Appends the metric scores to the input dataframe in-place.

    Parameters:
        df (pd.DataFrame): dataframe containing the ground truth and predictions
        metrics (list[str]): list of metrics to compute between predictions and ground truth
        metrics_for_shuffled_data (list[str]): list of metrics to compute for the shuffled data
        model_eval_input_cols (dict): name of the column containing the ground truth and the predictions
            - keys: what column names this function looks for
            - values: corresponding column names in the input
        by_feature: if True, evaluate performance for each feature/variable (column) separately
            if False, evaluate performance for each sample (row) separately
        avg_across_variables (bool): if True, average the metric across the features or samples
        shuffle_ground_truth (bool): if True, shuffle the ground truth labels to get a baseline score
        columns_to_keep (list[str]): list of the original dataframe columns to keep in the new dataframe
        classes (list): list of classes for classification problems

    Returns:
        averaged_df (pd.DataFrame): DataFrame containing the averaged data
    """
    if metrics_for_shuffled_data is None:
        metrics_for_shuffled_data = ["balanced_accuracy"]
    if metrics is None:
        metrics = ["r2", "corr"]
    logger = logging.getLogger(__name__)
    logger.info(f"Calculating model performance with the following parameters:")
    logger.info(f"Model performance metrics: [{metrics}]")
    logger.info(f"Calculate performance by feature or variable? [{by_feature}]")
    logger.info(
        f"Average performance across features or samples? [{avg_across_variables}]"
    )
    logger.info(
        f"Shuffle the ground truth labels to get a baseline score? [{shuffle_ground_truth}]"
    )
    logger.info(f"Columns to keep in the new dataframe: [{columns_to_keep}]")

    # calculate the model performance by computing a metric between the ground truth and the predictions
    # by default, will look for columns named "ground_truth" and "predictions"
    if model_eval_input_cols is None:
        model_eval_input_cols = {
            "ground_truth": "ground_truth",
            "predictions": "predictions",
            "ground_truth_shuffled": "ground_truth_shuffled",
        }
    for k, v in model_eval_input_cols.items():
        logger.info(f"Looking for {k} values in the dataframe column [{v}]")

    logger.info("Calculating model performance...")
    # this will modify the dataframe in place
    append_model_scores(
        df,
        metrics=metrics,
        by_feature=by_feature,
        gt_col=model_eval_input_cols["ground_truth"],
        pred_col=model_eval_input_cols["predictions"],
        classes=classes,
    )

    sanity_check = False
    if sanity_check:
        gt = df["ground_truth"][0][:, 0]
        pred = df["predictions"][0][:, 0]
        report_dict = classification_report(
            gt, pred, target_names=["0", "1"], output_dict=True
        )
        # Convert the dictionary to a list of lists for tabulate
        # Initialize lists for tabulate
        headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        rows = []

        # Append rows for each class
        for key, values in report_dict.items():
            if isinstance(values, dict):
                rows.append(
                    [key]
                    + [
                        values["precision"],
                        values["recall"],
                        values["f1-score"],
                        values["support"],
                    ]
                )
            else:
                rows.append([key, values, "", "", ""])
        print(f"Balanced accuracy: {balanced_accuracy_score(gt, pred)}")
        print("Classification Report:\n")
        # print(tabulate(rows, headers=headers, floatfmt=".2f"))

    # shuffle the ground_truth to get a baseline score
    if shuffle_ground_truth:
        # set seed to always get the same shuffled values
        np.random.seed(42)
        # permute the ground truth labels in the data column represented by model_eval_input_cols["ground_truth"]
        logger.info("Permuting the ground truth values...")
        df[model_eval_input_cols["ground_truth_shuffled"]] = df[
            model_eval_input_cols["ground_truth"]
        ].apply(np.random.permutation)
        # what the shuffled score will be labeled as in the dataframe
        shuffled_metric_col_names = {
            metric: f"{metric}_shuffled" for metric in metrics_for_shuffled_data
        }
        logger.info("Calculating model performance for shuffled ground truth...")
        append_model_scores(
            df,
            metrics=metrics_for_shuffled_data,
            by_feature=by_feature,
            gt_col=model_eval_input_cols["ground_truth_shuffled"],
            pred_col=model_eval_input_cols["predictions"],
            metric_col_names=shuffled_metric_col_names,
        )
        for metric in shuffled_metric_col_names.keys():
            metrics.append(f"{shuffled_metric_col_names[metric]}")

    # calculate the mean of the metric across the features or samples
    if avg_across_variables:
        logger.info("Averaging performance across features or samples...")
        for metric in metrics:
            if by_feature:
                df[f"{metric}_mean"] = np.mean(np.stack(df[metric]), axis=1)
            else:
                df[f"{metric}_mean"] = df[metric].apply(np.mean)
        metrics = [f"{metric}_mean" for metric in metrics]

    # ------------- average across the folds -------------
    # logger.info("Averaging performance across validation folds...")
    # averaged_df = average_across_iterations(
    #     df,
    #     iter_var="fold",
    #     target_var=metrics,
    #     columns_to_keep=columns_to_keep,
    # )

    return df


def append_model_scores(
    df: pd.DataFrame,
    metrics: list[str],
    by_feature: bool = True,
    gt_col: str = "ground_truth",
    pred_col: str = "predictions",
    metric_col_names: dict = None,
    classes: list = None,
) -> None:
    """
    Given a dataframe with "ground_truth" and "predictions" columns,
    calculate for each row of the dataframe the model performance.

    The ground truth and prediction should be shape (n_samples, n_features)
    If calculating by feature, the output will be shape (n_features) after calculating the metric across samples.
    If calculating by sample, the output will be shape (n_samples) after calculating the metric across features.
    Appends the metric scores to the input dataframe in place.

    Parameters:
        df: dataframe containing the ground truth and predictions
        metrics: list of metrics to compute between predictions and ground truth
        pred_col: name of the column containing the predictions
        gt_col: name of the column containing the ground truth
        by_feature: if True, evaluate performance for each feature separately
            if False, evaluate performance for each sample separately
        metric_col_names (dict): a dictionary that maps the metric name (key) to the column name in the dataframe (value)
        classes: list of classes for classification problems

    Returns:
        None, appends the metrics to the input df in place
    """
    # Create a tuple of metrics if only one metric is passed
    if isinstance(metrics, str):
        metrics = tuple(metrics)

    # Calculate the model performance for each row of the dataframe
    # Typically, each row is a cross-validation fold
    scores = []
    for i in range(len(df)):
        row_score = evaluate_model_performance(
            df[gt_col][i],
            df[pred_col][i],
            metric=metrics,
            by_feature=by_feature,
            classes=classes,
        )
        scores.append(row_score)

    # append the scores to the dataframe in place
    for metric in metrics:
        if metric_col_names is not None:
            df[metric_col_names[metric]] = [score[metric] for score in scores]
        else:
            df[metric] = [score[metric] for score in scores]


# noinspection GrazieInspection
def evaluate_model_performance(
    ground_truth: np.ndarray,
    predictions: np.ndarray,
    metric: list[str],
    by_feature=True,
    classes: list = None,
) -> dict[str, np.ndarray]:
    """
    Evaluate the performance of a model using a given metric.
    Takes the ground truth and predictions as input to evaluate the model performance.

    Parameters:
        ground_truth (np.ndarray): (n_samples, n_features)
        predictions (np.ndarray: (n_samples, n_features)
        metric (tuple[str]): metric to evaluate performance
        by_feature (bool): if True, evaluate performance for each feature separately
                        if False, evaluate performance for each sample separately
        classes (list): list of classes for classification problems

    Returns:
        scores (dict): list of scores for each feature or sample
    """
    # If the ground truth and predictions are 1d, reshape them to 2d (n_samples, n_features)
    ground_truth = reshape_into_2d(ground_truth)
    predictions = reshape_into_2d(predictions)

    # For each metric, calculate the score
    scores = {}
    for metric_name in metric:
        # for multilabel binary classification

        if metric_name in ["precision", "recall", "fscore", "support"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                (
                    scores["precision"],
                    scores["recall"],
                    scores["fscore"],
                    scores["support"],
                ) = precision_recall_fscore_support(
                    ground_truth, predictions, labels=classes
                )
            break
        else:
            metric_score = []
            if by_feature:
                # noinspection GrazieInspection
                for feature_idx in range(ground_truth.shape[1]):
                    feat_gt = ground_truth[:, feature_idx]
                    feat_pred = predictions[:, feature_idx]
                    score = evaluate_metric(feat_gt, feat_pred, metric_name, classes)
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
                    score = evaluate_metric(
                        sample_gt, sample_pred, metric_name, classes
                    )
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


def evaluate_metric(
    ground_truth: np.ndarray, predictions: np.ndarray, metric: str, classes: list = None
) -> float:
    """
    Evaluate the performance of a model using a given metric.
    Takes the ground truth and predictions as input to evaluate the model performance.

    Parameters:
        ground_truth (np.array): the ground truth values
        predictions (np.array): the predicted values
        metric (str): the metric to evaluate the model performance
        classes (list): list of classes for classification problems

    Returns:
        score (float): the score of the model performance
    """
    if (metric == "true_positives") or (metric == "tp"):
        # Calculate the number of true positives
        true_positives = np.sum((ground_truth == 1) & (predictions == 1))
        score = true_positives
    elif (metric == "all_gt_positives") or (metric == "all_positives"):
        # Calculate the number of all positives in the ground truth, assuming binary
        all_positives = np.sum(ground_truth == 1)
        score = all_positives
    elif metric == "recall_raw":
        # Boolean array where ground truth is positive
        is_positive = y_true == 1

        # Indices of positive cases
        positive_indices = np.where(is_positive)[0]

        # Predictions corresponding to positive ground truth cases
        predictions_for_positives = y_pred[positive_indices]

        # Boolean array indicating correct predictions
        correct_predictions = predictions_for_positives == 1

        # Number of correct predictions (True Positives)
        num_true_positives = np.sum(correct_predictions)

        # Total number of positives in ground truth
        num_positives_in_ground_truth = np.sum(is_positive)

        # Recall calculation
        recall = num_true_positives / num_positives_in_ground_truth

        score = recall
    elif (metric == "precision") or (metric == "prec"):
        score = precision_score(
            y_true=ground_truth, y_pred=predictions, labels=classes, average="weighted"
        )
    elif (metric == "recall") or (metric == "rec") or (metric == "sensitivity"):
        score = recall_score(
            y_true=ground_truth, y_pred=predictions, labels=classes, average="weighted"
        )
    elif (metric == "correlation") or (metric == "corr"):
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
    elif (metric == "balanced_accuracy") or (metric == "balanced_acc"):
        # UserWarning: y_pred contains classes not in y_true
        # UserWarning: A single label was found in 'y_true' and 'y_pred'. For the confusion matrix to have the correct shape, use the 'labels' parameter to pass all known labels.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            score = balanced_accuracy_score(
                y_true=ground_truth, y_pred=predictions, adjusted=False
            )
    elif metric == "cohen_kappa":
        # Calculate confusion matrix
        try:
            # tn, fp, fn, tp = confusion_matrix(
            #     ground_truth, predictions, labels=classes
            # ).ravel()
            # score = cohen_kappa_score(ground_truth, predictions)
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", category=UserWarning)
                warnings.simplefilter("always", category=RuntimeWarning)
                score = cohen_kappa_score(ground_truth, predictions)
                if caught_warnings:
                    print(
                        "Warning while calculating the confusion matrix for Cohen's Kappa score."
                    )
                    if len(set(ground_truth)) == 1 and len(set(predictions)) == 1:
                        print(
                            "Ground truth and predictions only contain one unique class."
                        )
                    else:
                        print("Other issue.")
                    print("Returning the raw accuracy instead of Cohen's Kappa...")
                    score = np.mean(ground_truth == predictions)

        except ValueError:
            print(
                "ValueError while calculating the confusion matrix for Cohen's Kappa score."
            )
            if len(set(ground_truth)) == 1 and len(set(predictions)) == 1:
                print("Ground truth and predictions only contain one unique class.")
            else:
                print("Other issue.")
            print("Returning the raw accuracy instead of Cohen's Kappa...")
            score = np.mean(ground_truth == predictions)

        # # ======== Calculate Cohen's Kappa manually ========
        """
        The manually calculated Cohen's Kappa is consistent with the sklearn implementation.
        """

        # # Calculate p0 (observed agreement)
        # p0 = (tn + tp) / (tn + fp + fn + tp)

        # # Calculate pe (expected agreement)
        # p_pred0 = (tn + fn) / (tn + fp + fn + tp)
        # p_pred1 = (tp + fp) / (tn + fp + fn + tp)
        # p_actual0 = (tn + fp) / (tn + fp + fn + tp)
        # p_actual1 = (tp + fn) / (tn + fp + fn + tp)
        # pe = (p_pred0 * p_actual0) + (p_pred1 * p_actual1)

        # # Calculate Cohen's Kappa
        # kappa = (p0 - pe) / (1 - pe)

        # print(f"Observed Agreement (p0): {p0:.2f}")
        # print(f"Expected Agreement (pe): {pe:.2f}")
        # print(f"Cohen's Kappa: {kappa:.2f}")

        # score = cohen_kappa_score(ground_truth, predictions)
        # print(f"Cohen's Kappa (sklearn): {score:.2f}")
    else:
        raise ValueError(f"Metric {metric} not supported.")
    return score


def residual_analysis(y_val, preds):
    # TODO: implement this function

    return None


def average_across_iterations(
    df: pd.DataFrame,
    iter_var: str,
    target_var: list[str],
    columns_to_keep: list[str] = (
        "brain_area",
        "bin_center",
        "bin_size",
        "embedding",
    ),
) -> pd.DataFrame:
    """
    Given a pandas DataFrame, average the target variable across all iterations.
    Assumes multiple iterations were run for the same set of parameters,
    and the same number of iterations were run for each set of parameters.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        iter_var (str): name of the column containing the iteration variable
        target_var (tuple[str]): name of the column(s) containing the target variable(s)
        columns_to_keep (tuple[str]): list of the original dataframe columns to keep in the new dataframe

    Returns:
        df_avg (pd.DataFrame): DataFrame containing the averaged data

    Example:
        A dataframe containing the results from a kfold cross-validation experiment.
        Average the correlation across all folds.
        df_avg = average_across_iterations(df, iter_var="fold", target_var="correlation")
    """

    #  dataframe conversion, for now just fixed by making it a string
    if isinstance(target_var, str):
        target_var = tuple(target_var)

    averaged_result_list_dict = []
    n_iter = len(np.unique(df[iter_var]))

    # logging.warning(
    #     "Averaging will not work if the iter_var is not repeated in consecutive order in the dataframe"
    # )

    for i in np.arange(0, len(df), n_iter):
        _temp = df.iloc[i : i + n_iter].reset_index(drop=True)
        averaged_result_dict = {}
        for col in columns_to_keep:
            averaged_result_dict[f"{col}"] = _temp[f"{col}"][0]

        for var in target_var:
            variable_avg = np.mean(_temp[var].to_numpy(), axis=0)
            variable_std = np.std(_temp[var].to_numpy(), axis=0)
            averaged_result_dict.update(
                {
                    f"{var}_avg": variable_avg,
                    f"{var}_std": variable_std,
                }
            )
        averaged_result_list_dict.append(averaged_result_dict)

    averaged_result_df = pd.DataFrame(averaged_result_list_dict)
    return averaged_result_df


def sum_across_iterations(
    df: pd.DataFrame,
    iter_var: str,
    target_var: list[str],
    columns_to_keep: list[str] = (
        "brain_area",
        "bin_center",
        "bin_size",
        "embedding",
    ),
) -> pd.DataFrame:
    #  dataframe conversion, for now just fixed by making it a string
    if isinstance(target_var, str):
        target_var = tuple(target_var)

    summed_result_list_dict = []
    n_iter = len(np.unique(df[iter_var]))

    # logging.warning(
    #     "Summing will not work if the iter_var is not repeated in consecutive order in the dataframe"
    # )

    for i in np.arange(0, len(df), n_iter):
        _temp = df.iloc[i : i + n_iter].reset_index(drop=True)
        summed_result_dict = {}
        for col in columns_to_keep:
            summed_result_dict[f"{col}"] = _temp[f"{col}"][0]

        for var in target_var:
            # Convert Series to DataFrame
            df = pd.DataFrame(_temp[var].tolist())

            # Sum each column
            column_sums = df.sum()

            # Convert sums to list if needed
            sums_list = column_sums.tolist()

            summed_result_dict.update(
                {
                    f"{var}_sum": sums_list,
                }
            )
            # variable_sum = np.sum(_temp[var].to_numpy(), axis=1)
            # summed_result_dict.update(
            #     {
            #         f"{var}_sum": variable_sum,
            #     }
            # )
        summed_result_list_dict.append(summed_result_dict)

    summed_result_df = pd.DataFrame(summed_result_list_dict)
    return summed_result_df


def reshape_into_2d(arr: np.ndarray) -> np.ndarray:
    """
    Turns a numpy ndarray of 1-d shape (n,) into (n, 1) or keeps 2-d shape (n, m) into (n, m).

    Args:
        arr (np.ndarray): input array

    Returns:
        arr (np.ndarray): reshaped array
    """
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    elif arr.ndim > 2:
        raise ValueError("Array must be 1d or 2d.")
    return arr
