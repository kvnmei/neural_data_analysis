from scipy import stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def t_test(data: np.ndarray, labels, alpha=0.05):
    """Perform a t-test on the data.

    Args:
        data:
        group_labels:
        alpha:

    Returns:

    """
    assert len(data) == len(labels), "Data and labels must be the same length."
    assert len(np.unique(labels)) == 2, "Number of unique group labels must be 2."
    group1, group2 = np.unique(labels)
    group1_data = data[labels == group1]
    group2_data = data[labels == group2]

    # Check assumptions
    normality_group1 = check_normality(group1_data)
    normality_group2 = check_normality(group2_data)
    homogeneity = check_homogeneity_of_variances(group1_data, group2_data)

    # Plot data distribution
    plot_data_distribution(group1_data, group2_data, group1, group2)
    print(f"Normality Group {group1}: {'Passed' if normality_group1 else 'Failed'}")
    print(f"Normality Group {group2}: {'Passed' if normality_group2 else 'Failed'}")
    print(f"Homogeneity of Variances: {'Passed' if homogeneity else 'Failed'}")

    t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
    print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

    if not (normality_group1 and normality_group2 and homogeneity):
        print("Assumptions not met for t-test. Consider using non-parametric tests.")

    return t_stat, p_value


def check_normality(data, alpha=0.05):
    stat, p_value = stats.shapiro(data)
    if p_value > alpha:
        return True
    else:
        return False


# Function to check homogeneity of variances using Levene's test
def check_homogeneity_of_variances(group1_data, group2_data, alpha=0.05):
    stat, p_value = stats.levene(group1_data, group2_data)
    if p_value > alpha:
        return True
    else:
        return False


# Function to plot data distribution
def plot_data_distribution(group1_data, group2_data, group1, group2):
    sns.histplot(
        group1_data,
        kde=True,  # computes kernel density estimate to smooth distribution
        label=f"Group {group1}",
        color="blue",
        stat="density",
        linewidth=0,
    )
    sns.histplot(
        group2_data,
        kde=True,
        label=f"Group {group2}",
        color="orange",
        stat="density",
        linewidth=0,
    )
    plt.legend()
    plt.xlabel("Data Values")
    plt.ylabel("Density")
    plt.title("Data Distribution")
    plt.show()


def anova():
    pass
