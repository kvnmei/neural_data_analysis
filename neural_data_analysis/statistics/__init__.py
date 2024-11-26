from .metrics import gini_coefficient
from .random_controls import (
    shift_binary_array,
    shuffle_binary_array_by_group,
    generate_random_clusters,
)
from .stats import (
    significance_stars,
    t_test,
    check_normality,
    check_homogeneity_of_variances,
    plot_data_distribution,
)

__version__ = "0.0.1"
__description__ = "A package for statistical tests."
__all__ = [f for f in dir() if not f.startswith("_")]
