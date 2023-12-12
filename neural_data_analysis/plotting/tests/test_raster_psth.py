import unittest
import os
import numpy as np
import pandas as pd
from neural_data_analysis.plotting import (
    plot_raster_psth,
    compute_psth,
    compute_psth_per_category,
    plot_gantt_bar_chart,
)


class TestComputePSTH(unittest.TestCase):
    def setUp(self):
        # Setup any repeated variables used in the tests
        num_events = 10
        self.spike_times = np.random.uniform(-1, 2, size=(num_events, 10))
        self.time_range = [-1, 2]
        self.bin_size = 0.1
        self.step_size = 0.05

    def test_compute_psth(self):
        # Test the basic functionality
        psth = compute_psth(
            self.spike_times, self.time_range, self.bin_size, self.step_size
        )
        # Check if psth is the right shape


class TestPlotRasterPSTH(unittest.TestCase):
    """
    Test the MLPModel class on random data.
    """

    def setUp(self):
        # Setup any repeated variables used in the tests
        num_events = 10
        rng = np.random.default_rng(seed=2023)
        spike_times = rng.uniform(-1, 2, size=(num_events, 10))  # Example spike times
        event_ids = np.arange(num_events)
        categories = np.random.randint(1, 4, size=num_events)
        palette = ["blue", "green", "red", "purple", "orange", "yellow"]
        unique_categories = np.unique(categories)
        color_map = {
            group: palette[i % len(palette)]
            for i, group in enumerate(unique_categories)
        }
        category_colors = [color_map[cat] for cat in categories]

        self.raster_df = pd.DataFrame(
            {
                "event_id": event_ids,
                "spike_times": list(spike_times),
                "category": categories,
                "color": category_colors,
            }
        )
        self.raster_df.sort_values(by=["category", "event_id"], inplace=True)
        # self.raster_df['new_event_id'] = np.arange(len(self.raster_df))[::-1]
        self.raster_df["event_id"] = np.arange(len(self.raster_df))
        time_window = [-1, 2]
        bin_size = 0.1
        step_size = 0.05
        bin_centers = (
            np.arange(time_window[0], time_window[1] + step_size - bin_size, step_size)
            + bin_size / 2
        )
        psth = compute_psth_per_category(
            self.raster_df, time_window, bin_size, step_size
        )
        # psth = np.random.rand(3, len(bin_centers))
        # psth_category_colors = [color_map[cat] for cat in np.arange(1, 4)]
        self.psth_df = pd.DataFrame(
            {
                "bin_centers": [bin_centers] * 3,
                "firing_rate": list(psth.values()),
                "category": unique_categories,
                "color": color_map.values(),
            }
        )
        self.plot_params = {
            "title": "Test Plot",
            "subtitle": "Test Subtitle",
            "xlabel": "Time",
            "ylabel": "Event",
            "save_path": "plots/test_plot_raster_psth",
            "figsize": (12, 8),
        }

    def test_plot_raster_psth_bokeh(self):
        # Test the basic functionality
        plot_raster_psth(
            self.raster_df, self.psth_df, self.plot_params, backend="bokeh"
        )
        # Check if file is created
        self.assertTrue(os.path.isfile(f"{self.plot_params['save_path']}.html"))

    def test_plot_raster_psth_seaborn(self):
        # Test the basic functionality
        plot_raster_psth(
            self.raster_df, self.psth_df, self.plot_params, backend="seaborn"
        )
        # Check if file is created
        self.assertTrue(os.path.isfile(f"{self.plot_params['save_path']}.png"))


class TestPlotGanttChart(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(seed=2023)

        # Setup any repeated variables used in the tests
        self.data = pd.DataFrame(
            {
                "frames": np.arange(10),
                "category": rng.integers(1, 4, size=10),
            }
        )
        self.plot_params = {
            "title": "Test Plot",
            "subtitle": "Test Subtitle",
            "xlabel": "Time",
            "ylabel": "Event",
            "save_path": "plots/test_gantt_bar_chart",
            "figsize": (12, 8),
            "colors": ["red", "green", "blue"],
        }

    def test_plot_gantt_bar_chart(self):
        # Test the basic functionality
        plot_gantt_bar_chart(
            df=self.data, plot_params=self.plot_params, backend="bokeh"
        )
        # Check if file is created
        self.assertTrue(os.path.isfile(f"{self.plot_params['save_path']}.html"))


if __name__ == "__main__":
    unittest.main()
