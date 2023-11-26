import unittest
import os
import numpy as np
import pandas as pd
from neural_data_analysis.plotting import plot_raster_psth2, plot_raster_psth


class TestPlotRasterPSTH(unittest.TestCase):
    """
    Test the MLPModel class on random data.
    """

    def setUp(self):
        # Setup any repeated variables used in the tests
        num_events = 10
        spike_times = np.random.uniform(-1, 2, size =(num_events, 10))  # Example spike times
        event_ids = np.arange(num_events)
        categories = np.random.randint(1, 4, size=num_events)
        palette = ['blue', 'green', 'red', 'purple', 'orange', 'yellow']
        unique_categories = np.unique(categories)
        color_map = {group: palette[i % len(palette)] for i, group in enumerate(unique_categories)}
        category_colors = [color_map[cat] for cat in categories]

        self.raster_df = pd.DataFrame({
            'event_id': event_ids,
            'spike_times': list(spike_times),
            'category': categories,
            'color': category_colors,
        })
        self.raster_df.sort_values(by=['category', 'event_id'], inplace=True)
        # self.raster_df['new_event_id'] = np.arange(len(self.raster_df))[::-1]
        self.raster_df['new_event_id'] = np.arange(len(self.raster_df))
        time_window = [-1, 2]
        bin_size = 0.1
        step_size = 0.05
        bin_centers = np.arange(time_window[0], time_window[1] + step_size - bin_size, step_size) + bin_size / 2
        psth = np.random.rand(3, len(bin_centers))
        psth_category_colors = [color_map[cat] for cat in np.arange(1, 4)]
        self.psth_df = pd.DataFrame({
            'bin_centers': [bin_centers]*3,
            'firing_rate': list(psth),
            'category': unique_categories,
            'color': psth_category_colors,
        })
        self.plot_params = {
            'title': 'Test Plot',
            'subtitle': 'Test Subtitle',
            'xlabel': 'Time',
            'ylabel': 'Event',
            'save_path': 'plots/test_plot_raster_psth.html',
            'figsize': (12, 8),
            'colors': ['red', 'green', 'blue']
        }

    def test_plot_raster_psth_seaborn(self):
        # Test the basic functionality
        plot_raster_psth2(self.spike_times, self.psth, self.bin_centers, self.time_window, self.bin_size, self.step_size, self.plot_params, backend="seaborn")
        # Check if file is created
        self.assertTrue(os.path.isfile(self.plot_params['save_path']))

    def test_plot_raster_psth_new(self):
        # Test the basic functionality
        plot_raster_psth(self.raster_df, self.psth_df, self.plot_params, backend="bokeh")
        # Check if file is created
        self.assertTrue(os.path.isfile(self.plot_params['save_path']))

if __name__ == "__main__":
    unittest.main()
