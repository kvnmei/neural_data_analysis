import base64
import io
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image


def plot_scatter_with_images(
    data_points: np.ndarray,
    images: List[Image],
    descriptors: Dict = None,
    color_by: str = None,
    title: str = "Scatter plot with image hover",
    legend_title: str = None,
    save_dir: Path = "../plots",
    filename: str = "scatterplot_with_hover.html",
    show_plot: bool = False,
):
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
    from bokeh.plotting import figure, show, save
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.io import output_file
    from bokeh.transform import factor_cmap
    from bokeh.palettes import Category10

    # Convert the PIL images to base64-encoded strings
    images_base64 = []
    for img in images:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.uint8(img))
        buffer = io.BytesIO()
        img.save(buffer, format="png")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        images_base64.append("data:image/png;base64," + img_str)

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
    TOOLTIPS = [
        ("index", "$index"),
        ("(x,y)", "($x, $y)"),
    ] + descriptors_tooltips

    # configure the output file
    output_file(f"{save_dir}/{filename}")

    # Create the plot
    plot = figure(
        title=title,
        x_axis_label="Dimension 1",
        y_axis_label="Dimension 2",
        tools="hover, pan, wheel_zoom, zoom_in, zoom_out, box_zoom, reset, save",
        tooltips=TOOLTIPS,
        width=1000,
        height=1000,
    )
    color_descriptor = descriptors[color_by]
    categories = [str(i) for i in sorted(np.unique(color_descriptor))]
    cmap = factor_cmap(color_by, palette=Category10[10], factors=categories)
    # Add the scatter plot markers
    plot.scatter(
        "x",
        "y",
        source=source,
        color=cmap,
        size=10,
        line_color=None,
        fill_alpha=0.7,
        legend_group=color_by,
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

    if show_plot:
        show(plot)
    else:
        save(plot)
