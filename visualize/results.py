import numpy as np
import pandas as pd
from plotly import express as px


def plot_results(y_true, y_pred, output_file_path=None):
    """
    Plots the predictions of a model against the true values.
    It's useful to visualize the predictions of a model in a binary classification problem.


    :param y_true: The true values of the target variable.
    :type y_true: Numpy array of size n_samples
    :param y_pred: _description_
    :type y_pred: Numpy array of size n_samples
    :param output_file_path: The path to save the plot.
    :type output_file_path: Path
    """
    y_proba_frame = pd.DataFrame(
        {
            "Sample Integer Index": np.arange(0, len(y_true)),
            "Positive Class Probability": y_pred,
            "Class": y_true.astype(str),
        },
    )

    fig = px.scatter(
        y_proba_frame,
        x="Positive Class Probability",
        y="Sample Integer Index",
        color="Class",
        color_discrete_sequence=["#010D36", "#FF2079"],
        category_orders={"Class": ("0", "1")},
        symbol="Class",
        symbol_sequence=["diamond", "circle"],
        opacity=0.6,
        height=540,
        width=840,
        title="Training Dataset - Out of Fold Predictions",
    )
    fig.update_layout(
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            xanchor="right",
            y=1.05,
            x=1,
            title="Class",
            itemsizing="constant",
        ),
        xaxis_range=[-0.02, 1.02],
    )
    fig.update_traces(marker_size=6)
    if not output_file_path:
        return fig
    fig.write_image(str(output_file_path))


def get_plot_results(y_true, y_pred):

    y_proba_frame = pd.DataFrame(
        {
            "Sample Integer Index": np.arange(0, len(y_true)),
            "Positive Class Probability": y_pred,
            "Class": y_true.astype(str),
        },
    )

    fig = px.scatter(
        y_proba_frame,
        x="Positive Class Probability",
        y="Sample Integer Index",
        symbol="Class",
        symbol_sequence=["diamond", "circle"],
        color="Class",
        color_discrete_sequence=["#010D36", "#FF2079"],
        category_orders={"Class": ("0", "1")},
        opacity=0.6,
        height=540,
        width=840,
        title="Training Dataset - Out of Fold Predictions",
    )

    return fig
