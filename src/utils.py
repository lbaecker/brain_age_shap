"""Functions used in this repository.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from pathlib import Path

PROJECT_ROOT = Path.cwd()


def get_sample_sizes(dataset_train, dataset_validation, output_path=None, dir_name='sample_sizes'):
    """Function to calculate sample sizes.

    Parameters
    ----------
    dataset_train : DataFrame
        Pandas DataFrame containing all subjects in training datasets (healthy controls only).
    dataset_validation : DataFrame
        Pandas DataFrame containing all subjects in validation datasets (healthy controls and clinical subjects).
    output_path : Path, default=None
        Path to directory where sample sizes will be saved.
    dir_name : str, default='sample_sizes'
        Directory name
    """
    n_train_total = dataset_train.shape[0]
    n_train_scanner_df = dataset_train.scanner.value_counts().to_frame()
    n_validation_total = dataset_validation.shape[0]
    n_val_scanner_df = dataset_validation.groupby('scanner').Diagn.value_counts().to_frame()
    n_val_dataset_df = dataset_validation.groupby('dataset').Diagn.value_counts().to_frame()
    n_val_diagnosis_df = dataset_validation.Diagn.value_counts().to_frame()

    total_df = pd.DataFrame({
        'n_train_total': [n_train_total],
        'n_validation_total': [n_validation_total]
    })

    # Save sample sizes
    if output_path is not None:
        output_path.mkdir(exist_ok=True)
        dir_path = output_path / dir_name
        dir_path.mkdir(exist_ok=True)

        total_df.to_csv(dir_path / 'n_total.csv')
        n_train_scanner_df.to_csv(dir_path / 'n_train_by_scanner.csv')
        n_val_scanner_df.to_csv(dir_path / 'n_val_diagn_by_scanner.csv')
        n_val_dataset_df.to_csv(dir_path / 'n_val_diagn_by_dataset.csv')
        n_val_diagnosis_df.to_csv(dir_path / 'n_val_diagn.csv')

    return total_df, n_train_scanner_df, n_val_scanner_df, n_val_diagnosis_df


def histogram_plot(df_original, df_resampled, output_dir=PROJECT_ROOT, filename='histogram.png',
                   title_top="Original Data", title_bottom="Resampled Data",
                   x="Age", hue="Gender",
                   stacked=False, colors=None, colors2=None, include_two_legends=True):
    """Histogram plot comparing original and resampled sample across ages, split by gender.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    output_dir : Path
        Path to directory where output figure is saved.
    filename : str
        File name of output figure.
    title_top : str
        Title of top plot.
    title_bottom : str
        Title of bottom plot.
    x : str, default='Age'
        Name of variable on x axis of histogram. Must be a column in both DataFrames.
    hue : str, default='Gender'
        Name of variable to split the data by. Must be a column in both DataFrames.
    stacked : bool, default=False
        Whether multiple categories should be represented as stacked or overlapping.
    colors : list of str, optional
        List of colour names for hue. Dynamically generated if not provided.
    colors2 : list of str, optional
        List of colour names for the second DataFrame hue. Defaults to `colors` if not provided.
    include_two_legends : bool, default=True
        If df_original and df_resampled have the same legend, set to False to show one legend only.

    Returns
    -------
    fig : plt
        Histogram plot.
    """
    # Determine the hue categories in alphabetical order
    hue_order = sorted(df_original[hue].unique())

    # Dynamically generate a color palette if none is provided
    if colors is None:
        colors = sns.color_palette('husl', n_colors=len(hue_order))
    if colors2 is None:
        colors2 = colors

    # Define the x-axis range and create histogram bins
    xmin, xmax = pd.concat([df_original, df_resampled])[x].quantile(q=[0.0001, 0.9999])
    bins = np.arange(xmin, xmax + 1)

    # Create subplots
    fig, axis = plt.subplots(figsize=(20*0.7, 13*0.7), ncols=1, nrows=2, sharex=True, sharey=True)
    # fig, axis = plt.subplots(figsize=(16*0.7, 9*0.7), ncols=1, nrows=2, sharex=True)

    # Plot the original DataFrame
    if stacked is True:
        sns.histplot(data=df_original, bins=bins, x=x, hue=hue, hue_order=hue_order,
                     multiple='stack', palette=colors, ax=axis[0])
        if include_two_legends is False:
            sns.histplot(data=df_resampled, bins=bins, x=x, hue=hue, hue_order=hue_order,
                         multiple='stack', palette=colors2, ax=axis[1], legend=False)
        else:
            sns.histplot(data=df_resampled, bins=bins, x=x, hue=hue, hue_order=hue_order,
                         multiple='stack', palette=colors2, ax=axis[1])
    else:
        sns.histplot(data=df_original, bins=bins, x=x, hue=hue, hue_order=hue_order, palette=colors, ax=axis[0])
        if include_two_legends is False:
            sns.histplot(data=df_resampled, bins=bins, x=x, hue=hue,
                         hue_order=hue_order, palette=colors2, ax=axis[1], legend=False)
        else:
            sns.histplot(data=df_resampled, bins=bins, x=x, hue=hue, hue_order=hue_order, palette=colors2, ax=axis[1])

    # Add grid lines
    axis[0].grid(True)
    axis[1].grid(True)

    # Add subplot titles
    axis[0].set_title(title_top, y=0.98, loc="left")
    axis[1].set_title(title_bottom, y=0.98, loc="left")

    # Move legends
    sns.move_legend(axis[0], loc='upper left', bbox_to_anchor=(1.02, 1))
    if include_two_legends is True:
        sns.move_legend(axis[1], loc='upper left', bbox_to_anchor=(1.02, 1))

    # Enable minor ticks
    plt.minorticks_on()

    # Save the plot to the specified output directory
    plt.savefig(output_dir / filename, bbox_inches='tight')

    return fig


def histogram_plot_bottomlegend(
        df_original, df_resampled, output_dir=PROJECT_ROOT, filename='histogram.png',
        title_top="Original Data", title_bottom="Resampled Data",
        x="Age", hue="Diagn_labels",
        stacked=False, colors=None, colors2=None, include_two_legends=True):
    """Histogram plot comparing original and resampled sample across ages, split by hue categories.

    Note: The position of the legend differentiates this from the histogram_plot function.
    """
    # Determine the hue categories
    hue_order = sorted(df_original[hue].unique()) + sorted(df_resampled[hue].unique())
    hue_order = sorted(set(hue_order))  # Ensure no duplicates

    # Dynamically generate a color palette if none is provided
    if colors is None:
        colors = sns.color_palette('husl', n_colors=len(hue_order))
    if colors2 is None:
        colors2 = colors

    # Define the x-axis range and create histogram bins
    xmin, xmax = pd.concat([df_original, df_resampled])[x].quantile(q=[0.0001, 0.9999])
    bins = np.arange(xmin, xmax + 1)

    # Create subplots
    fig, axis = plt.subplots(figsize=(20 * 0.7, 13 * 0.7), ncols=1, nrows=2, sharex=True, sharey=True)

    # Plot the original DataFrame
    sns.histplot(
        data=df_original, bins=bins, x=x, hue=hue, hue_order=hue_order,
        multiple='stack' if stacked else 'layer', palette=colors, ax=axis[0]
    )
    sns.histplot(
        data=df_resampled, bins=bins, x=x, hue=hue, hue_order=hue_order,
        multiple='stack' if stacked else 'layer', palette=colors2, ax=axis[1]
    )

    # Add grid lines
    axis[0].grid(True)
    axis[1].grid(True)

    # Add subplot titles
    axis[0].set_title(title_top, y=0.98, loc="left")
    axis[1].set_title(title_bottom, y=0.98, loc="left")

    # Remove legends from individual axes if needed
    if not include_two_legends:
        axis[0].legend_.remove()
        axis[1].legend_.remove()

        # Create a single combined legend
        handles = [Patch(facecolor=colors2[i], label=name, edgecolor='black') for i, name in enumerate(hue_order)]
        legend = fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=True)

        # Add border around each patch in the legend
        for patch in legend.get_patches():
            patch.set_edgecolor('black')

        # Adjust the spacing to optimize layout
        fig.subplots_adjust(bottom=0.13)

    # Enable minor ticks
    plt.minorticks_on()

    # Save the plot to the specified output directory
    plt.savefig(output_dir / filename, bbox_inches='tight')

    return fig


def violin_plot_gender_age(df_original, df_resampled, x="Age", y="Diagn", hue="Gender", colors=["#13c1c7", "#f7ef69"]):
    """Violin plot comparing original and resampled sample across ages by diagnosis, split by age and gender.

    Parameters
    ----------
    df_original : DataFrame
        Original pandas DataFrame.
    df_resampled : DataFrame
        Resampled pandas DataFrame.
    x : str, default='Age'
        Name of variable on x axis of violin plot. Must be a column in both DataFrames.
    y : str, default='Diagn'
        Name of variable on y axis of violin plot. Must be a column in both DataFrames.
    hue : str, default='Gender'
        Name of variable to split the data by. Must be a column in both DataFrames.
    colors : list of str, default=["#6c6c6c", "#FF7070"]
        List of colour names for hue.

    Returns
    -------
    fig : plt
        Histogram plot.
    """
    fig, axis = plt.subplots(figsize=(16*0.7, 9*0.7), nrows=2, sharex=True)
    sns.violinplot(x=x, y=y, hue=hue, data=df_original, ax=axis[0], palette=cycle(colors), width=0.95,
                   inner="box", scale="width", bw=0.45, scale_hue=False, split=True, cut=0.1, orient="h", fontsize=7)
    sns.violinplot(x=x, y=y, hue=hue, data=df_resampled, ax=axis[1],
                   palette=cycle(colors), width=0.95, inner="box", scale="width", bw=0.45, scale_hue=False, split=True,
                   cut=0.1, orient="h", fontsize=7)
    axis[0].grid(True)
    axis[1].grid(True)
    axis[0].minorticks_on()
    axis[1].minorticks_on()
    axis[0].set_title("Original Data", y=0.98, loc="left")
    axis[1].set_title("Oversampled Data", y=0.98, loc="left")
    plt.tight_layout()
    return fig
