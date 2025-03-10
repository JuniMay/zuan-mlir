import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import argparse

plt.rcParams["font.family"] = "sans-serif"

def compute_product_size(size_str: str) -> int:
    """Compute the product of numerical values in a size string."""
    try:
        return np.prod([int(x) for x in size_str.split("/")])
    except ValueError:
        return 0  # Fallback for non-numeric values

def plot_and_annotate(ax, df, y_column, ylabel, title, use_log_scale, category_colors):
    """Plot data and annotate points with improved spacing and academic styling."""
    sns.lineplot(
        x="data_size",
        y=y_column,
        hue="category",
        data=df,
        marker="o",
        ax=ax,
        palette=category_colors,
    )

    if use_log_scale:
        ax.set_yscale("log")

    ax.set_title(
        f'Performance Comparison - {title}{" (Log Scale)" if use_log_scale else ""}',
        fontsize=12,
    )
    ax.set_ylabel(ylabel + (" (log scale)" if use_log_scale else ""), fontsize=10)
    ax.set_xlabel("Data Size", fontsize=10)
    ax.tick_params(axis="both", which="major", labelsize=8)

    sns.set_style("ticks")
    sns.despine(ax=ax)

    y_min, y_max = ax.get_ylim()
    margin = 0.1 * (y_max - y_min)
    ax.set_ylim(y_min - margin, y_max + margin)

    # Adjust annotation offsets for multiple categories
    for data_size, group in df.groupby("data_size"):
        n = len(group)
        if n == 1:
            df.loc[group.index, "annotation_offset"] = 10
        else:
            sorted_group = group.sort_values(by=y_column)
            for i, idx in enumerate(sorted_group.index):
                offset = 10 * (i - (n - 1) / 2)
                df.loc[idx, "annotation_offset"] = offset

    # Annotate points with category-specific colors and conditional formatting
    for idx, row in df.iterrows():
        offset = row["annotation_offset"]
        if use_log_scale:
            annotation = f"{row[y_column]:.3e}"  # Scientific notation for log scale
        else:
            annotation = f"{row[y_column]:.3f}"  # Fixed-point for linear scale
        ax.annotate(
            annotation,
            (row["data_size"], row[y_column]),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=8,
            color=category_colors[row["category"]],
        )

    plt.xticks(rotation=45, ha="right")
    ax.legend(fontsize=9)
    plt.tight_layout()

def draw_figure(csvpath: str, output_dir: str, prefix: str, use_log_scale=True):
    """Generate and save performance comparison figures."""
    df = pd.read_csv(csvpath)
    performance_data = []

    # Process each row to extract category and data_size
    for i, row in df.iterrows():
        name = row["name"]
        cpu_time = row["cpu_time"]
        parts = name.split("/")
        if len(parts) >= 3:
            category = parts[1]  # e.g., zuan_16_2, autovec_16
            data_size = "/".join(parts[2:])  # e.g., 128/256
        else:
            category = "Unknown"
            data_size = name
        size_product = compute_product_size(data_size)
        items_per_sec = size_product / cpu_time if cpu_time > 0 else 0
        performance_data.append(
            {
                "data_size": data_size,
                "size_product": size_product,
                "cpu_time": cpu_time,
                "items_per_sec": items_per_sec,
                "category": category,
            }
        )

    # Create and sort DataFrame
    perf_df = pd.DataFrame(performance_data)
    perf_df["cpu_time"] = pd.to_numeric(perf_df["cpu_time"], errors="coerce")
    perf_df["items_per_sec"] = pd.to_numeric(perf_df["items_per_sec"], errors="coerce")
    perf_df = perf_df.sort_values(by="size_product")
    perf_df["annotation_offset"] = 0

    # Assign colors: Zuan-related in blue shades, others in different colors
    unique_categories = perf_df["category"].unique()
    zuan_categories = [cat for cat in unique_categories if "zuan" in cat.lower()]
    other_categories = [cat for cat in unique_categories if cat not in zuan_categories]

    if zuan_categories:
        zuan_colors = sns.color_palette("Blues", n_colors=len(zuan_categories))
    else:
        zuan_colors = []

    if other_categories:
        other_colors = sns.color_palette("husl", n_colors=len(other_categories))
    else:
        other_colors = []

    category_colors = {cat: color for cat, color in zip(zuan_categories, zuan_colors)}
    category_colors.update(
        {cat: color for cat, color in zip(other_categories, other_colors)}
    )

    # Set up output directories and filenames
    os.makedirs(output_dir, exist_ok=True)
    output_dir = os.path.join(output_dir, prefix)
    perf_filename = (
        f"{output_dir}-performance-log.pdf"
        if use_log_scale
        else f"{output_dir}-performance.pdf"
    )
    throughput_filename = (
        f"{output_dir}-throughput-log.pdf"
        if use_log_scale
        else f"{output_dir}-throughput.pdf"
    )

    # Plot CPU time
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_and_annotate(
        ax, perf_df, "cpu_time", "CPU Time", "CPU Time", use_log_scale, category_colors
    )
    plt.savefig(perf_filename)
    plt.close(fig)

    # Plot throughput
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_and_annotate(
        ax,
        perf_df,
        "items_per_sec",
        "Items per Second",
        "Items per Second",
        use_log_scale,
        category_colors,
    )
    plt.savefig(throughput_filename)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvdir", type=str, required=True)
    args = parser.parse_args()

    output_dir = os.path.join(args.csvdir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    for csv in os.listdir(args.csvdir):
        if csv.endswith(".csv"):
            draw_figure(os.path.join(args.csvdir, csv), output_dir, csv.split(".")[0])

if __name__ == "__main__":
    main()