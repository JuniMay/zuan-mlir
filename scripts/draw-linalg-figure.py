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

    for data_size, group in df.groupby("data_size"):
        if len(group) == 1:
            df.loc[group.index, "annotation_offset"] = 10
        else:
            sorted_group = group.sort_values(by=y_column)
            df.loc[sorted_group.iloc[0].name, "annotation_offset"] = -15
            df.loc[sorted_group.iloc[1].name, "annotation_offset"] = 15

    for idx, row in df.iterrows():
        offset = row["annotation_offset"]
        ax.annotate(
            f"{row[y_column]:.2f}",
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
    for i, row in df.iterrows():
        name = row["name"]
        cpu_time = row["cpu_time"]
        data_size = "/".join(name.split("/")[2:])
        size_product = compute_product_size(data_size)
        items_per_sec = size_product / cpu_time if cpu_time > 0 else 0
        category = "Zuan" if "zuan" in name.lower() else "Base"
        performance_data.append(
            {
                "data_size": data_size,
                "size_product": size_product,
                "cpu_time": cpu_time,
                "items_per_sec": items_per_sec,
                "category": category,
            }
        )

    perf_df = pd.DataFrame(performance_data)
    perf_df["cpu_time"] = pd.to_numeric(perf_df["cpu_time"], errors="coerce")
    perf_df["items_per_sec"] = pd.to_numeric(perf_df["items_per_sec"], errors="coerce")
    perf_df = perf_df.sort_values(by="size_product")

    perf_df["annotation_offset"] = 0

    category_colors = {"Zuan": "blue", "Base": "orange"}

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

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_and_annotate(
        ax, perf_df, "cpu_time", "CPU Time", "CPU Time", use_log_scale, category_colors
    )
    plt.savefig(perf_filename)
    plt.close(fig)

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

    output_dir = args.csvdir + "/figures"
    os.makedirs(output_dir, exist_ok=True)

    for csv in os.listdir(args.csvdir):
        if csv.endswith(".csv"):
            draw_figure(args.csvdir + "/" + csv, output_dir, csv.split(".")[0])


if __name__ == "__main__":
    main()
