"""Generate LODO accuracy figures from the selected LSTM and ResNet CSV runs.

Run from any working directory:
    python graphics/generation_scripts/plot_lodo_results.py

The LSTM results come from the first run; the Temporal ResNet v2 results come
from the corrected second run. Both models are evaluated on the six common
gestures; the runs differ in training class scope, normalization, and
augmentation, so the figures do not isolate architecture effects.
"""

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
LOGS = ROOT / "lodo" / "logs"
OUTPUT = ROOT / "graphics"
INPUTS = (
    (LOGS / "log_2026-11-21_12-20-57_test_accuracies.csv", "lstm", "LSTM"),
    (
        LOGS / "log_2026-11-21_18-16-02_test_accuracies.csv",
        "temporal_resnet_v2",
        "Temporal ResNet v2",
    ),
)
TRAINING_LOGS = {
    "LSTM": LOGS / "log_2026-11-21_12-20-57.log",
    "Temporal ResNet v2": LOGS / "log_2026-11-21_18-16-02.log",
}
MODEL_IDS = {display_name: model_id for _, model_id, display_name in INPUTS}
DOMAINS = ("room", "user", "location", "orientation")
MODEL_COLORS = {"LSTM": "#2878B5", "Temporal ResNet v2": "#E76F32"}
MODEL_MARKERS = {"LSTM": "o", "Temporal ResNet v2": "s"}
DOMAIN_COLORS = {
    "room": "#6C5CE7",
    "user": "#159D8C",
    "location": "#D99A22",
    "orientation": "#D65A6A",
}
Y_MIN, Y_MAX = 0.30, 1.00


def load_model_results(path, expected_model):
    results = {}
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"model", "split", "domain", "held_out", "test_accuracy"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} is missing required columns: {sorted(required)}")

        for line_number, row in enumerate(reader, start=2):
            if row["model"] != expected_model:
                continue
            try:
                accuracy = float(row["test_accuracy"])
                if not 0.0 <= accuracy <= 1.0:
                    raise ValueError("accuracy must be between 0 and 1")

                if row["split"] == "in_domain":
                    key = ("in_domain", "", "")
                elif row["split"] == "lodo":
                    domain = row["domain"]
                    if domain not in DOMAINS:
                        raise ValueError(f"unknown LODO domain: {domain!r}")
                    key = ("lodo", domain, int(row["held_out"]))
                else:
                    raise ValueError(f"unknown split: {row['split']!r}")
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid data at {path}, line {line_number}: {error}") from error

            if key in results:
                raise ValueError(f"Duplicate result for {key} in {path}")
            results[key] = accuracy

    if not any(key[0] == "in_domain" for key in results):
        raise ValueError(f"No in-domain result for {expected_model} in {path}")
    if not any(key[0] == "lodo" for key in results):
        raise ValueError(f"No LODO results for {expected_model} in {path}")
    return results


def load_training_sizes(path, expected_model):
    run_pattern = re.compile(
        r"^Running (?P<model>[A-Za-z0-9_]+): "
        r"(?P<domain>room|user|location|orientation)=(?P<held_out>\d+)\b"
    )
    split_pattern = re.compile(
        r"^\s*split: Held out (?P<domain>room|user|location|orientation) number "
        r"(?P<held_out>\d+); train=(?P<train>\d+),"
    )
    training_sizes = {}
    active_run = None

    with path.open(encoding="utf-8") as log_file:
        for line_number, line in enumerate(log_file, start=1):
            if line.startswith("Running "):
                run_match = run_pattern.match(line)
                active_run = None
                if run_match and run_match["model"] == expected_model:
                    active_run = (run_match["domain"], int(run_match["held_out"]))
                continue

            if active_run is None:
                continue
            split_match = split_pattern.match(line)
            if split_match is None:
                continue

            key = (split_match["domain"], int(split_match["held_out"]))
            if key != active_run:
                raise ValueError(f"Run header and split disagree at {path}, line {line_number}")
            if key in training_sizes:
                raise ValueError(f"Duplicate training size for {key} in {path}")
            training_sizes[key] = int(split_match["train"])
            active_run = None

    if not training_sizes:
        raise ValueError(f"No LODO training sizes for {expected_model} in {path}")
    return training_sizes


def load_selected_results():
    models = {
        display_name: load_model_results(path, model_id)
        for path, model_id, display_name in INPUTS
    }
    training_sizes = {}

    reference = models["LSTM"]
    for model_name, results in models.items():
        lodo_keys = {key for key in results if key[0] == "lodo"}
        if lodo_keys != {
            key for key in reference if key[0] == "lodo"
        }:
            raise ValueError(f"{model_name} does not have the same LODO folds as the LSTM run")
        sizes = load_training_sizes(TRAINING_LOGS[model_name], MODEL_IDS[model_name])
        size_keys = {("lodo", domain, held_out) for domain, held_out in sizes}
        if size_keys != lodo_keys:
            raise ValueError(f"{model_name} training sizes do not match its LODO CSV rows")
        training_sizes[model_name] = sizes
    return models, training_sizes


def style_axes(axis):
    axis.set_ylim(Y_MIN, Y_MAX)
    axis.set_yticks((0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
    axis.set_yticklabels(("40%", "50%", "60%", "70%", "80%", "90%", "100%"))
    axis.grid(axis="y", color="#DCE2E8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AAB4BF")
    axis.spines["bottom"].set_color("#AAB4BF")
    axis.tick_params(colors="#48525D")


def domain_values(results, domain):
    return sorted(
        (held_out, score)
        for (split, result_domain, held_out), score in results.items()
        if split == "lodo" and result_domain == domain
    )


def make_training_size_accuracy(models, training_sizes):
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.5), sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    for axis, model_name in zip(axes, ("LSTM", "Temporal ResNet v2")):
        style_axes(axis)
        sizes = training_sizes[model_name]
        all_sizes = [size for size in sizes.values()]
        mean_size = sum(all_sizes) / len(all_sizes)

        for domain in DOMAINS:
            folds = domain_values(models[model_name], domain)
            x_values = [sizes[(domain, held_out)] / 1000 for held_out, _ in folds]
            y_values = [score for _, score in folds]
            axis.scatter(
                x_values,
                y_values,
                s=54,
                color=DOMAIN_COLORS[domain],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.88,
                label=domain.title(),
                zorder=3,
            )

        axis.axvline(mean_size / 1000, color="#202833", linestyle="--", linewidth=1.1, zorder=2)
        axis.text(
            0.97,
            0.05,
            f"Mean training size: {mean_size:,.0f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#202833",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
        )
        padding = max(0.8, (max(all_sizes) - min(all_sizes)) / 1000 * 0.08)
        axis.set_xlim(min(all_sizes) / 1000 - padding, max(all_sizes) / 1000 + padding)
        axis.set_xlabel("Training set size (thousands of samples)", fontsize=10)
        axis.set_title(model_name, loc="left", fontsize=12, fontweight="bold")
        axis.legend(frameon=False, loc="upper left", ncol=2, fontsize=8)

    axes[0].set_ylabel("Test accuracy", fontsize=10)
    fig.suptitle(
        "Training-set size versus LODO accuracy",
        x=0.06,
        y=1.045,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.99,
        "Each point is one LODO run; x is the logged training sample count and y is its CSV test accuracy. "
        "Dashed lines show each model's mean training size.",
        ha="left",
        fontsize=10,
        color="#48525D",
    )
    fig.text(
        0.06,
        0.015,
        "Associations are descriptive: held-out domain and test-set composition also vary. "
        "Training class scope differs between models.",
        ha="left",
        fontsize=8.5,
        color="#68737D",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.94), w_pad=1.8)
    output_path = OUTPUT / "lodo_training_size_vs_accuracy.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def make_domain_overview(models):
    domain_names = ("in_domain", "user", "room", "location", "orientation")
    fold_counts = {
        domain: len(domain_values(models["LSTM"], domain))
        for domain in DOMAINS
    }
    widths = [0.9, 1.45, 0.9, 1.0, 0.95]
    fig, axes = plt.subplots(
        1,
        len(domain_names),
        figsize=(13.5, 6.0),
        sharey=True,
        gridspec_kw={"width_ratios": widths},
    )
    fig.patch.set_facecolor("#FFFFFF")

    for axis, domain in zip(axes, domain_names):
        style_axes(axis)
        axis.set_xlim(-0.45, 1.45)
        axis.set_xticks((0, 1), ("LSTM", "ResNet\nv2"))
        axis.tick_params(axis="x", labelsize=9)

        for model_index, model_name in enumerate(("LSTM", "Temporal ResNet v2")):
            if domain == "in_domain":
                scores = [models[model_name][("in_domain", "", "")]]
            else:
                scores = [score for _, score in domain_values(models[model_name], domain)]

            fold_ids = (
                [held_out for held_out, _ in domain_values(models[model_name], domain)]
                if domain != "in_domain"
                else [0]
            )
            color = MODEL_COLORS[model_name]
            jitter = [((fold_id * 0.61803398875) % 1 - 0.5) * 0.24 for fold_id in fold_ids]
            x_positions = [model_index + value for value in jitter]
            axis.scatter(
                x_positions,
                scores,
                s=37,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.78,
                zorder=3,
            )
            mean_score = sum(scores) / len(scores)
            axis.scatter(
                [model_index],
                [mean_score],
                marker="D",
                s=58,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
            )
            axis.annotate(
                f"{mean_score:.0%}",
                (model_index, mean_score),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                fontweight="bold",
                color="#202833",
            )

        if domain == "in_domain":
            axis.set_title("In-domain\n1 run each", loc="left", fontsize=11, fontweight="bold")
        else:
            axis.set_title(
                f"Held-out {domain}\nn = {fold_counts[domain]} folds",
                loc="left",
                fontsize=11,
                fontweight="bold",
            )

    axes[0].set_ylabel("Test accuracy", fontsize=11)
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=MODEL_COLORS["LSTM"], markeredgecolor="white",
               markersize=8, label="LSTM"),
        Line2D([0], [0], marker="o", linestyle="none", color="none",
               markerfacecolor=MODEL_COLORS["Temporal ResNet v2"], markeredgecolor="white",
               markersize=8, label="Temporal ResNet v2"),
        Line2D([0], [0], marker="D", linestyle="none", color="#48525D",
               markersize=7, label="Unweighted mean"),
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.98, 0.995), ncol=3, frameon=False)
    fig.suptitle(
        "Accuracy across in-domain and leave-one-domain-out tests",
        x=0.06,
        y=1.08,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        1.025,
        "Both models are scored on the same six common gestures. Training scope differs "
        "(LSTM: 6 classes; ResNet: all 22), as do normalization and augmentation.",
        ha="left",
        fontsize=10,
        color="#48525D",
    )
    fig.text(
        0.06,
        -0.015,
        "Sources: LSTM from the first run CSV; Temporal ResNet v2 from the corrected second run CSV. "
        "Means weight each fold equally.",
        ha="left",
        fontsize=8.5,
        color="#68737D",
    )
    fig.tight_layout(rect=(0.04, 0.045, 0.98, 0.91), w_pad=0.8)
    output_path = OUTPUT / "lodo_accuracy_by_domain.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def make_generalization_drop(models):
    domain_names = ("user", "room", "location", "orientation")
    domain_labels = ("User\n17 folds", "Room\n3 folds", "Location\n8 folds", "Orientation\n5 folds")
    model_names = ("LSTM", "Temporal ResNet v2")
    drops = {model_name: [] for model_name in model_names}

    for model_name in model_names:
        baseline = models[model_name][("in_domain", "", "")]
        for domain in domain_names:
            scores = [score for _, score in domain_values(models[model_name], domain)]
            drops[model_name].append((baseline - sum(scores) / len(scores)) * 100)

    fig, axis = plt.subplots(figsize=(11.5, 6.2))
    fig.patch.set_facecolor("#FFFFFF")
    positions = list(range(len(domain_names)))
    bar_width = 0.34

    for model_index, model_name in enumerate(model_names):
        offset = (model_index - 0.5) * bar_width
        bars = axis.bar(
            [position + offset for position in positions],
            drops[model_name],
            width=bar_width,
            color=MODEL_COLORS[model_name],
            edgecolor="white",
            linewidth=0.8,
            label=model_name,
            zorder=3,
        )
        for bar, value in zip(bars, drops[model_name]):
            label_y = value + 0.45 if value >= 0 else value - 0.45
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:.1f} pp",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
                fontweight="bold",
                color="#202833",
            )

    max_drop = max(max(values) for values in drops.values())
    min_drop = min(min(values) for values in drops.values())
    padding = max(1.5, (max_drop - min(0, min_drop)) * 0.12)
    axis.set_ylim(min(0, min_drop - padding), max_drop + padding)
    axis.set_xticks(positions, domain_labels)
    axis.set_ylabel("Accuracy drop (percentage points)", fontsize=11)
    axis.axhline(0, color="#68737D", linewidth=1.0, zorder=2)
    axis.grid(axis="y", color="#DCE2E8", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#AAB4BF")
    axis.spines["bottom"].set_color("#AAB4BF")
    axis.tick_params(colors="#48525D")
    axis.legend(frameon=False, loc="upper right", ncol=2)

    fig.suptitle(
        "Generalization drop by held-out domain",
        x=0.06,
        y=1.04,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.985,
        "Drop = in-domain accuracy minus mean LODO accuracy; positive values indicate lower accuracy under shift. "
        "Both models are scored on the same six common gestures.",
        ha="left",
        fontsize=10,
        color="#48525D",
    )
    fig.text(
        0.06,
        0.015,
        "Fold means weight each held-out ID equally. Training scope, normalization, and augmentation differ between runs.",
        ha="left",
        fontsize=8.5,
        color="#68737D",
    )
    fig.tight_layout(rect=(0.04, 0.06, 0.98, 0.92))
    output_path = OUTPUT / "lodo_generalization_drop_by_domain.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def make_fold_detail(models):
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5), sharey=True)
    fig.patch.set_facecolor("#FFFFFF")

    for axis, domain in zip(axes.flat, ("user", "room", "location", "orientation")):
        style_axes(axis)
        reference_values = domain_values(models["LSTM"], domain)
        held_out_ids = [held_out for held_out, _ in reference_values]

        for model_index, model_name in enumerate(("LSTM", "Temporal ResNet v2")):
            values = dict(domain_values(models[model_name], domain))
            offset = -0.13 if model_index == 0 else 0.13
            axis.scatter(
                [held_out + offset for held_out in held_out_ids],
                [values[held_out] for held_out in held_out_ids],
                s=49,
                marker=MODEL_MARKERS[model_name],
                color=MODEL_COLORS[model_name],
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
            )

        axis.set_xticks(held_out_ids, [str(value) for value in held_out_ids])
        axis.set_xlim(min(held_out_ids) - 0.6, max(held_out_ids) + 0.6)
        axis.set_xlabel(f"Held-out {domain} ID", fontsize=10)
        axis.set_title(f"{domain.title()} folds (n = {len(held_out_ids)})", loc="left",
                       fontsize=12, fontweight="bold")
        axis.tick_params(axis="x", labelsize=8, rotation=35 if len(held_out_ids) > 8 else 0)

    axes[0, 0].set_ylabel("Test accuracy", fontsize=10)
    axes[1, 0].set_ylabel("Test accuracy", fontsize=10)
    legend = [
        Line2D([0], [0], marker=MODEL_MARKERS[name], linestyle="none", color="none",
               markerfacecolor=MODEL_COLORS[name], markeredgecolor="white",
               markersize=8, label=name)
        for name in ("LSTM", "Temporal ResNet v2")
    ]
    fig.legend(handles=legend, loc="upper right", bbox_to_anchor=(0.97, 0.985), ncol=2, frameon=False)
    fig.suptitle(
        "Fold-by-fold test accuracy",
        x=0.06,
        y=1.035,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.06,
        0.985,
        "Both models are scored on the same six common gestures. Training scope differs "
        "(LSTM: 6 classes; ResNet: all 22); normalization and augmentation also differ.",
        ha="left",
        fontsize=10,
        color="#48525D",
    )
    fig.text(
        0.06,
        0.015,
        "Sources: LSTM from the first run CSV; Temporal ResNet v2 from the corrected second run CSV.",
        ha="left",
        fontsize=8.5,
        color="#68737D",
    )
    fig.tight_layout(rect=(0.04, 0.05, 0.98, 0.94), h_pad=2.2, w_pad=1.8)
    output_path = OUTPUT / "lodo_accuracy_by_fold.png"
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main():
    models, training_sizes = load_selected_results()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outputs = (
        make_domain_overview(models),
        make_fold_detail(models),
        make_generalization_drop(models),
        make_training_size_accuracy(models, training_sizes),
    )
    for output in outputs:
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
