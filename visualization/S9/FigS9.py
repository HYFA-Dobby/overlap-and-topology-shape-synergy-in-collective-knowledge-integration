from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


"""
Visualization of Fig. S9.

Fig. S9A:
    Strict external-synergy discovery events as a function of
    overlap degree D_ov for group sizes n = 2,...,10.

Fig. S9B:
    Composition of common-region discovery events evaluated
    at the strict-synergy peak for each group size.

Fig. S9C:
    Maximum discovery events under three criteria:
        strict        : r = n
        fixed-order   : r >= 2
        half-coverage : r >= ceil(n / 2)

Input:
    visualization_data/Fig.S3/small_world/
        WS_syn_result_2agent.csv
        ...
        WS_syn_result_10agent.csv

CSV format:
    ov, pattern, mean, sem

Output:
    figures/Fig.S9/
"""


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    "./visualization_data/Fig.3/small_world"
)

OUTPUT_DIR = Path(
    "./figures/Fig.S9"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

NETWORK_LABEL = "WS"

GROUP_SIZES = range(
    2,
    11
)

SUBNETWORK_SIZE = 100

# Common region + at least two individual-specific regions
FIXED_ORDER = 2

# Common region + at least half of individual-specific regions
FRACTION = 0.5


# ============================================================
# Figure style
# ============================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial",
        "DejaVu Sans"
    ],
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6.8,
    "mathtext.fontset": "stixsans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Data functions
# ============================================================

def find_csv(
    data_dir: Path,
    n: int
) -> Path:
    """
    Return the Boolean-pattern summary CSV
    for group size n.
    """

    path = (
        data_dir
        / f"{NETWORK_LABEL}_syn_result_{n}agent.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            f"CSV for n={n} was not found:\n"
            f"{path.resolve()}"
        )

    return path


def load_pattern_data(
    path: Path,
    n: int
) -> pd.DataFrame:
    """
    Load one CSV and recover the Boolean vector.

    Vector convention:
        first n bits : connection to each
                       individual-specific region
        final bit    : connection to common region

    Example for n = 3:
        1011
        -> unique regions 1 and 3
           + common region
    """

    df = pd.read_csv(
        path
    )

    required = {
        "ov",
        "pattern",
        "mean",
        "sem"
    }

    missing = required.difference(
        df.columns
    )

    if missing:

        raise ValueError(
            f"{path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Recover leading zeros
    # --------------------------------------------------------

    df["pattern"] = (
        df["pattern"]
        .astype("Int64")
        .astype(str)
        .str.zfill(n + 1)
    )

    if (
        ~df["pattern"]
        .str.fullmatch(r"[01]+")
    ).any():

        raise ValueError(
            f"Non-binary pattern found in "
            f"{path.name}"
        )

    # --------------------------------------------------------
    # Pattern information
    # --------------------------------------------------------

    df["unique_bits"] = (
        df["pattern"]
        .str[:n]
    )

    df["common"] = (
        df["pattern"]
        .str[-1]
        .astype(int)
    )

    df["order"] = (
        df["unique_bits"]
        .map(
            lambda s: s.count("1")
        )
    )

    df["group_size"] = n

    # --------------------------------------------------------
    # Convert overlap node number to D_ov
    # --------------------------------------------------------

    if df["ov"].max() > 1:

        df["D_ov"] = (
            df["ov"]
            / SUBNETWORK_SIZE
        )

    else:

        df["D_ov"] = (
            df["ov"]
            .astype(float)
        )

    return df


def criterion_curve(
    df: pd.DataFrame,
    n: int,
    criterion: str
) -> pd.DataFrame:
    """
    Return the mean discovery count
    for a criterion at every D_ov.
    """

    if criterion == "strict":

        mask = (
            (df["common"] == 1)
            & (df["order"] == n)
        )

    elif criterion == "fixed":

        mask = (
            (df["common"] == 1)
            & (df["order"] >= FIXED_ORDER)
        )

    elif criterion == "fractional":

        threshold = math.ceil(
            FRACTION * n
        )

        mask = (
            (df["common"] == 1)
            & (df["order"] >= threshold)
        )

    else:

        raise ValueError(
            f"Unknown criterion: "
            f"{criterion}"
        )

    overlap_grid = (
        df[
            [
                "ov",
                "D_ov"
            ]
        ]
        .drop_duplicates()
        .sort_values("ov")
        .reset_index(drop=True)
    )

    aggregated = (
        df.loc[mask]
        .groupby(
            [
                "ov",
                "D_ov"
            ],
            as_index=False
        )["mean"]
        .sum()
        .rename(
            columns={
                "mean": criterion
            }
        )
    )

    return (
        overlap_grid
        .merge(
            aggregated,
            on=[
                "ov",
                "D_ov"
            ],
            how="left"
        )
        .fillna({
            criterion: 0.0
        })
    )


# ============================================================
# Calculate panel data
# ============================================================

strict_curves = {}

composition_records = []

peak_records = []


for n in GROUP_SIZES:

    df = load_pattern_data(
        find_csv(
            DATA_DIR,
            n
        ),
        n
    )


    # ========================================================
    # Panel A
    # Strict criterion
    # ========================================================

    strict_pattern = (
        "1"
        * (n + 1)
    )

    overlap_grid = (
        df[
            [
                "ov",
                "D_ov"
            ]
        ]
        .drop_duplicates()
        .sort_values("ov")
        .reset_index(drop=True)
    )

    strict = (
        overlap_grid
        .merge(
            df.loc[
                df["pattern"]
                == strict_pattern,
                [
                    "ov",
                    "D_ov",
                    "mean",
                    "sem"
                ]
            ],
            on=[
                "ov",
                "D_ov"
            ],
            how="left"
        )
        .fillna({
            "mean": 0.0,
            "sem": 0.0
        })
    )

    strict_curves[
        n
    ] = strict


    # --------------------------------------------------------
    # Strict-synergy peak
    # --------------------------------------------------------

    strict_peak_index = (
        strict["mean"]
        .idxmax()
    )

    strict_peak_ov = int(
        strict.loc[
            strict_peak_index,
            "ov"
        ]
    )

    strict_peak_D = float(
        strict.loc[
            strict_peak_index,
            "D_ov"
        ]
    )

    strict_peak_mean = float(
        strict.loc[
            strict_peak_index,
            "mean"
        ]
    )

    strict_peak_sem = float(
        strict.loc[
            strict_peak_index,
            "sem"
        ]
    )


    # ========================================================
    # Panel B
    # Composition at strict-synergy peak
    # ========================================================

    peak_df = df[
        (df["ov"] == strict_peak_ov)
        & (df["common"] == 1)
    ].copy()

    common_total = (
        peak_df["mean"]
        .sum()
    )

    half_order = math.ceil(
        n / 2
    )

    categories = {

        "Common only":
            peak_df["order"] == 0,

        "Low-order partial":
            (
                (peak_df["order"] >= 1)
                & (
                    peak_df["order"]
                    < half_order
                )
            ),

        "High-order partial":
            (
                (
                    peak_df["order"]
                    >= half_order
                )
                & (
                    peak_df["order"]
                    < n
                )
            ),

        "Strict":
            peak_df["order"] == n
    }

    for category, mask in categories.items():

        category_mean = (
            peak_df.loc[
                mask,
                "mean"
            ]
            .sum()
        )

        percentage = (
            100.0
            * category_mean
            / common_total
            if common_total > 0
            else np.nan
        )

        composition_records.append({
            "group_size":
                n,

            "strict_peak_ov":
                strict_peak_ov,

            "strict_peak_D_ov":
                strict_peak_D,

            "category":
                category,

            "percentage":
                percentage
        })


    # ========================================================
    # Panel C
    # Criterion comparison
    # ========================================================

    criterion_results = {}

    for criterion in (
        "strict",
        "fixed",
        "fractional"
    ):

        curve = criterion_curve(
            df,
            n,
            criterion
        )

        peak_index = (
            curve[criterion]
            .idxmax()
        )

        criterion_results[
            criterion
        ] = {

            "peak":
                float(
                    curve.loc[
                        peak_index,
                        criterion
                    ]
                ),

            "peak_D_ov":
                float(
                    curve.loc[
                        peak_index,
                        "D_ov"
                    ]
                )
        }

    peak_records.append({

        "group_size":
            n,

        "strict_peak":
            strict_peak_mean,

        "strict_sem_at_peak":
            strict_peak_sem,

        "strict_peak_D_ov":
            strict_peak_D,

        "fixed_peak":
            criterion_results[
                "fixed"
            ]["peak"],

        "fixed_peak_D_ov":
            criterion_results[
                "fixed"
            ]["peak_D_ov"],

        "fractional_peak":
            criterion_results[
                "fractional"
            ]["peak"],

        "fractional_peak_D_ov":
            criterion_results[
                "fractional"
            ]["peak_D_ov"]
    })


composition = pd.DataFrame(
    composition_records
)

peaks = pd.DataFrame(
    peak_records
)


# ============================================================
# Save numerical values plotted in B and C
# ============================================================

composition.to_csv(
    OUTPUT_DIR
    / "SI_connection_order_composition.csv",
    index=False
)

peaks.to_csv(
    OUTPUT_DIR
    / "SI_criterion_peak_summary.csv",
    index=False
)


# ============================================================
# Draw the three panels
# ============================================================

fig, (
    ax_a,
    ax_b,
    ax_c
) = plt.subplots(
    1,
    3,
    figsize=(
        7.15,
        2.85
    ),
    gridspec_kw={
        "width_ratios": [
            1.15,
            1.20,
            1.16
        ]
    }
)


# ============================================================
# A. Strict-synergy curves
# ============================================================

cmap = plt.get_cmap(
    "viridis"
)

norm = Normalize(
    vmin=min(GROUP_SIZES),
    vmax=max(GROUP_SIZES)
)


for n in GROUP_SIZES:

    curve = strict_curves[
        n
    ]

    color = cmap(
        norm(n)
    )

    ax_a.plot(
        curve["D_ov"],
        curve["mean"],
        color=color,
        linewidth=1.25
    )

    ax_a.fill_between(
        curve["D_ov"],
        np.maximum(
            curve["mean"]
            - curve["sem"],
            0
        ),
        curve["mean"]
        + curve["sem"],
        color=color,
        alpha=0.10,
        linewidth=0
    )


ax_a.set_xlabel(
    r"Overlap degree, $D_{ov}$"
)

ax_a.set_ylabel(
    "Strict discovery events"
)

ax_a.set_xlim(
    0,
    1
)

ax_a.set_ylim(
    bottom=0
)


scalar_mappable = ScalarMappable(
    norm=norm,
    cmap=cmap
)

scalar_mappable.set_array(
    []
)

cbar = fig.colorbar(
    scalar_mappable,
    ax=ax_a,
    pad=0.025,
    fraction=0.062,
    ticks=list(
        GROUP_SIZES
    )
)

cbar.ax.set_title(
    r"$n$",
    pad=4
)


# ============================================================
# B. Connection-order composition
# ============================================================

category_order = [
    "Common only",
    "Low-order partial",
    "High-order partial",
    "Strict"
]

category_labels = [
    r"Common only ($r=0$)",
    "Low-order partial",
    "High-order partial",
    r"Strict ($r=n$)"
]


x = peaks[
    "group_size"
].to_numpy()

bottom = np.zeros(
    len(x)
)

legend_handles = []


for category, label in zip(
    category_order,
    category_labels
):

    values = (
        composition.loc[
            composition["category"]
            == category
        ]
        .set_index(
            "group_size"
        )
        .reindex(
            x
        )["percentage"]
        .to_numpy()
    )

    bars = ax_b.bar(
        x,
        values,
        bottom=bottom,
        width=0.72,
        edgecolor="white",
        linewidth=0.35,
        label=label
    )

    legend_handles.append(
        bars[0]
    )

    bottom += values


ax_b.set_xlabel(
    r"Group size, $n$"
)

ax_b.set_ylabel(
    "Composition (%)"
)

ax_b.set_xticks(
    x
)

ax_b.set_ylim(
    0,
    100
)


fig.legend(
    legend_handles,
    category_labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(
        0.53,
        0.995
    ),
    ncol=4,
    handlelength=1.25,
    columnspacing=0.9
)


# ============================================================
# C. Criterion comparison
# ============================================================

line_strict, = ax_c.plot(
    peaks["group_size"],
    peaks["strict_peak"],
    marker="o",
    linewidth=1.25
)

line_fixed, = ax_c.plot(
    peaks["group_size"],
    peaks["fixed_peak"],
    marker="s",
    linewidth=1.25
)

line_fractional, = ax_c.plot(
    peaks["group_size"],
    peaks["fractional_peak"],
    marker="^",
    linewidth=1.25
)


ax_c.set_xlabel(
    r"Group size, $n$"
)

ax_c.set_ylabel(
    "Peak discovery events"
)

ax_c.set_xticks(
    peaks["group_size"]
)

ax_c.set_xlim(
    1.7,
    11.8
)

ax_c.set_ylim(
    0,
    max(
        peaks["fixed_peak"].max(),
        peaks["fractional_peak"].max()
    )
    * 1.08
)


# ------------------------------------------------------------
# Direct labels
# ------------------------------------------------------------

last_n = int(
    peaks[
        "group_size"
    ].iloc[-1]
)


ax_c.text(
    last_n + 0.40,
    peaks[
        "strict_peak"
    ].iloc[-1],
    r"Strict: $r=n$",
    color=line_strict.get_color(),
    va="center",
    fontsize=6.8
)

ax_c.text(
    last_n + 0.40,
    peaks[
        "fixed_peak"
    ].iloc[-1],
    rf"Fixed: $r\geq {FIXED_ORDER}$",
    color=line_fixed.get_color(),
    va="center",
    fontsize=6.8
)

ax_c.text(
    last_n + 0.40,
    peaks[
        "fractional_peak"
    ].iloc[-1],
    rf"Half-coverage: "
    rf"$r\geq\lceil n/2\rceil$",
    color=line_fractional.get_color(),
    va="center",
    fontsize=6.8
)


# ============================================================
# Layout
# ============================================================

fig.subplots_adjust(
    left=0.075,
    right=0.97,
    bottom=0.19,
    top=0.79,
    wspace=0.60
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUTPUT_DIR
    / "Fig.S9.pdf",
    bbox_inches="tight"
)

plt.close(
    fig
)


print(
    "\nSaved:"
)

print(
    OUTPUT_DIR
    / "Fig.S9.pdf"
)

print(
    OUTPUT_DIR
    / "Fig.S9.png"
)

print(
    OUTPUT_DIR
    / "SI_connection_order_composition.csv"
)

print(
    OUTPUT_DIR
    / "SI_criterion_peak_summary.csv"
)

print(
    "\nPeak summary:"
)

print(
    peaks
    .round(3)
    .to_string(index=False)
)