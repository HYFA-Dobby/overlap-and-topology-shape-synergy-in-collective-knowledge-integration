import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.optimize import least_squares
from matplotlib.lines import Line2D


"""
Reproduction of Figure S11.

Phenomenological effective-contact approximation of external synergy.

Panel A:
    Effective per-contact probabilities theta_U and theta_R
    for weak rewiring (p = 1e-4).

Panel B:
    Boundary-conditioned probabilities for:
        - A-specific only
        - Shared-region only
        - Strict synergy (A + B + R)

Panel C:
    Simulated boundary concentration rho_syn from Fig. 4C
    compared with the fitted effective-contact approximation
    across rewiring probabilities.

Input:
    visualization_data/Fig.4/boundary/
        syn_result_ps{p_s}.csv

Required columns:
    D_ov
    syn_ratio_mean

Output:
    visualization_data/Fig.S11/
        fitted_effective_contact_parameters.csv

    figures/Fig.S11/
        Fig.S11.pdf
"""


# ============================================================
# Paths
# ============================================================

data_dir = Path(
    "./visualization_data/Fig.4/boundary"
)

output_data_dir = Path(
    "./visualization_data/Fig.S11"
)

figure_dir = Path(
    "./figures/Fig.S11"
)

output_data_dir.mkdir(
    parents=True,
    exist_ok=True
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Settings
# ============================================================

K = 20

# Figure S11C uses the 10 positive rewiring probabilities only.
prob_list = np.logspace(
    -4,
    0,
    10
)

weak_rewiring = 1e-4


# ============================================================
# Effective-contact model
# ============================================================

def theta_values(
    D_ov,
    phi,
    eta
):
    """
    Effective per-contact probabilities.

    theta_A = theta_B = theta_U
    theta_R = shared-region probability
    """

    D_ov = np.asarray(
        D_ov,
        dtype=float
    )

    denominator = (
        2 * (1 - D_ov)
        + eta * D_ov**2
    )

    theta_U = np.divide(
        phi * (1 - D_ov),
        denominator,
        out=np.zeros_like(
            D_ov,
            dtype=float
        ),
        where=denominator > 0
    )

    theta_R = np.divide(
        phi * eta * D_ov**2,
        denominator,
        out=np.zeros_like(
            D_ov,
            dtype=float
        ),
        where=denominator > 0
    )

    return (
        theta_U,
        theta_R
    )


def strict_synergy_probability(
    D_ov,
    phi,
    eta,
    k=K
):
    """
    Boundary-conditioned strict synergy probability.

    Strict synergy requires at least one connection to:
        U_A
        U_B
        R
    """

    theta_U, theta_R = theta_values(
        D_ov,
        phi,
        eta
    )

    theta_0 = (
        1
        - 2 * theta_U
        - theta_R
    )


    # --------------------------------------------------------
    # Unconditional strict synergy probability
    # --------------------------------------------------------

    P_syn = (
        1
        - 2 * (1 - theta_U)**k
        - (1 - theta_R)**k
        + (1 - 2 * theta_U)**k
        + 2 * (1 - theta_U - theta_R)**k
        - theta_0**k
    )


    # --------------------------------------------------------
    # External-boundary probability
    # --------------------------------------------------------

    P_boundary = (
        1
        - theta_0**k
    )


    return np.divide(
        P_syn,
        P_boundary,
        out=np.zeros_like(
            P_syn,
            dtype=float
        ),
        where=P_boundary > 0
    )


# ============================================================
# Probabilities used in Panel B
# ============================================================

def boundary_probabilities(
    D_ov,
    phi,
    eta,
    k=K
):

    theta_U, theta_R = theta_values(
        D_ov,
        phi,
        eta
    )

    theta_0 = (
        1
        - 2 * theta_U
        - theta_R
    )

    P_boundary = (
        1
        - theta_0**k
    )


    # --------------------------------------------------------
    # A-specific only
    # --------------------------------------------------------

    P_A_only = (
        (1 - theta_U - theta_R)**k
        - theta_0**k
    )


    # --------------------------------------------------------
    # Shared region only
    # --------------------------------------------------------

    P_R_only = (
        (1 - 2 * theta_U)**k
        - theta_0**k
    )


    # --------------------------------------------------------
    # Strict synergy
    # --------------------------------------------------------

    P_syn = (
        1
        - 2 * (1 - theta_U)**k
        - (1 - theta_R)**k
        + (1 - 2 * theta_U)**k
        + 2 * (1 - theta_U - theta_R)**k
        - theta_0**k
    )


    P_A_only_cond = np.divide(
        P_A_only,
        P_boundary,
        out=np.zeros_like(
            P_A_only,
            dtype=float
        ),
        where=P_boundary > 0
    )

    P_R_only_cond = np.divide(
        P_R_only,
        P_boundary,
        out=np.zeros_like(
            P_R_only,
            dtype=float
        ),
        where=P_boundary > 0
    )

    P_syn_cond = np.divide(
        P_syn,
        P_boundary,
        out=np.zeros_like(
            P_syn,
            dtype=float
        ),
        where=P_boundary > 0
    )


    return (
        P_A_only_cond,
        P_R_only_cond,
        P_syn_cond
    )


# ============================================================
# Load Fig. 4C data
# ============================================================

def load_boundary_data(
    p_s
):

    path = (
        data_dir
        / f"syn_result_ps{p_s}.csv"
    )

    df = pd.read_csv(
        path
    )


    if "D_ov" not in df.columns:

        if "ov" not in df.columns:

            raise ValueError(
                f"{path.name} requires either "
                "'D_ov' or 'ov'."
            )

        df["D_ov"] = (
            df["ov"]
            / 100
        )


    if "syn_ratio_mean" not in df.columns:

        raise ValueError(
            f"{path.name} does not contain "
            "'syn_ratio_mean'."
        )


    df = (
        df[
            [
                "D_ov",
                "syn_ratio_mean"
            ]
        ]
        .dropna()
        .sort_values(
            "D_ov"
        )
        .reset_index(
            drop=True
        )
    )


    return df


# ============================================================
# Fit phi_p and eta_p
# ============================================================

def fit_effective_contact(
    df
):

    x = (
        df["D_ov"]
        .to_numpy(
            dtype=float
        )
    )

    y = (
        df["syn_ratio_mean"]
        .to_numpy(
            dtype=float
        )
    )


    def residuals(
        params
    ):

        phi, eta = params

        y_fit = (
            strict_synergy_probability(
                x,
                phi,
                eta
            )
        )

        return (
            y_fit - y
        )


    result = least_squares(
        residuals,
        x0=[
            0.2,
            0.3
        ],
        bounds=(
            [
                1e-8,
                1e-8
            ],
            [
                0.999999,
                np.inf
            ]
        )
    )


    phi_fit = float(
        result.x[0]
    )

    eta_fit = float(
        result.x[1]
    )

    sse = float(
        np.sum(
            result.fun**2
        )
    )


    return (
        phi_fit,
        eta_fit,
        sse
    )


# ============================================================
# Fit all rewiring probabilities
# ============================================================

fit_records = []

simulation_data = {}


for p_s in prob_list:

    df = load_boundary_data(
        p_s
    )

    simulation_data[
        p_s
    ] = df


    phi_fit, eta_fit, sse = (
        fit_effective_contact(
            df
        )
    )


    fit_records.append({

        "p_s":
            p_s,

        "phi":
            phi_fit,

        "eta":
            eta_fit,

        "SSE":
            sse
    })


fit_df = pd.DataFrame(
    fit_records
)


# ============================================================
# Save fitted parameters
# ============================================================

parameter_path = (
    output_data_dir
    / "fitted_effective_contact_parameters.csv"
)

fit_df.to_csv(
    parameter_path,
    index=False
)


# ============================================================
# Weak-rewiring parameters for Panels A and B
# ============================================================

weak_row = fit_df[
    np.isclose(
        fit_df["p_s"],
        weak_rewiring
    )
].iloc[0]


phi_weak = float(
    weak_row["phi"]
)

eta_weak = float(
    weak_row["eta"]
)


# ============================================================
# Smooth overlap grid
# ============================================================

D_smooth = np.linspace(
    0,
    1,
    500
)


# ============================================================
# Figure layout
# ============================================================

fig = plt.figure(
    figsize=(
        8,
        6
    )
)


gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[
        1,
        1.15
    ],
    hspace=0.45,
    wspace=0.40
)


ax_a = fig.add_subplot(
    gs[0, 0]
)

ax_b = fig.add_subplot(
    gs[0, 1]
)

ax_c = fig.add_subplot(
    gs[1, :]
)


# ============================================================
# Panel A
# Effective contact probabilities
# ============================================================

theta_U, theta_R = theta_values(
    D_smooth,
    phi_weak,
    eta_weak
)


ax_a.plot(
    D_smooth,
    theta_U,
    linewidth=2,
    label=r"$\theta_U=\theta_A=\theta_B$"
)

ax_a.plot(
    D_smooth,
    theta_R,
    linewidth=2,
    label=r"$\theta_R$"
)


ax_a.set_xlim(
    0,
    1
)

ax_a.set_ylim(
    bottom=0
)

ax_a.set_xlabel(
    r"Overlap degree, $D_{ov}$"
)

ax_a.set_ylabel(
    "Effective contact\nprobability",
    labelpad=5
)

ax_a.set_title(
    r"Weak rewiring: $p=10^{-4}$",
    fontsize=10
)

ax_a.legend(
    frameon=False,
    fontsize=8
)


# ============================================================
# Panel B
# Boundary-conditioned probabilities
# ============================================================

(
    P_A_only,
    P_R_only,
    P_syn
) = boundary_probabilities(
    D_smooth,
    phi_weak,
    eta_weak
)


ax_b.plot(
    D_smooth,
    P_A_only,
    linewidth=2,
    label=r"$P_{\mathrm{Unq}(A)\mathrm{\ only}|\partial}$"
)

ax_b.plot(
    D_smooth,
    P_R_only,
    linewidth=2,
    label=r"$P_{\mathrm{Rnd}\mathrm{\ only}|\partial}$"
)

ax_b.plot(
    D_smooth,
    P_syn,
    linewidth=2,
    label=r"$P_{\mathrm{Syn}|\partial}$"
)


ax_b.set_xlim(
    0,
    1
)

ax_b.set_ylim(
    bottom=0
)

ax_b.set_xlabel(
    r"Overlap degree, $D_{ov}$"
)

ax_b.set_ylabel(
    "Boundary-conditioned\nprobability",
    labelpad=5
)

ax_b.legend(
    frameon=False,
    fontsize=8
)


# ============================================================
# Panel C
# Simulation vs effective-contact fit
# ============================================================

# Use the same discrete color cycle as the original Figure S11.
colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()[
    "color"
]


for i, p_s in enumerate(
    prob_list
):

    df = simulation_data[
        p_s
    ]


    fit_row = fit_df[
        np.isclose(
            fit_df["p_s"],
            p_s
        )
    ].iloc[0]


    phi = float(
        fit_row["phi"]
    )

    eta = float(
        fit_row["eta"]
    )


    color = colors[
        i % len(colors)
    ]


    # --------------------------------------------------------
    # Simulation
    # --------------------------------------------------------

    ax_c.plot(
        df["D_ov"],
        df["syn_ratio_mean"],
        marker="o",
        markersize=3,
        linewidth=1.2,
        color=color
    )


    # --------------------------------------------------------
    # Effective-contact fit
    # --------------------------------------------------------

    fit_curve = (
        strict_synergy_probability(
            D_smooth,
            phi,
            eta
        )
    )


    ax_c.plot(
        D_smooth,
        fit_curve,
        linestyle="--",
        linewidth=1.3,
        color=color
    )


# ============================================================
# Panel C axis
# ============================================================

ax_c.set_xlim(
    0,
    1
)

ax_c.set_ylim(
    bottom=0
)

ax_c.set_xlabel(
    r"Overlap degree, $D_{ov}$"
)

ax_c.set_ylabel(
    "Synergy probability\nat the boundary",
    labelpad=5
)


# ============================================================
# Simulation / fit style legend
# ============================================================

style_handles = [

    Line2D(
        [0],
        [0],
        color="tab:blue",
        marker="o",
        markersize=4,
        linewidth=1.2,
        label="Simulation"
    ),

    Line2D(
        [0],
        [0],
        color="tab:blue",
        linestyle="--",
        linewidth=1.3,
        label="Effective-contact fit"
    )
]


style_legend = ax_c.legend(
    handles=style_handles,
    loc="upper left",
    frameon=False,
    fontsize=8
)

ax_c.add_artist(
    style_legend
)


# ============================================================
# Rewiring-probability legend
# Same discrete-color style as original Figure S11
# ============================================================

p_handles = []


for i, p_s in enumerate(
    prob_list
):

    color = colors[
        i % len(colors)
    ]

    p_handles.append(

        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            label=rf"$p={p_s:.1e}$"
        )

    )


ax_c.legend(
    handles=p_handles,
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.25
    ),
    ncol=5,
    frameon=False,
    fontsize=7,
    columnspacing=1.0,
    handlelength=2.0
)


# ============================================================
# Panel labels
# ============================================================

ax_a.text(
    -0.16,
    1.04,
    "A",
    transform=ax_a.transAxes,
    fontsize=13,
    fontweight="bold"
)

ax_b.text(
    -0.16,
    1.04,
    "B",
    transform=ax_b.transAxes,
    fontsize=13,
    fontweight="bold"
)

ax_c.text(
    -0.07,
    1.02,
    "C",
    transform=ax_c.transAxes,
    fontsize=13,
    fontweight="bold"
)


# ============================================================
# Style
# ============================================================

for ax in [
    ax_a,
    ax_b,
    ax_c
]:

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


# ============================================================
# Layout
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.97,
    bottom=0.19,
    top=0.94
)


# ============================================================
# Save
# ============================================================

pdf_path = (
    figure_dir
    / "Fig.S11.pdf"
)



fig.savefig(
    pdf_path,
    bbox_inches="tight"
)



plt.close(
    fig
)


print(
    f"Saved: {pdf_path}"
)


print(
    f"Saved: {parameter_path}"
)


print(
    "\nFitted parameters:"
)

print(
    fit_df.to_string(
        index=False
    )
)