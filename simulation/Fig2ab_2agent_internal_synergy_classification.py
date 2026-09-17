import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count
from pathlib import Path

from src import subnetwork as sn
from src import integnetwork as ig

"""
Simulation of internal synergy for two-agent knowledge integration. (Fig. 2A and 2B)

For each combination of within-subnetwork wiring probability (p_w)
and overlap degree (D_ov), two subnetworks are generated from a
background network and subsequently integrated.

Node pairs are classified as:
    - Synergy
    - Unique to agent A
    - Unique to agent B
    - Redundancy

The mean and SEM of each quantity are saved as matrices whose
rows correspond to D_ov and columns correspond to p_w.
"""

"""multi processing"""

def task_2agent(args):
    """
    Run repeated two-agent simulations for one (p_w, D_ov) condition.

    Parameters
    ----------
    args : tuple
        p_w : float
            Within-subnetwork wiring probability.
        sub1_size : int
            Size of the first subnetwork.
        sub2_size : int
            Size of the second subnetwork.
        ov : int
            Number of nodes shared by the two subnetworks.
        N_iteration : int
            Number of independent simulation realizations.

    Returns
    -------
    tuple
        Mean and SEM of synergy, unique-A, unique-B, and redundancy.
    """

    p_w, sub1_size, sub2_size, ov, N_iteration, network_type = args

    print(
        f"Calculation start | "
        f"Overlapped : {ov/sub1_size} | "
        f"Pw : {round(p_w, 2)}"
    )

    syn_result = []
    unq_a_result = []
    unq_b_result = []
    red_result = []

    for num in range(N_iteration):

        # ------------- background network ------------- 
        if network_type == "random":
            G = nx.fast_gnp_random_graph(
                10**5,
                20 / (10**5 - 1)
            )
    
        elif network_type == "scale_free":
            G = nx.barabasi_albert_graph(
                10**5,
                10
            )
    
        elif network_type == "modular":
            block_sizes = [10**4] * 10
    
            p_in = 0.00182
            p_out = 0.000022
    
            probabilities = [
                [
                    p_in if i == j else p_out
                    for j in range(len(block_sizes))
                ]
                for i in range(len(block_sizes))
            ]
    
            G = nx.stochastic_block_model(
                block_sizes,
                probabilities
            )
    
        elif network_type == "small_world":
            G = nx.watts_strogatz_graph(
                10**5,
                20,
                10**(-4)
            )
    
        else:
            raise ValueError(
                f"Unknown network type : {network_type}"
            )

        while True:

            base_node = np.random.choice(
                list(G.nodes())
            )

            sub_nodes, sub_networks = sn.generate_subnetworks(
                [sub1_size, sub2_size],
                G,
                base_node,
                ov,
                p_w
            )

            if (
                len(sub_nodes[0]) == sub1_size
                and len(sub_nodes[1]) == sub2_size
            ):
                break

        integ_network = ig.generate_integrated_network(
            G,
            sub_networks[0],
            sub_networks[1]
        )

        integ_pair_lengths = dict(
            nx.all_pairs_shortest_path_length(
                integ_network
            )
        )

        synergy = 0
        unq_a = 0
        unq_b = 0
        red = 0

        # Calculate node pairs one by one
        for source in integ_pair_lengths:

            for target, integ_val in integ_pair_lengths[source].items():

                # Delete duplicated node pairs
                if source >= target:
                    continue

                in_sub1 = (
                    source in sub_networks[0]
                    and target in sub_networks[0]
                )

                in_sub2 = (
                    source in sub_networks[1]
                    and target in sub_networks[1]
                )

                # Pair contained in both subnetworks
                if in_sub1 and in_sub2:

                    sub1_lengths = nx.shortest_path_length(
                        sub_networks[0],
                        source,
                        target
                    )

                    sub2_lengths = nx.shortest_path_length(
                        sub_networks[1],
                        source,
                        target
                    )

                    min_val = min(
                        sub1_lengths,
                        sub2_lengths
                    )

                    if min_val > integ_val:
                        synergy += 1

                    elif min_val == integ_val:

                        if sub1_lengths < sub2_lengths:
                            unq_a += 1

                        elif sub1_lengths > sub2_lengths:
                            unq_b += 1

                        else:
                            red += 1

                # Pair contained only in subnetwork 1
                elif in_sub1:

                    sub1_lengths = nx.shortest_path_length(
                        sub_networks[0],
                        source,
                        target
                    )

                    if sub1_lengths > integ_val:
                        synergy += 1

                    elif sub1_lengths == integ_val:
                        unq_a += 1

                # Pair contained only in subnetwork 2
                elif in_sub2:

                    sub2_lengths = nx.shortest_path_length(
                        sub_networks[1],
                        source,
                        target
                    )

                    if sub2_lengths > integ_val:
                        synergy += 1

                    elif sub2_lengths == integ_val:
                        unq_b += 1

        syn_result.append(synergy)
        unq_a_result.append(unq_a)
        unq_b_result.append(unq_b)
        red_result.append(red)

    syn_mean = np.mean(syn_result)
    unq_a_mean = np.mean(unq_a_result)
    unq_b_mean = np.mean(unq_b_result)
    red_mean = np.mean(red_result)

    syn_err = np.std(syn_result) / np.sqrt(len(syn_result))
    unq_a_err = np.std(unq_a_result) / np.sqrt(len(unq_a_result))
    unq_b_err = np.std(unq_b_result) / np.sqrt(len(unq_b_result))
    red_err = np.std(red_result) / np.sqrt(len(red_result))

    print(
        f"Calculation end | "
        f"Overlapped : {ov/sub1_size} | "
        f"Pw : {round(p_w, 2)}"
    )

    return (
        syn_mean,
        unq_a_mean,
        unq_b_mean,
        red_mean,
        syn_err,
        unq_a_err,
        unq_b_err,
        red_err
    )


def two_agent_internal_synergy_simulation():
    import os

    """
    Sweep p_w and D_ov and save the resulting two-dimensional matrices.

    Rows correspond to D_ov and columns correspond to p_w.
    """

    network_type = "random"
    # network_type = "scale_free"
    # network_type = "modular"
    # network_type = "small_world"

    sub1_size = 100
    sub2_size = 100

    N_iteration = 2500
    pw_size = 100
    ov_size = 100

    prob_wire = np.linspace(0, 1, pw_size)

    ov_node = np.round(np.linspace(1, sub1_size, ov_size)).astype(int)

    # containers
    syn_mean_matrix = []
    unq_a_mean_matrix = []
    unq_b_mean_matrix = []
    red_mean_matrix = []

    syn_err_matrix = []
    unq_a_err_matrix = []
    unq_b_err_matrix = []
    red_err_matrix = []

    for ov in ov_node:

        task_args = [
            (
                p_w,
                sub1_size,
                sub2_size,
                ov,
                N_iteration,
                network_type
            )
            for p_w in prob_wire
        ]

        with Pool(processes=cpu_count()) as pool:
            results = pool.map(
                task_2agent,
                task_args
            )

        syn_mean_matrix.append(
            [r[0] for r in results]
        )

        unq_a_mean_matrix.append(
            [r[1] for r in results]
        )

        unq_b_mean_matrix.append(
            [r[2] for r in results]
        )

        red_mean_matrix.append(
            [r[3] for r in results]
        )

        syn_err_matrix.append(
            [r[4] for r in results]
        )

        unq_a_err_matrix.append(
            [r[5] for r in results]
        )

        unq_b_err_matrix.append(
            [r[6] for r in results]
        )

        red_err_matrix.append(
            [r[7] for r in results]
        )

    # ---------- DataFrame ----------
    # Output matrices:
    # rows    = overlap degree D_ov = overlap size / subnetwork size
    # columns = within-subnetwork wiring probability p_w
    columns = [
        f"{p_w:.3f}"
        for p_w in prob_wire
    ]

    index = [
        f"{ov / sub1_size:.3f}"
        for ov in ov_node
    ]

    syn_mean_df = pd.DataFrame(
        syn_mean_matrix,
        index=index,
        columns=columns
    )

    unq_a_mean_df = pd.DataFrame(
        unq_a_mean_matrix,
        index=index,
        columns=columns
    )

    unq_b_mean_df = pd.DataFrame(
        unq_b_mean_matrix,
        index=index,
        columns=columns
    )

    red_mean_df = pd.DataFrame(
        red_mean_matrix,
        index=index,
        columns=columns
    )

    syn_err_df = pd.DataFrame(
        syn_err_matrix,
        index=index,
        columns=columns
    )

    unq_a_err_df = pd.DataFrame(
        unq_a_err_matrix,
        index=index,
        columns=columns
    )

    unq_b_err_df = pd.DataFrame(
        unq_b_err_matrix,
        index=index,
        columns=columns
    )

    red_err_df = pd.DataFrame(
        red_err_matrix,
        index=index,
        columns=columns
    )

    # ---------- output ----------
    output_dir = "./results/internal_synergy/two_agent"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    network_label = {
    "random": "ER",
    "scale_free": "BA",
    "modular": "SBM",
    "small_world": "WS"
    }[network_type]

    for name, df in [

        ("synergy_mean", syn_mean_df),
        ("unique_A_mean", unq_a_mean_df),
        ("unique_B_mean", unq_b_mean_df),
        ("redundancy_mean", red_mean_df),

        ("synergy_sem", syn_err_df),
        ("unique_A_sem", unq_a_err_df),
        ("unique_B_sem", unq_b_err_df),
        ("redundancy_sem", red_err_df),

    ]:

        path = os.path.join(
            output_dir,
            f"{network_label}_{name}.csv"
        )

        df.to_csv(
            path,
            encoding="utf-8"
        )


if __name__ == "__main__":
    two_agent_internal_synergy_simulation()