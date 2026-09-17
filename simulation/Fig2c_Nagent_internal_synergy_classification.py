import numpy as np
import pandas as pd
import networkx as nx
from multiprocessing import Pool, cpu_count
import os

from src import subnetwork as sn
from src import integnetwork as ig


"""
Simulation of internal synergy for multi-agent knowledge integration (Fig. 2C).

Two simulation modes are available:

    single_pw:
        Calculate synergy as a function of D_ov
        for a fixed within-subnetwork wiring probability p_w.

    matrix:
        Calculate synergy over the full D_ov x p_w parameter space.
"""


"""multi processing"""

def task_Nagent(args):

    network_type, p_w, sub_sizes, ov, N_iteration = args
    N_agents = len(sub_sizes)

    print(
        f"Calculation start | "
        f"Network : {network_type} | "
        f"Agents : {N_agents} | "
        f"D_ov : {ov/sub_sizes[0]:.3f} | "
        f"Pw : {p_w:.3f}"
    )

    
    syn_result = []

    for _ in range(N_iteration):
        # ---------- background network ----------
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
    
        elif network_type == "small_world":
    
            rewiring_probability = 10**(-4)
    
            G = nx.watts_strogatz_graph(
                10**5,
                20,
                rewiring_probability
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
    
        else:
    
            raise ValueError(
                f"Unknown network type: {network_type}"
            )
    
        while True:

            base_node = np.random.choice(
                list(G.nodes())
            )

            sub_nodes, sub_networks = sn.generate_subnetworks(
                sub_sizes,
                G,
                base_node,
                ov,
                p_w
            )

            if all(
                len(sub_nodes[i]) == sub_sizes[i]
                for i in range(N_agents)
            ):
                break

        integ_network = ig.generate_integrated_network(
            G,
            *sub_networks
        )

        integ_pair_lengths = dict(
            nx.all_pairs_shortest_path_length(
                integ_network
            )
        )

        synergy = 0

        # Calculate node pairs one by one
        for source in integ_pair_lengths:

            for target, integ_val in integ_pair_lengths[source].items():

                # Delete duplicated node pairs
                if source >= target:
                    continue

                containing_networks = []

                for net in sub_networks:

                    if source in net and target in net:
                        containing_networks.append(net)

                if len(containing_networks) == 0:
                    continue

                sub_lengths = []

                for net in containing_networks:

                    path_len = nx.shortest_path_length(
                        net,
                        source,
                        target
                    )

                    sub_lengths.append(path_len)

                min_val = min(sub_lengths)

                if min_val > integ_val:
                    synergy += 1

        syn_result.append(synergy)

    syn_mean = np.mean(syn_result)

    syn_err = (
        np.std(syn_result)
        / np.sqrt(len(syn_result))
    )

    print(
        f"Calculation end | "
        f"Network : {network_type} | "
        f"Agents : {N_agents} | "
        f"D_ov : {ov/sub_sizes[0]:.3f} | "
        f"Pw : {p_w:.3f}"
    )

    return syn_mean, syn_err


def generalized_internal_synergy_simulation():

    # ==========================================================
    # Background network
    # ==========================================================

    network_type = "random"
    # network_type = "scale_free"
    # network_type = "modular"
    # network_type = "small_world"


    # ==========================================================
    # Simulation mode
    # ==========================================================

    simulation_mode = "single_pw"
    # simulation_mode = "matrix"


    # ==========================================================
    # Simulation parameters
    # ==========================================================

    N_agents = 5
    subnetwork_size = 100

    # Automatically generate subnetwork sizes
    sub_sizes = [subnetwork_size] * N_agents

    N_iteration = 2500

    ov_size = 50

    ov_node = np.round(
        np.linspace(
            1,
            subnetwork_size,
            ov_size
        )
    ).astype(int)


    # ==========================================================
    # Network label
    # ==========================================================

    network_label = {
        "random": "ER",
        "scale_free": "BA",
        "modular": "SBM",
        "small_world": "WS"
    }[network_type]


    # ==========================================================
    # Mode 1: fixed p_w
    # ==========================================================

    if simulation_mode == "single_pw":

        prob_wire = 0.5

        task_args = [
            (
                network_type,
                prob_wire,
                sub_sizes,
                ov,
                N_iteration
            )
            for ov in ov_node
        ]

        with Pool(processes=cpu_count()) as pool:

            results = pool.map(
                task_Nagent,
                task_args
            )

        syn_res_mean = np.array(
            [r[0] for r in results]
        )

        syn_res_err = np.array(
            [r[1] for r in results]
        )


        # ---------- DataFrame ----------

        syn_df_mean = pd.DataFrame({
            "ov": ov_node,
            "D_ov": ov_node / subnetwork_size,
            "synergy_mean": syn_res_mean
        })

        syn_df_err = pd.DataFrame({
            "ov": ov_node,
            "D_ov": ov_node / subnetwork_size,
            "synergy_sem": syn_res_err
        })


        # ---------- output ----------

        output_dir = (
            "./results/internal_synergy/"
            f"multi_agent/{network_type}"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        syn_df_mean.to_csv(
            os.path.join(
                output_dir,
                f"{network_label}_synergy_mean_"
                f"{N_agents}agent_pw_{prob_wire:.3f}.csv"
            ),
            index=False,
            encoding="utf-8"
        )

        syn_df_err.to_csv(
            os.path.join(
                output_dir,
                f"{network_label}_synergy_sem_"
                f"{N_agents}agent_pw_{prob_wire:.3f}.csv"
            ),
            index=False,
            encoding="utf-8"
        )


    # ==========================================================
    # Mode 2: D_ov × p_w matrix
    # ==========================================================

    elif simulation_mode == "matrix":

        pw_size = 50

        prob_wire = np.linspace(
            0,
            1,
            pw_size
        )

        syn_mean_matrix = []
        syn_err_matrix = []


        for ov in ov_node:

            task_args = [
                (
                    network_type,
                    p_w,
                    sub_sizes,
                    ov,
                    N_iteration
                )
                for p_w in prob_wire
            ]

            with Pool(processes=cpu_count()) as pool:

                results = pool.map(
                    task_Nagent,
                    task_args
                )


            syn_mean_matrix.append(
                [r[0] for r in results]
            )

            syn_err_matrix.append(
                [r[1] for r in results]
            )


        # ---------- DataFrame ----------

        columns = [
            f"{p_w:.3f}"
            for p_w in prob_wire
        ]

        index = [
            f"{ov / subnetwork_size:.3f}"
            for ov in ov_node
        ]


        syn_mean_df = pd.DataFrame(
            syn_mean_matrix,
            index=index,
            columns=columns
        )

        syn_err_df = pd.DataFrame(
            syn_err_matrix,
            index=index,
            columns=columns
        )

        syn_mean_df.index.name = "D_ov"
        syn_err_df.index.name = "D_ov"


        # ---------- output ----------

        output_dir = (
            "./results/internal_synergy/"
            f"multi_agent/{network_type}"
        )

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        syn_mean_df.to_csv(
            os.path.join(
                output_dir,
                f"{network_label}_synergy_matrix_mean_"
                f"{N_agents}agent.csv"
            ),
            encoding="utf-8"
        )

        syn_err_df.to_csv(
            os.path.join(
                output_dir,
                f"{network_label}_synergy_matrix_sem_"
                f"{N_agents}agent.csv"
            ),
            encoding="utf-8"
        )


    else:

        raise ValueError(
            "simulation_mode must be "
            "'single_pw' or 'matrix'"
        )


if __name__ == "__main__":
    generalized_internal_synergy_simulation()