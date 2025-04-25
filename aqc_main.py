# Copyright [2025] D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
A repository of code to demonstrate simple Kibble-Zurek and Landau-Zener experiments
compatible with https://doi.org/10.1038/s41586-023-05867-2 (Figure 2C)
and https://doi.org/10.1038/s41567-022-01741-6 (Figure 2A)
"""

import numpy as np
import networkx as nx
import minorminer.subgraph
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
import dwave.system
import matplotlib.pyplot as plt
from time import perf_counter
import pickle
from dwave.samplers import TreeDecompositionSolver
from dwave_networkx import draw_parallel_embeddings, draw_zephyr, draw_chimera


def get_model(model="Landau-Zener", model_idx=0):
    if model == "Kibble-Zurek":
        num_vars = 512
        if model_idx == 0:
            J0 = -1  ## Ferromagnetic, strength 1
            label = "FM, J=-1"
        elif model_idx == 1:
            J0 = 1
            label = "AFM, J=1"
        else:
            J0 = 1 / 2
            label = "AFM, J=1/2"
        h = {x: 0 for x in range(num_vars)}
        J = {(x, (x + 1) % num_vars): J0 for x in range(num_vars)}  # 1D ring
        EGS = -num_vars * abs(J0)
    elif model == "Landau-Zener":
        # TO DO, plot spectrum by brute force with published schedules.
        with open("Andrew220713800Fig2c.pkl", "rb") as f:
            X = pickle.load(f)
        Js = list(X.values())
        J = Js[model_idx]
        if model_idx == 0:
            label = "Large Gap"
        elif model_idx == 1:
            label = "Intermediate Gap"
        else:
            label = "Small Gap"
        nodes = sorted({n for e in J for n in e})
        h = {n: 0 for n in nodes}
        ss = TreeDecompositionSolver().sample_ising(h, J)
        EGS = ss.first.energy
        # print(h, J, EGS)
    return h, J, EGS, label


def main(
    model="Landau-Zener",
    *,
    solver=None,
    tas_nanosec=(5, 7, 10, 14, 20),
    num_models=3,
    parallelize_embedding=False,
):
    """Experiment with customizable parameters"""
    # Source graph connectivity:
    h, J, EGS, label = get_model(model=model, model_idx=0)  # For topology only
    num_vars = len(h)
    source_graph = nx.from_edgelist(J.keys())

    # Connect to solver and infer target connectivity:
    qpu = dwave.system.DWaveSampler(solver=solver)
    qpu_kwargs = dict(
        num_reads=100, answer_mode="raw", fast_anneal=True, auto_scale=False
    )
    target_graph = qpu.to_networkx_graph()

    # Generation of embeddings:
    time0 = perf_counter()
    embedder_kwargs = {"timeout": 10}
    if parallelize_embedding:
        # This embedding is currently limited to https://github.com/jackraymond/dwave-system/tree/feature/parallel_embeddings
        # Intention is to make a pull request
        max_num_emb = max(1, len(qpu.nodelist) // num_vars)  # Aim for half-full:
        embeddings = find_multiple_embeddings(
            source_graph,
            target_graph,
            max_num_emb=max_num_emb,
            timeout=30,
            embedder_kwargs=embedder_kwargs,
        )
        sampler = dwave.system.ParallelEmbeddingComposite(
            qpu, embeddings=[{k: (v,) for k, v in emb.items()} for emb in embeddings]
        )
        draw_parallel_embeddings(
            G=qpu.to_networkx_graph(),
            embeddings=embeddings,
            S=nx.from_edgelist(J.keys()),
            one_to_iterable=False,
        )
    else:
        embedding = minorminer.subgraph.find_subgraph(
            source_graph, target_graph, **embedder_kwargs
        )
        linear_biases = {
            embedding[v]: h for v, h in h.items()
        }  # 1:1 transformation to qubits
        quadratic_biases = {
            (embedding[e[0]], embedding[e[1]]): J for e, J in J.items()
        }  # 1:1 transformation to qubits
        # NB More generally can use draw_zephyr_embeddings or draw_pegasus_embeddings
        if qpu.properties["topology"]["type"] == "zephyr":
            draw_zephyr(
                target_graph,
                linear_biases=linear_biases,
                quadratic_biases=quadratic_biases,
                node_size = 0
            )
        elif qpu.properties["topology"]["type"] == "pegasus":
            draw_pegasus(
                target_graph,
                linear_biases=linear_biases,
                quadratic_biases=quadratic_biases,
                node_size = 0
            )
        else:
            print("Expecting either pegasus or zephyr topology")
        sampler = dwave.system.FixedEmbeddingComposite(
            qpu, embedding={k: (v,) for k, v in embedding.items()}
        )
    print("Embedding time:", perf_counter() - time0)
    plt.savefig(f"{model}_embeddings_{solver}.png")
    plt.show()

    for model_idx in range(num_models):
        h, J, EGS, label = get_model(model=model, model_idx=model_idx)  # For topology
        stats = []
        print(
            f"{len(tas_nanosec)}",
            " sequential QPU programmings (varying in annealing time)",
        )
        for ta_nanosec in tas_nanosec:
            qpu_kwargs["annealing_time"] = ta_nanosec / 1000
            ss = sampler.sample_ising(h, J, **qpu_kwargs)
            if model == "Landau-Zener":
                stats.append(
                    1 - np.mean(ss.record.energy <= EGS + 1e-8)
                )  # 1 - Prob(Ground state)
            else:
                stats.append(
                    (np.mean(ss.record.energy) - EGS) / (2 * abs(EGS))
                )  # Kink density
        plt.figure(model)
        plt.plot(tas_nanosec, stats, marker="x", label=label)

    # Plot formatting:
    if model == "Landau-Zener":
        plt.ylabel("1 - Ground state probability")
        plt.title("Landau-Zener: Exponential decay as function of the gap")
    else:
        plt.ylabel("Kink density ((<H>/J + N) / 2)")
        plt.xscale("log")
        plt.title("Kibble-Zurek: power law decay as function of critical exponent")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Annealing time (ns)")
    plt.savefig(f"{model}_data_{solver}.png")
    plt.show()


if __name__ == "__main__":
    # To do, command line options:
    solver = "Advantage2_prototype2.6"
    model = "Landau-Zener"
    model = "Kibble-Zurek"
    tas_nanosec = [5, 7, 10, 14, 20]  # Larger values can show thermal deviations
    main(solver=solver, model=model, tas_nanosec=tas_nanosec)
