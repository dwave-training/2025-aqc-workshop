# Copyright 2025 D-Wave
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
import argparse
from time import perf_counter

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

import minorminer.subgraph
from minorminer.utils.parallel_embeddings import find_multiple_embeddings
from dwave.preprocessing.composites import SpinReversalTransformComposite
import dwave.system
from dwave.samplers import TreeDecompositionSolver
from dwave_networkx import draw_parallel_embeddings


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
    use_srt=False,
):
    """Experiment with customizable parameters; see main function descriptions."""
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

    sampler = qpu
    # Apply a spin-reversal transform
    if use_srt == True:
        sampler = SpinReversalTransformComposite(sampler)
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
            sampler,
            embeddings=[{k: (v,) for k, v in emb.items()} for emb in embeddings],
        )
    else:
        embedding = minorminer.subgraph.find_subgraph(
            source_graph, target_graph, **embedder_kwargs
        )
        sampler = dwave.system.FixedEmbeddingComposite(
            sampler, embedding={k: (v,) for k, v in embedding.items()}
        )
        embeddings = [embedding]
    draw_parallel_embeddings(
        G=qpu.to_networkx_graph(),
        embeddings=embeddings,
        S=nx.from_edgelist(J.keys()),
        one_to_iterable=False,
    )  # There are many different dwave_networkx visualization tools suiting other purposes.
    print("Embedding time (seconds):", perf_counter() - time0)
    fname = f"{model}_embeddings_{solver}.png"
    plt.savefig(fname)
    plt.show()
    print(
        f"An image of the embedding has been saved to {fname}, see local directory, or panel (left) if using codespaces"
    )
    for model_idx in range(num_models):
        h, J, EGS, label = get_model(model=model, model_idx=model_idx)  # For topology
        stats = []
        print(
            f"{len(tas_nanosec)}",
            " sequential QPU programmings (varying in annealing time)",
            f" {model} with {label}",
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
    fname = f"{model}_data_{solver}.png"
    plt.savefig(fname)
    plt.show()
    print(
        f"A scaling plot has been saved to {fname}, see local directory, or panel (left) if using codespaces."
    )
    if model == "Kibble-Zurek":
        print(
            "The density of defects (also called kink density in 1D"
            " scales as a power-law 1/2. Demonstrated by the approximately"
            " straight-line dependence in log-log plots."
        )
    else:
        print(
            "The rate of diabatic transitions from the ground state"
            " to the first excited state decays exponentially, in proportion"
            " the the inverse gap-squared. Demonstrated by the approximately"
            " straight-line dependence in log-linear plots."
        )


if __name__ == "__main__":

    tas_nanosec = [5, 7, 10, 14, 20]  # Larger values can show thermal deviation

    parser = argparse.ArgumentParser(
        description="AQC2025 workshop (ocean-sdk for physics)"
    )

    parser.add_argument(
        "--model",
        default="Kibble-Zurek",
        type=str,
        help="model: either 'Landau-Zener' or 'Kibble-Zurek'",
    )
    parser.add_argument(
        "--solver_name",
        type=str,
        help="option to specify QPU solver",
    )
    parser.add_argument(
        "--parallelize_embedding",
        action="store_true",
        help="parallelize, where possible, the embedding",
    )

    parser.add_argument(
        "--use_srt",
        action="store_true",
        help="use spin reversal transform",
    )
    args = parser.parse_args()

    main(
        model=args.model,
        solver=args.solver_name,
        tas_nanosec=tas_nanosec,
        use_srt=args.use_srt,
        parallelize_embedding=args.parallelize_embedding,
    )
