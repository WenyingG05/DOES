"""
DOES Model Main Script

"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os

from DOES_model import opinion_init, run_simulation


def generate_network(n_nodes=100, m=3):
    """Generate a Barabási–Albert (BA) network."""
    return nx.barabasi_albert_graph(n_nodes, m, seed=42)


def initialize_model(graph):
    """Initialize opinions and emotions on the network."""
    opinion_graph = opinion_init(graph.copy(), mode=1)
    emotion_graph = opinion_init(graph.copy(), mode=2)
    return opinion_graph, emotion_graph


def plot_simulation(opinion_results, emotion_results, filename):
    """Plot opinion evolution and emotional order parameter R."""
    def kuramoto_order(emo_values):
        nodes_len = len(emo_values)
        sum_exp = np.sum(np.exp(1j * np.array(list(emo_values.values()))))
        return np.abs(sum_exp) / nodes_len

    time_steps = opinion_results['time_steps']
    opi_values = opinion_results['opi_values']
    emo_values = emotion_results['emo_values']

    # Compute R over time
    r_values = []
    for t in range(len(time_steps)):
        emo_at_t = {node: emo_values[node][t] for node in emo_values}
        r_values.append(kuramoto_order(emo_at_t))

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))

    # Plot opinions
    for node in opi_values.keys():
        axs[0].plot(time_steps, opi_values[node])
    axs[0].set_title('(a)', fontsize=16)
    axs[0].set_xlabel('Time Step', fontsize=14)
    axs[0].set_ylabel('Opinion', fontsize=14)
    axs[0].set_ylim(0, 1)

    # Plot emotional order parameter
    axs[1].plot(time_steps, r_values, linewidth=2)
    axs[1].set_title('(b)', fontsize=16)
    axs[1].set_xlabel('Time Step', fontsize=14)
    axs[1].set_ylabel('$R$', fontsize=14)
    axs[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


def run_example():
    """Run a simple DOES example on a BA network."""
    graph = generate_network(n_nodes=100, m=3)
    opinion_graph, emotion_graph = initialize_model(graph)

    opinion_results, emotion_results = run_simulation(
        opinion_graph, emotion_graph,
        T=20.0, dt=0.01,
        lamb=0.5, alpha=0.5, mu=0.5, beta=0.5
    )

    plot_simulation(opinion_results, emotion_results, './DOES_example.png')


if __name__ == '__main__':
    run_example()