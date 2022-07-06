#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx

if __name__ == "__main__":
    # Nodes
    # 0
    # 1, 2, 3
    # 4, 5, 6, 7, 8, 9, ...
    # ...

    levels = 3
    branching_factor = 2

    G = nx.Graph()

    node_index = 1

    all_level_nodes = []

    for level in range(0, levels):
        single_level_nodes = [
            i for i in range(node_index, node_index + branching_factor**level)
        ]

        node_index += branching_factor**level
        all_level_nodes.append(single_level_nodes)

    for i, level in enumerate(range(1, levels)):
        single_level_nodes = all_level_nodes[i]

        start_i = 0

        for node in single_level_nodes:
            for target_node in all_level_nodes[i + 1][
                start_i : start_i + branching_factor
            ]:
                G.add_edge(node, target_node)
                print(node, target_node)
            start_i += branching_factor

    # explicitly set positions

    #         1.5           0  2   2**2 / 2**0 = 4 / 1  = 4
    #   0.5         2.5     1  1   2**1 / 2**1          = 1
    # 0     1     2     3   2  0   2**0 / 2**2 = 1 / 4  = 0.25

    # (bf**2 - 1) / bf -> 3/2
    # (bf**1 - 1) / bf -> 1/2

    pos = {}
    for level in range(0, levels):
        single_level_nodes = all_level_nodes[level]

        y = -level

        start = (branching_factor ** (levels - 1 - level) - 1) / branching_factor
        offset = branching_factor ** (levels - 1 - level)

        for x_index, node in enumerate(single_level_nodes):
            x = start + x_index * offset
            pos[node] = (x, y)

    total_n_nodes = sum(branching_factor**levels for levels in range(0, levels))

    node_colors = ["C0" for n in range(1, total_n_nodes + 1)]

    for index in [1, 2, 5]:
        node_colors[index - 1] = "C2"
    for index in [6, 7]:
        node_colors[index - 1] = tuple([0.3] * 3)

    options = {}
    nx.draw_networkx(
        G,
        pos=pos,
        font_size=36,
        node_size=3000,
        node_color=node_colors,
        edgecolors="black",
        linewidths=5,
        width=5,
    )

    # Set margins for the axes so that nodes aren't clipped
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")

    plt.savefig(Path("~/tmp/iter_opt_graph.png").expanduser())

    plt.close()
