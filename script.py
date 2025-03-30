from datetime import datetime
import numpy as np
from scipy.spatial import distance
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib import cm
from optimized import OptimizedLeachSimulation
from copy import deepcopy


def initialize_nodes(num_nodes, area_size, initial_energy) -> List[Dict]:
    # Initialize nodes with random positions and initial energy
    nodes = [
        {
            "id": i,
            "x": np.random.uniform(0, area_size),
            "y": np.random.uniform(0, area_size),
            "energy": initial_energy,
            "is_cluster_head": False,
            "cluster": None,
            "rounds_as_ch": 0,
        }
        for i in range(num_nodes)
    ]
    x_coords = [node["x"] for node in nodes]
    y_coords = [node["y"] for node in nodes]
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, c="blue", label="Nodes", s=50)
    plt.title("2D Plane of Nodes")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.grid(True)
    plt.legend()
    plt.savefig("node_2d.png")
    return nodes


class LeachSimulation:
    def __init__(self, num_nodes, area_size, initial_energy, p, rounds, nodes):
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.initial_energy = initial_energy
        self.p = p
        self.rounds = rounds
        self.base_station = (area_size / 2, area_size / 2)

        # Energy parameters (in Joules)
        self.E_elec = 50e-9  # Energy for running radio electronics
        self.E_amp = 100e-12  # Energy for amplifier
        self.E_da = 5e-9  # Energy for data aggregation
        self.packet_size = 2000  # Size of data packets in bits

        self.nodes = nodes
        self.round_stats = []

    def select_cluster_heads(self, round_number: int):
        # Reset previous cluster heads
        for node in self.nodes:
            node["is_cluster_head"] = False
            node["cluster"] = None

        # Select new cluster heads
        for node in self.nodes:
            if node["energy"] > 0:  # Only alive nodes can become cluster heads
                threshold = self.p / (1 - self.p * (round_number % (1 / self.p)))
                if np.random.random() < threshold:
                    node["is_cluster_head"] = True
                    node["rounds_as_ch"] += 1

    def assign_clusters(self):
        ch_nodes = [ch for ch in self.nodes if ch["is_cluster_head"]]

        if not ch_nodes:  # If no cluster heads, skip assignment
            return

        for node in self.nodes:
            if not node["is_cluster_head"] and node["energy"] > 0:
                distances = [
                    distance.euclidean((node["x"], node["y"]), (ch["x"], ch["y"]))
                    for ch in ch_nodes
                ]
                nearest_ch = ch_nodes[np.argmin(distances)]
                node["cluster"] = nearest_ch["id"]

    def calculate_energy_dissipation(self, sender, receiver, distance: float) -> float:
        return (
            self.E_elec * self.packet_size + self.E_amp * self.packet_size * distance**2
        )

    def transmit_data(self):
        # Simulate data transmission and energy dissipation
        # First, normal nodes transmit to cluster heads
        for node in self.nodes:
            if (
                not node["is_cluster_head"]
                and node["energy"] > 0
                and node["cluster"] is not None
            ):
                ch = next(ch for ch in self.nodes if ch["id"] == node["cluster"])
                dist = distance.euclidean((node["x"], node["y"]), (ch["x"], ch["y"]))

                energy_tx = self.calculate_energy_dissipation(node, ch, dist)

                if node["energy"] >= energy_tx:
                    node["energy"] -= energy_tx
                    # Cluster head receives and aggregates data
                    ch["energy"] = max(
                        0,
                        ch["energy"]
                        - (
                            self.E_elec * self.packet_size
                            + self.E_da * self.packet_size
                        ),
                    )
                else:
                    node["energy"] = 0

        # Then, cluster heads transmit to base station
        for node in self.nodes:
            if node["is_cluster_head"] and node["energy"] > 0:
                dist = distance.euclidean((node["x"], node["y"]), self.base_station)
                energy_tx = self.calculate_energy_dissipation(
                    node, self.base_station, dist
                )

                if node["energy"] >= energy_tx:
                    node["energy"] -= energy_tx

    def collect_round_statistics(self, round_number: int):
        alive_nodes = sum(1 for node in self.nodes if node["energy"] > 0)
        total_energy = sum(node["energy"] for node in self.nodes)
        num_ch = sum(1 for node in self.nodes if node["is_cluster_head"])

        stats = {
            "round": round_number,
            "alive_nodes": alive_nodes,
            "total_energy": total_energy,
            "avg_energy": total_energy / self.num_nodes if alive_nodes > 0 else 0,
            "num_cluster_heads": num_ch,
        }
        self.round_stats.append(stats)

    def run_simulation(self):
        print("Starting LEACH Protocol Simulation...")

        for round_num in range(self.rounds):
            print(f"\nRound {round_num + 1}/{self.rounds}")

            self.select_cluster_heads(round_num)
            self.assign_clusters()

            self.transmit_data()

            self.collect_round_statistics(round_num)

            if sum(1 for node in self.nodes if node["energy"] > 0) == 0:
                print("\nAll nodes have depleted their energy. Simulation ended.")
                break

        self.plot_statistics()
        return self.round_stats

    def plot_statistics(self):
        stats_df = pd.DataFrame(self.round_stats)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(stats_df["round"], stats_df["alive_nodes"], "b-", label="Alive Nodes")
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Number of Alive Nodes")
        ax1.set_title("Network Lifetime")
        ax1.legend()
        ax1.grid(True)

        ax2.plot(
            stats_df["round"], stats_df["avg_energy"], "r-", label="Average Energy"
        )
        ax2.set_xlabel("Round")
        ax2.set_ylabel("Average Energy (J)")
        ax2.set_title("Energy Consumption")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig("leach_original_simulation_results.png")
        # plt.show()


def plot_comparison_statistics(round_stats1, round_stats2):
    stats_df1 = pd.DataFrame(round_stats1)
    stats_df2 = pd.DataFrame(round_stats2)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot Alive Nodes comparison
    ax1.plot(
        stats_df1["round"], stats_df1["alive_nodes"], "b-", label="LEACH Alive Nodes"
    )
    ax1.plot(
        stats_df2["round"],
        stats_df2["alive_nodes"],
        "g-",
        label="Optimized LEACH Alive Nodes",
    )
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Number of Alive Nodes")
    ax1.set_title("Network Lifetime Comparison")
    ax1.legend()
    ax1.grid(True)

    # Plot Average Energy comparison
    ax2.plot(
        stats_df1["round"], stats_df1["avg_energy"], "r-", label="LEACH Average Energy"
    )
    ax2.plot(
        stats_df2["round"],
        stats_df2["avg_energy"],
        "m-",
        label="Optimized LEACH Average Energy",
    )
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Average Energy (J)")
    ax2.set_title("Energy Consumption Comparison")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("leach_comparison_results.png")


initalNodes = initialize_nodes(50, 100, 0.005)
nodes_copy1 = deepcopy(initalNodes)
nodes_copy2 = deepcopy(initalNodes)


round_stats1 = LeachSimulation(
    num_nodes=50,
    area_size=100,
    initial_energy=0.005,
    p=0.1,
    rounds=100,
    nodes=nodes_copy1,
).run_simulation()

round_stats2 = OptimizedLeachSimulation(
    num_nodes=50,
    area_size=100,
    initial_energy=0.005,
    p=0.1,
    rounds=100,
    nodes=nodes_copy2,
).run_simulation()

plot_comparison_statistics(round_stats1, round_stats2)
