from datetime import datetime
import os
import uuid
import json
import numpy as np
from scipy.spatial import distance
import pandas as pd
from typing import List, Dict
import matplotlib.pyplot as plt
from matplotlib import cm


class LeachSimulation:
    def __init__(
        self,
        num_nodes: int = 20,
        area_size: int = 100,
        initial_energy: float = 0.5,
        p: float = 0.1,  # Probability of becoming cluster head
        rounds: int = 20,
    ):
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

        self.nodes = self._initialize_nodes()
        self.round_stats = []

    def _initialize_nodes(self) -> List[Dict]:
        # Initialize nodes with random positions and initial energy
        return [
            {
                "id": i,
                "x": np.random.uniform(0, self.area_size),
                "y": np.random.uniform(0, self.area_size),
                "energy": self.initial_energy,
                "is_cluster_head": False,
                "cluster": None,
                "rounds_as_ch": 0,
            }
            for i in range(self.num_nodes)
        ]

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
                    ch["energy"] -= (
                        self.E_elec * self.packet_size + self.E_da * self.packet_size
                    )

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
        folder_name = self._create_simulation_folder()

        self._save_initial_stats(folder_name)

        for round_num in range(self.rounds):
            print(f"\nRound {round_num + 1}/{self.rounds}")

            self.select_cluster_heads(round_num)
            self.assign_clusters()

            self.transmit_data()

            self.collect_round_statistics(round_num)

            # Every 10 rounds
            if (round_num + 1) % 10 == 0 or round_num == self.rounds - 1:
                self.plot_cluster_and_energy_distribution(round_num, folder_name)

            self.print_round_summary(round_num, folder_name)

            if sum(1 for node in self.nodes if node["energy"] > 0) == 0:
                print("\nAll nodes have depleted their energy. Simulation ended.")
                break
        self.plot_statistics(folder_name)

    def print_round_summary(self, round_num: int, folder_name: str):
        stats = self.round_stats[-1]
        print(f"\nRound {round_num + 1} Summary:")
        print(f"Alive Nodes: {stats['alive_nodes']}/{self.num_nodes}")
        print(f"Number of Cluster Heads: {stats['num_cluster_heads']}")
        print(f"Average Energy: {stats['avg_energy']:.6f} J")

        print("\nDetailed Node Status:")
        for node in self.nodes:
            status = "CH" if node["is_cluster_head"] else "Member"
            state = "Alive" if node["energy"] > 0 else "Dead"
            print(
                f"Node {node['id']}: {status} | {state} | Energy: {node['energy']:.6f} J | "
                + f"Cluster: {node['cluster'] if node['cluster'] is not None else 'None'}"
            )

        if (round_num + 1) % 10 == 0 or round_num == self.rounds - 1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.set_title("Node Status")
            ax1.bar(
                ["Alive", "Dead"],
                [stats["alive_nodes"], self.num_nodes - stats["alive_nodes"]],
                color=["green", "red"],
            )
            ax1.set_ylabel("Number of Nodes")

            node_ids = [node["id"] for node in self.nodes]
            energies = [node["energy"] for node in self.nodes]
            colors = [
                "blue" if node["is_cluster_head"] else "orange" for node in self.nodes
            ]
            ax2.bar(node_ids, energies, color=colors)
            ax2.set_title("Node Energy Levels")
            ax2.set_xlabel("Node ID")
            ax2.set_ylabel("Energy (J)")

            plt.tight_layout()
            plt.savefig(os.path.join(folder_name, f"round_{round_num + 1}_summary.png"))
            plt.close(fig)  # Close the figure to save memory

    def plot_statistics(self, folder_name: str):
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
        plt.savefig(os.path.join(folder_name, "leach_simulation_results.png"))
        # plt.show()

    def plot_cluster_and_energy_distribution(self, round_number: int, folder_name: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        ax1.set_title(f"Node Clustering - Round {round_number}")
        ax1.set_xlim(0, self.area_size)
        ax1.set_ylim(0, self.area_size)

        for node in self.nodes:
            color = "blue" if node["is_cluster_head"] else "green"
            marker = "s" if node["is_cluster_head"] else "o"
            ax1.scatter(
                node["x"],
                node["y"],
                color=color,
                marker=marker,
                s=50 if node["is_cluster_head"] else 20,
            )
            if node["cluster"] is not None and not node["is_cluster_head"]:
                ch_node = next(n for n in self.nodes if n["id"] == node["cluster"])
                ax1.plot(
                    [node["x"], ch_node["x"]],
                    [node["y"], ch_node["y"]],
                    "gray",
                    linestyle="--",
                    linewidth=0.5,
                )
        ax1.set_xlabel("X Position")
        ax1.set_ylabel("Y Position")
        ax1.legend(["Cluster Head", "Normal Node"], loc="upper right")

        ax2.set_title(f"Energy Depletion Heatmap - Round {round_number}")
        x_positions = [node["x"] for node in self.nodes]
        y_positions = [node["y"] for node in self.nodes]
        energies = [node["energy"] for node in self.nodes]
        sc = ax2.scatter(x_positions, y_positions, c=energies, cmap=cm.hot, s=40)
        ax2.set_xlim(0, self.area_size)
        ax2.set_ylim(0, self.area_size)
        ax2.set_xlabel("X Position")
        ax2.set_ylabel("Y Position")
        plt.colorbar(sc, ax=ax2, label="Remaining Energy (J)")

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                folder_name, f"cluster_energy_distribution_round_{round_number}.png"
            )
        )
        plt.close(fig)

    def _create_simulation_folder(self):
        sim_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"simulation_{sim_id}_{timestamp}"
        os.makedirs(folder_name, exist_ok=True)
        return folder_name

    def _save_initial_stats(self, folder_name: str):
        stats = {
            "num_nodes": self.num_nodes,
            "area_size": self.area_size,
            "initial_energy": self.initial_energy,
            "probability_of_cluster_head": self.p,
            "rounds": self.rounds,
            "base_station": self.base_station,
            "E_elec": self.E_elec,
            "E_amp": self.E_amp,
            "E_da": self.E_da,
            "packet_size": self.packet_size,
            "nodes": self.nodes,
        }
        with open(os.path.join(folder_name, "initial_stats.json"), "w") as f:
            json.dump(stats, f, indent=4)


sim = LeachSimulation(
    num_nodes=50, area_size=100, initial_energy=0.005, p=0.1, rounds=30
).run_simulation()
