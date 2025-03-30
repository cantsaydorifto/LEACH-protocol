import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.cluster import KMeans
from typing import List, Dict
import matplotlib.pyplot as plt


class OptimizedLeachSimulation:
    def __init__(
        self,
        num_nodes: int,
        area_size: int,
        initial_energy: float,
        p: float,
        rounds: int,
        nodes,
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

        self.nodes = nodes
        self.round_stats = []

    def create_clusters(self):
        # Use K-Means clustering to create clusters
        X = np.array([[node["x"], node["y"]] for node in self.nodes])
        kmeans = KMeans(n_clusters=int(self.p * self.num_nodes), random_state=42)
        kmeans.fit(X)

        # Assign clusters to nodes
        for node, cluster_id in zip(self.nodes, kmeans.labels_):
            node["cluster"] = cluster_id

    def select_cluster_heads(self, round_number: int):
        # Reset previous cluster heads
        for node in self.nodes:
            node["is_cluster_head"] = False

        # Select new cluster heads based on energy/distance to base station
        for cluster_id in set(node["cluster"] for node in self.nodes):
            cluster_nodes = [
                node for node in self.nodes if node["cluster"] == cluster_id
            ]
            if cluster_nodes:
                best_node = max(
                    cluster_nodes,
                    key=lambda x: x["energy"]
                    / distance.euclidean((x["x"], x["y"]), self.base_station),
                )
                best_node["is_cluster_head"] = True
                best_node["rounds_as_ch"] += 1

    def transmit_data(self):
        # Simulate data transmission and energy dissipation
        # First, normal nodes transmit to cluster heads
        for node in self.nodes:
            if (
                not node["is_cluster_head"]
                and node["energy"] > 0
                and node["cluster"] is not None
            ):
                cluster_head = next(
                    (
                        ch
                        for ch in self.nodes
                        if ch["id"] == node["cluster"] and ch["is_cluster_head"]
                    ),
                    None,
                )
                if cluster_head:
                    dist = distance.euclidean(
                        (node["x"], node["y"]), (cluster_head["x"], cluster_head["y"])
                    )

                    energy_tx = self.calculate_energy_dissipation(
                        node, cluster_head, dist
                    )

                    # Check if node has enough energy to transmit
                    if node["energy"] >= energy_tx:
                        node["energy"] -= energy_tx
                        # Cluster head receives and aggregates data
                        energy_rx = (
                            self.E_elec * self.packet_size
                            + self.E_da * self.packet_size
                        )
                        if cluster_head["energy"] >= energy_rx:
                            cluster_head["energy"] -= energy_rx
                        else:
                            cluster_head["energy"] = (
                                0  # Set to zero if not enough energy to receive
                            )
                    else:
                        node["energy"] = (
                            0  # Set to zero if not enough energy to transmit
                        )

        # Then, cluster heads transmit to base station
        for node in self.nodes:
            if node["is_cluster_head"] and node["energy"] > 0:
                dist = distance.euclidean((node["x"], node["y"]), self.base_station)
                energy_tx = self.calculate_energy_dissipation(
                    node, self.base_station, dist
                )

                # Check if cluster head has enough energy to transmit to base station
                if node["energy"] >= energy_tx:
                    node["energy"] -= energy_tx
                else:
                    node["energy"] = 0  # Set to zero if not enough energy to transmit

    def calculate_energy_dissipation(self, sender, receiver, distance: float) -> float:
        return (
            self.E_elec * self.packet_size + self.E_amp * self.packet_size * distance**2
        )

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
        print("Starting Optimized LEACH Protocol Simulation...")
        self.create_clusters()

        for round_num in range(self.rounds):
            print(f"\nRound {round_num + 1}/{self.rounds}")

            self.select_cluster_heads(round_num)
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
        plt.savefig("leach_optimized_simulation_results.png")
        # plt.show()
