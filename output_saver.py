"""
Output Saving Module for Federated Itemset Mining
Handles saving outputs from clients and server to files
"""

import os
import json
from datetime import datetime
from collections import defaultdict


class OutputSaver:
    """Handles saving outputs from clients and server to files"""

    def __init__(self, output_dir="federated_outputs"):
        """
        Initialize output saver
        Args:
            output_dir: Directory to save all outputs
        """
        self.output_dir = output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.timestamp}")

        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "clients"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "server"), exist_ok=True)

        print(f"[OutputSaver] Created output directory: {self.run_dir}")

    def save_client_output(self, client_id, data, itemsets, algorithm, min_support):
        """
        Save client's local data and mining results
        Args:
            client_id: Client identifier
            data: Local transaction data
            itemsets: Mined itemsets with frequencies
            algorithm: Algorithm used
            min_support: Minimum support threshold
        """
        client_file = os.path.join(self.run_dir, "clients", f"client_{client_id}_output.json")

        output = {
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
            "algorithm": algorithm,
            "min_support": min_support,
            "num_transactions": len(data),
            "transactions": data,
            "mined_itemsets": {str(k): v for k, v in itemsets.items()},
            "total_itemsets": len(itemsets)
        }

        with open(client_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[OutputSaver] Saved Client {client_id} output to {client_file}")
        return client_file

    def save_server_output(self, final_itemsets, client_details, total_clients, total_transactions):
        """
        Save server's aggregated results
        Args:
            final_itemsets: Merged itemsets from all clients
            client_details: Details from each client
            total_clients: Total number of clients
            total_transactions: Total transactions across all clients
        """
        server_file = os.path.join(self.run_dir, "server", "server_aggregated_results.json")

        # Group itemsets by size
        itemsets_by_size = defaultdict(list)
        for itemset, count in final_itemsets.items():
            itemsets_by_size[len(itemset)].append({
                "itemset": list(itemset),
                "frequency": count
            })

        # Sort each group by frequency
        for size in itemsets_by_size:
            itemsets_by_size[size].sort(key=lambda x: x['frequency'], reverse=True)

        output = {
            "timestamp": datetime.now().isoformat(),
            "total_clients": total_clients,
            "total_transactions": total_transactions,
            "unique_itemsets": len(final_itemsets),
            "client_summary": [
                {
                    "client_id": c['client_id'],
                    "algorithm": c['algorithm'],
                    "num_transactions": c['num_transactions'],
                    "num_itemsets": len(c['itemsets'])
                }
                for c in client_details
            ],
            "itemsets_by_size": {str(k): v for k, v in itemsets_by_size.items()},
            "all_itemsets": {str(k): v for k, v in final_itemsets.items()}
        }

        with open(server_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"[OutputSaver] Saved server output to {server_file}")

        # Also save a human-readable summary
        summary_file = os.path.join(self.run_dir, "server", "summary.txt")
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("FEDERATED ITEMSET MINING - SUMMARY\n")
            f.write("="*70 + "\n\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Clients: {total_clients}\n")
            f.write(f"Total Transactions: {total_transactions}\n")
            f.write(f"Unique Itemsets Found: {len(final_itemsets)}\n\n")

            f.write("-"*70 + "\n")
            f.write("Client Summary:\n")
            f.write("-"*70 + "\n")
            for c in client_details:
                f.write(f"Client {c['client_id']}: {c['algorithm'].upper()} algorithm, ")
                f.write(f"{c['num_transactions']} transactions, {len(c['itemsets'])} itemsets\n")

            f.write("\n" + "-"*70 + "\n")
            f.write("Top Itemsets by Size:\n")
            f.write("-"*70 + "\n")

            for size in sorted(itemsets_by_size.keys()):
                f.write(f"\n{size}-Itemsets (Total: {len(itemsets_by_size[size])}):\n")
                for item in itemsets_by_size[size][:10]:  # Top 10
                    f.write(f"  {set(item['itemset'])} -> Frequency: {item['frequency']}\n")

        print(f"[OutputSaver] Saved human-readable summary to {summary_file}")
        return server_file, summary_file
