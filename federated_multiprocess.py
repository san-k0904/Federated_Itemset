"""
federated_multiprocess.py

Parent process (CentralServer) spawns child processes (ClientNodes).
Children mine locally and send encrypted results back to the parent via Pipe.

Now updated to use the UCI 'chess.txt' dataset for federated itemset mining.

Run:
    python federated_multiprocess.py
"""

import multiprocessing as mp
import time
import random
from collections import defaultdict
import os

# Import your project modules
from output_saver import OutputSaver
from federated_nodes import ClientNode, CentralServer
from encryption_manager import EncryptionManager


# ============================================================
# DATA LOADING FUNCTIONS
# ============================================================

def load_transactions(file_path="chess.txt"):
    """
    Load dataset from chess.txt, where each line is a transaction
    with space-separated integers (e.g. '1 3 5 7 9 ...').
    Returns a list of lists of strings.
    """
    dataset = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            if items:
                dataset.append(items)
    return dataset


def split_for_clients(dataset, num_clients):
    """Split the dataset evenly into num_clients subsets."""
    n = len(dataset)
    chunk_size = (n + num_clients - 1) // num_clients
    splits = [dataset[i*chunk_size:(i+1)*chunk_size] for i in range(num_clients)]
    return splits


def generate_sample_data(num_clients=5, file_path="chess.txt", limit_rows=10):
    """
    Load and split the chess dataset into client-specific subsets.
    For quick testing, limit total rows to 'limit_rows'.
    """
    print(f"[DataLoader] Loading dataset from {file_path} ...")
    dataset = load_transactions(file_path)

    if limit_rows is not None and limit_rows < len(dataset):
        dataset = dataset[:limit_rows]
        print(f"[DataLoader] ⚠️ Using only first {limit_rows} transactions for test run.")

    print(f"[DataLoader] Loaded {len(dataset)} transactions.")
    splits = split_for_clients(dataset, num_clients)
    for i, s in enumerate(splits, start=1):
        print(f"[DataLoader] Client {i} assigned {len(s)} transactions.")
    return splits


# ============================================================
# CHILD PROCESS ENTRYPOINT
# ============================================================

def client_process(child_conn, client_id, data, algorithm, min_support, encryption_key, use_output_saver):
    """
    Runs in child process:
    - Creates a ClientNode with local data
    - Sets encryption key
    - Mines itemsets
    - Encrypts result and sends encrypted bytes through child_conn
    """
    import time
    try:
        start_total = time.time()
        random.seed(client_id + int(time.time()) % 1000)

        t0 = time.time()
        output_saver = OutputSaver() if use_output_saver else None
        client = ClientNode(client_id=client_id, data=data, algorithm=algorithm,
                            min_support=min_support, output_saver=output_saver)
        t1 = time.time()
        print(f"[Client {client_id}] ✅ Init done in {t1 - t0:.2f}s")

        client.set_encryption_key(encryption_key)

        t2 = time.time()
        print(f"[Client {client_id}] 🚀 Starting mining with {len(data)} txns using {algorithm} ...")
        encrypted_blob = client.process_and_send()
        t3 = time.time()
        print(f"[Client {client_id}] 🔐 Mining + Encryption done in {t3 - t2:.2f}s")

        child_conn.send(encrypted_blob)
        child_conn.close()
        print(f"[Client {client_id}] ✅ Sent results in {time.time() - start_total:.2f}s")

    except Exception as e:
        try:
            child_conn.send(("error", client_id, str(e)))
            child_conn.close()
        except:
            pass
        print(f"[Client {client_id}] ❌ Exception: {e}")


# ============================================================
# PARENT ORCHESTRATION
# ============================================================

def run_multiprocess_federated(num_clients=5, use_output_saver=False, dataset_path="chess.txt"):
    """
    Parent process creates a CentralServer, spawns child processes for each client,
    collects encrypted blobs, decrypts and merges results.
    """
    output_saver = OutputSaver() if use_output_saver else None

    # Load limited data for testing
    client_datasets = generate_sample_data(num_clients=num_clients, file_path=dataset_path, limit_rows=50)

    # Assign algorithms and min_support per client
    algorithms = ['fp-growth', 'fp-growth', 'fp-growth']
    client_specs = [
        (i + 1, client_datasets[i], algorithms[i], 3)
        for i in range(num_clients)
    ]

    server = CentralServer(output_saver=output_saver)

    print("\n[Parent] Initializing clients and distributing encryption key...")
    encryption_key = server.encryption_manager.get_key()

    processes = []
    for (client_id, data, algorithm, min_support) in client_specs:
        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(
            target=client_process,
            args=(child_conn, client_id, data, algorithm, min_support, encryption_key, use_output_saver),
            name=f"ClientProcess-{client_id}"
        )
        processes.append((p, parent_conn))

    print(f"[Parent] Launching {len(processes)} client processes...")
    for p, _ in processes:
        p.start()
        print(f"[Parent] Started {p.name} (PID={p.pid})")

    server.client_count = len(processes)
    print(f"[Parent] Total clients running: {server.client_count}\n")

    encrypted_results = []
    for p, parent_conn in processes:
        try:
            msg = parent_conn.recv()
            if isinstance(msg, tuple) and msg[0] == "error":
                _, client_id_err, err_str = msg
                print(f"[Parent] ❌ Error from Client {client_id_err}: {err_str}")
            else:
                if isinstance(msg, tuple) and len(msg) == 3 and isinstance(msg[0], (bytes, bytearray)):
                    envelope_bytes, signature, pubkey_bytes = msg
                    print(f"[Parent] ✅ Received encrypted blob (size={len(envelope_bytes)} bytes)")
                    encrypted_results.append((envelope_bytes, signature, pubkey_bytes))
                else:
                    print(f"[Parent] ⚠️ Unexpected message format from child: {msg}")
        except EOFError:
            print(f"[Parent] ⚠️ No data from {p.name} (EOF).")
        finally:
            parent_conn.close()

    for p, _ in processes:
        p.join(timeout=10.0)
        if p.is_alive():
            print(f"[Parent] ⚠️ {p.name} did not exit in time; terminating.")
            p.terminate()
        else:
            print(f"[Parent] ✅ {p.name} finished.")

    server.results = encrypted_results
    final_itemsets, client_details = server.decrypt_and_merge()

    print(f"\n[Parent] ✅ Total clients participated: {server.client_count}")
    server.display_results(final_itemsets, client_details, server.client_count)

    return final_itemsets, server, client_specs


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # For Windows compatibility (use 'fork' on Linux)
    mp.set_start_method('spawn', force=True)

    print("\n=== Federated Itemset Mining: Multiprocess Simulation (Chess Dataset) ===\n")
    start = time.time()
    final_itemsets, server, clients = run_multiprocess_federated(
        num_clients=3,
        use_output_saver=False,
        dataset_path="chess.txt"
    )
    end = time.time()
    print(f"\nTotal (wall-clock) execution time: {end - start:.2f} seconds\n")
