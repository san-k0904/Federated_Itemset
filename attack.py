"""
attack.py

Simulates a tampering attack on one of the client payloads during
federated itemset mining. The CentralServer detects the tampered
data through signature verification and aborts aggregation safely.

Run:
    python attack.py
"""

import multiprocessing as mp
import time
import random
from collections import defaultdict

from output_saver import OutputSaver
from federated_nodes import ClientNode, CentralServer
from encryption_manager import EncryptionManager

# --- sample data ---
def generate_sample_data():
    data1 = [['bread', 'milk', 'eggs'], ['bread', 'butter', 'milk'], ['milk', 'eggs', 'cheese'],
             ['bread', 'milk', 'butter', 'eggs'], ['bread', 'milk'], ['eggs', 'butter', 'cheese'],
             ['bread', 'milk', 'eggs', 'butter'], ['milk', 'cheese', 'yogurt'], ['bread', 'eggs'],
             ['milk', 'butter', 'eggs']]
    data2 = [['bread', 'butter', 'sugar'], ['bread', 'flour', 'eggs'], ['flour', 'sugar', 'butter', 'eggs'],
             ['bread', 'butter'], ['flour', 'eggs', 'milk'], ['bread', 'flour', 'sugar'],
             ['butter', 'eggs', 'sugar'], ['bread', 'flour', 'butter', 'eggs']]
    data3 = [['coffee', 'milk', 'sugar'], ['tea', 'milk', 'sugar'], ['coffee', 'eggs', 'bread'], ['yogurt', 'milk'],
             ['coffee', 'milk', 'bread', 'butter'], ['tea', 'sugar', 'milk'], ['coffee', 'sugar'],
             ['eggs', 'bread', 'butter', 'milk'], ['yogurt', 'milk', 'sugar'], ['coffee', 'milk', 'sugar', 'bread']]
    data4 = [['rice', 'onion', 'tomato'], ['pasta', 'tomato', 'onion', 'cheese'], ['rice', 'tomato'],
             ['pasta', 'cheese', 'butter'], ['onion', 'tomato', 'rice'], ['pasta', 'tomato', 'cheese'],
             ['rice', 'onion'], ['pasta', 'onion', 'tomato', 'cheese']]
    data5 = [['bread', 'milk', 'coffee'], ['rice', 'pasta', 'tomato'], ['eggs', 'cheese', 'yogurt'],
             ['bread', 'butter', 'milk', 'eggs'], ['coffee', 'sugar', 'milk'], ['pasta', 'cheese', 'tomato'],
             ['bread', 'milk', 'butter'], ['rice', 'onion', 'tomato'], ['eggs', 'milk', 'bread'],
             ['coffee', 'milk', 'sugar', 'bread']]
    return [data1, data2, data3, data4, data5]


# --- Child process entrypoint ---
def client_process(child_conn, client_id, data, algorithm, min_support, encryption_key, use_output_saver):
    try:
        random.seed(client_id + int(time.time()) % 1000)
        output_saver = OutputSaver() if use_output_saver else None
        client = ClientNode(client_id=client_id, data=data, algorithm=algorithm,
                            min_support=min_support, output_saver=output_saver)
        client.set_encryption_key(encryption_key)
        encrypted_blob = client.process_and_send()
        child_conn.send(encrypted_blob)
        child_conn.close()
    except Exception as e:
        try:
            child_conn.send(("error", client_id, str(e)))
            child_conn.close()
        except:
            pass


# --- Attack simulation---
def run_attack_simulation(num_clients=5, use_output_saver=False):
    output_saver = OutputSaver() if use_output_saver else None
    client_datasets = generate_sample_data()
    client_specs = [
        (1, client_datasets[0], 'apriori', 3),
        (2, client_datasets[1], 'apriori', 3),
        (3, client_datasets[2], 'apriori', 3),  # target client for tampering
        (4, client_datasets[3], 'fp-growth', 3),
        (5, client_datasets[4], 'fp-growth', 3),
    ][:num_clients]

    random.seed(int(time.time()) % 100000) 
    client_ids = [c_id for (c_id, _, _, _) in client_specs]
    tampered_client_id = random.choice(client_ids)
    print(f"[ATTACK] Selected Client {tampered_client_id} for tampering (uniform random).")

    server = CentralServer(output_saver=output_saver)
    encryption_key = server.encryption_manager.get_key()

    print("\n[Parent] Initializing clients and distributing encryption key...")
    processes, parent_conns = [], []

    for (client_id, data, algorithm, min_support) in client_specs:
        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(target=client_process,
                       args=(child_conn, client_id, data, algorithm, min_support, encryption_key, use_output_saver),
                       name=f"ClientProcess-{client_id}")
        processes.append((p, parent_conn))
        parent_conns.append(parent_conn)

    print(f"[Parent] Launching {len(processes)} client processes...")
    for p, _ in processes:
        p.start()
        print(f"[Parent] Started child process {p.name} (PID={p.pid})")

    server.client_count = len(processes)
    print(f"[Parent] Total client processes running: {server.client_count}\n")

    encrypted_results = []

    for idx, (p, parent_conn) in enumerate(processes, start=1):
        try:
            msg = parent_conn.recv()
            if isinstance(msg, tuple) and msg[0] == "error":
                _, client_id_err, err_str = msg
                print(f"[Parent] Received error from child {client_id_err}: {err_str}")
            elif isinstance(msg, tuple) and len(msg) == 3:
                envelope_bytes, signature, pubkey_bytes = msg
                print(f"[Parent] Received encrypted+signed blob from Client {idx} (size={len(envelope_bytes)} bytes)")

                if idx == tampered_client_id:
                    print(f"[ATTACK] Simulating tampering on Client {tampered_client_id}'s payload...")
                    tampered = bytearray(envelope_bytes)
                    if len(tampered) > 20:
                        tampered[len(tampered)//2] ^= 0xAA  # flip one byte
                    envelope_bytes = bytes(tampered)
                    print(f"[ATTACK] Payload for Client {tampered_client_id} has been altered in transit.")

                encrypted_results.append((envelope_bytes, signature, pubkey_bytes))
            else:
                print(f"[Parent] Unexpected message format from child: {msg}")
        except EOFError:
            print(f"[Parent] No data received from process {p.name} (EOF).")
        finally:
            parent_conn.close()

    for p, _ in processes:
        p.join(timeout=10.0)
        if p.is_alive():
            print(f"[Parent] Warning: {p.name} did not exit in time; terminating.")
            p.terminate()
        else:
            print(f"[Parent] Child {p.name} finished.")

    server.results = encrypted_results

    print("\n=== ATTACK SIMULATION PHASE ===")
    final_itemsets, client_details = server.decrypt_and_merge()

    if final_itemsets == {}:
        print("\n[Server] ❌ Aggregation aborted due to compromised client data.")
    else:
        print("\n[Server] ✅ Aggregation completed successfully (no compromise detected).")

    print(f"\n[Parent] ✅ Total clients participated: {server.client_count}")
    return final_itemsets, server, client_specs


if __name__ == "__main__":
    mp.set_start_method('fork', force=True)

    print("\n=== Federated Itemset Mining: Attack Simulation ===\n")
    start = time.time()
    final_itemsets, server, clients = run_attack_simulation(num_clients=5, use_output_saver=False)
    end = time.time()
    print(f"\nTotal (wall-clock) execution time: {end - start:.2f} seconds\n")
