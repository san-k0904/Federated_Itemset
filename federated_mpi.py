"""
federated_mpi.py
================

Federated Itemset Mining using MPI (mpi4py)
-------------------------------------------
• Rank 0 acts as Central Server
• Ranks 1..N act as Client Nodes
• Each client mines locally, encrypts + signs results
• Server verifies, decrypts, and merges verified results

Run locally:
    mpirun -np 6 python federated_mpi.py
(= 1 server + 5 clients)
"""

from mpi4py import MPI
import time
import pickle
from collections import defaultdict

from encryption_manager import EncryptionManager
from signature_manager import generate_keypair, public_key_bytes, sign, load_public_key, verify
from mining_algorithm import AprioriMiner, FPGrowthMiner
from dataset_loader import load_datasets  



def client_node(rank, encryption_key):
    """Each MPI rank > 0 runs as a federated client node."""
    comm = MPI.COMM_WORLD
    datasets = load_datasets()
    data = datasets[rank - 1]

    algorithms = ['apriori', 'fp-growth', 'fp-growth', 'apriori', 'fp-growth']
    algorithm = algorithms[rank - 1]
    min_support = 20

    print(f"[Client {rank}] ✅ Initialized with {len(data)} transactions using {algorithm}...")

    start_mine = time.time()
    if algorithm == 'apriori':
        miner = AprioriMiner(min_support=min_support)
    else:
        miner = FPGrowthMiner(min_support=min_support)

    print(f"[Client {rank}] 🚀 Starting mining ...")
    itemsets = miner.mine(data)
    end_mine = time.time()

    # --- Prepare Result ---
    result = {
        'client_id': rank,
        'itemsets': {tuple(sorted(k)): v for k, v in itemsets.items()},
        'algorithm': algorithm,
        'num_transactions': len(data),
        'mine_time': round(end_mine - start_mine, 2)
    }

    # --- Encryption + Signing ---
    enc_mgr = EncryptionManager(key=encryption_key)
    priv_key, pub_key = generate_keypair()
    pub_bytes = public_key_bytes(pub_key)

    plaintext = pickle.dumps(result)
    envelope = enc_mgr.encrypt(plaintext)
    envelope_bytes = pickle.dumps(envelope)
    signature = sign(envelope_bytes, priv_key)

    # --- Send to Server ---
    comm.send((envelope_bytes, signature, pub_bytes), dest=0, tag=rank)
    print(f"[Client {rank}] 🔐 Sent encrypted results to server (itemsets={len(itemsets)})")



def central_server(size):
    """MPI rank 0 acts as the central server."""
    comm = MPI.COMM_WORLD
    enc_mgr = EncryptionManager()
    encryption_key = enc_mgr.get_key()

    # Broadcast key to clients
    comm.bcast(encryption_key, root=0)
    print(f"[Server] 🧩 Distributed AES encryption key to {size - 1} clients.\n")

    results = []
    for i in range(1, size):
        msg = comm.recv(source=i, tag=i)
        results.append(msg)
        print(f"[Server] ✅ Received payload from Client {i}")

    print("\n==================== SERVER VERIFICATION ====================")
    decrypted_results = []
    for idx, (enc_bytes, sig, pub_bytes) in enumerate(results, start=1):
        try:
            pub_key = load_public_key(pub_bytes)
            verify(enc_bytes, sig, pub_key)

            envelope = pickle.loads(enc_bytes)
            plaintext = enc_mgr.decrypt(envelope)
            data = pickle.loads(plaintext)

            decrypted_results.append(data)
            print(f"[Server] ✓ Verified + Decrypted Client {idx} "
                  f"({len(data['itemsets'])} itemsets, algo={data['algorithm']})")

        except Exception as e:
            print(f"[Server] ❌ Client {idx} verification failed: {e}")
            continue

    # Merge
    print("\n[Server] 🔄 Aggregating itemsets...")
    merged = defaultdict(int)
    for r in decrypted_results:
        for k, v in r['itemsets'].items():
            merged[k] += v

    print(f"[Server] ✅ Aggregated {len(merged)} unique itemsets "
          f"from {len(decrypted_results)} verified clients.\n")

    # Summary
    print("=" * 70)
    print("FINAL FEDERATED RESULTS")
    print("=" * 70)
    print(f"Total Clients Participated: {len(decrypted_results)} / {size - 1}")
    print(f"Unique Itemsets: {len(merged)}")

    grouped = defaultdict(list)
    for items, count in merged.items():
        grouped[len(items)].append((items, count))

    for k in sorted(grouped):
        print(f"\n{k}-Itemsets (top 5):")
        for items, count in grouped[k][:5]:
            print(f"  {set(items)} -> {count}")

    print("\n[Server] 🏁 Federated Mining Complete.")



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        start = time.time()
        central_server(size)
        end = time.time()
        print(f"\n⏱️ Total execution time: {round(end - start, 2)}s")
    else:
        encryption_key = comm.bcast(None, root=0)
        client_node(rank, encryption_key)


if __name__ == "__main__":
    main()
