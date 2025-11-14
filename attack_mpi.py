"""
attack_mpi.py

MPI-based attack simulation for Federated Itemset Mining.

• Rank 0 = CentralServer
• Ranks 1..N = ClientNodes
• A random client is selected to be tampered (ciphertext corrupted)
• Server verifies signatures and aborts aggregation if tampering is detected

Run:
    mpirun -np 6 python attack_mpi.py
"""

from mpi4py import MPI
import time
import random
import pickle
from collections import defaultdict

# --- Project imports (must be in same project) ---
from encryption_manager import EncryptionManager
from signature_manager import generate_keypair, public_key_bytes, sign, load_public_key, verify
from mining_algorithm import AprioriMiner, FPGrowthMiner

# =========================================================
#                Dataset (same as your other scripts)
# =========================================================
def generate_sample_data():
    data1 = [['bread', 'milk', 'eggs'],
             ['bread', 'butter', 'milk'],
             ['milk', 'eggs', 'cheese'],
             ['bread', 'milk', 'butter', 'eggs'],
             ['bread', 'milk'],
             ['eggs', 'butter', 'cheese'],
             ['bread', 'milk', 'eggs', 'butter'],
             ['milk', 'cheese', 'yogurt'],
             ['bread', 'eggs'],
             ['milk', 'butter', 'eggs']]

    data2 = [['bread', 'butter', 'sugar'],
             ['bread', 'flour', 'eggs'],
             ['flour', 'sugar', 'butter', 'eggs'],
             ['bread', 'butter'],
             ['flour', 'eggs', 'milk'],
             ['bread', 'flour', 'sugar'],
             ['butter', 'eggs', 'sugar'],
             ['bread', 'flour', 'butter', 'eggs']]

    data3 = [['coffee', 'milk', 'sugar'],
             ['tea', 'milk', 'sugar'],
             ['coffee', 'eggs', 'bread'],
             ['yogurt', 'milk'],
             ['coffee', 'milk', 'bread', 'butter'],
             ['tea', 'sugar', 'milk'],
             ['coffee', 'sugar'],
             ['eggs', 'bread', 'butter', 'milk'],
             ['yogurt', 'milk', 'sugar'],
             ['coffee', 'milk', 'sugar', 'bread']]

    data4 = [['rice', 'onion', 'tomato'],
             ['pasta', 'tomato', 'onion', 'cheese'],
             ['rice', 'tomato'],
             ['pasta', 'cheese', 'butter'],
             ['onion', 'tomato', 'rice'],
             ['pasta', 'tomato', 'cheese'],
             ['rice', 'onion'],
             ['pasta', 'onion', 'tomato', 'cheese']]

    data5 = [['bread', 'milk', 'coffee'],
             ['rice', 'pasta', 'tomato'],
             ['eggs', 'cheese', 'yogurt'],
             ['bread', 'butter', 'milk', 'eggs'],
             ['coffee', 'sugar', 'milk'],
             ['pasta', 'cheese', 'tomato'],
             ['bread', 'milk', 'butter'],
             ['rice', 'onion', 'tomato'],
             ['eggs', 'milk', 'bread'],
             ['coffee', 'milk', 'sugar', 'bread']]

    return [data1, data2, data3, data4, data5]


# =========================================================
#                Client logic (ranks > 0)
# =========================================================
def client_node(rank, encryption_key, tampered_client_id):
    """
    Each MPI rank > 0 runs as a federated client.
    If this client's rank == tampered_client_id, it will corrupt its envelope bytes
    before sending to simulate tampering in transit.
    """
    datasets = generate_sample_data()
    data = datasets[rank - 1]  # rank 1 -> dataset[0], etc.
    algorithm = 'apriori' if rank <= 3 else 'fp-growth'
    min_support = 3

    print(f"[Client {rank}] Starting mining using {algorithm}...")

    # Mining
    if algorithm == 'apriori':
        miner = AprioriMiner(min_support=min_support)
    else:
        miner = FPGrowthMiner(min_support=min_support)

    itemsets = miner.mine(data)

    result = {
        'client_id': rank,
        'itemsets': {tuple(sorted(k)): v for k, v in itemsets.items()},
        'algorithm': algorithm,
        'num_transactions': len(data)
    }

    # Encryption & signing
    enc_mgr = EncryptionManager(key=encryption_key)
    sign_priv, sign_pub = generate_keypair()
    sign_pub_bytes = public_key_bytes(sign_pub)

    plaintext = pickle.dumps(result)
    envelope = enc_mgr.encrypt(plaintext)
    envelope_bytes = pickle.dumps(envelope)
    signature = sign(envelope_bytes, sign_priv)

    # If selected as the tampered client, corrupt envelope_bytes (simulate MITM)
    if tampered_client_id is not None and rank == tampered_client_id:
        print(f"[Client {rank}] >>> Simulating tampering: corrupting ciphertext before send.")
        # flip a middle byte (guard against very short messages)
        b = bytearray(envelope_bytes)
        if len(b) > 20:
            b[len(b) // 2] ^= 0xAA
        else:
            b[0] ^= 0xAA
        envelope_bytes = bytes(b)

    # Send the (envelope_bytes, signature, pubkey) tuple to server (rank 0)
    comm = MPI.COMM_WORLD
    comm.send((envelope_bytes, signature, sign_pub_bytes), dest=0, tag=rank)
    print(f"[Client {rank}] Sent signed encrypted data to server.")


# =========================================================
#                Server logic (rank 0)
# =========================================================
def central_server(size, tampered_client_id):
    """
    Rank 0: central server.
    Broadcasts symmetric key and selected tampered client id,
    receives all client payloads, verifies signatures, decrypts valid ones.
    If tampering is detected (signature or decryption failure), abort aggregation and alert.
    """
    comm = MPI.COMM_WORLD
    enc_mgr = EncryptionManager()

    # Generate symmetric key and broadcast (object bcast)
    encryption_key = enc_mgr.get_key()
    # Note: we'll broadcast encryption_key and tampered_client_id together in main()
    print(f"[Server] Distributed symmetric AES key to {size - 1} clients.")

    # Receive payloads
    results = []
    for i in range(1, size):
        try:
            msg = comm.recv(source=i, tag=i)
            results.append((i, msg))  # store tuple (client_rank, payload)
            print(f"[Server] Received payload from Client {i}")
        except Exception as e:
            print(f"[Server] Error receiving from client {i}: {e}")

    print("\n[Server] Starting signature verification and decryption phase...")
    decrypted_results = []

    # Process each received payload in the order we collected them
    for client_rank, payload in results:
        try:
            envelope_bytes, signature, pubkey_bytes = payload
        except Exception as e:
            print(f"[Server] Invalid payload format from client {client_rank}: {e}")
            # treat as compromised
            print(f"[Server] !!! ALERT: Client {client_rank} COMPROMISED — aborting aggregation.")
            return {}, []

        # Verify signature first
        try:
            client_pub = load_public_key(pubkey_bytes)
            verify(envelope_bytes, signature, client_pub)
            print(f"[Server] ✓ Signature verification passed for Client {client_rank}")
        except Exception as e:
            # Signature verification failed -> tampering or wrong key
            print(f"[Server] ⚠️ Signature verification FAILED for Client {client_rank}: {e}")
            print(f"[Server] !!! ALERT: Client {client_rank} COMPROMISED — Aborting aggregation.")
            return {}, []

        # If signature OK, attempt to decrypt
        try:
            envelope = pickle.loads(envelope_bytes)
            plaintext = enc_mgr.decrypt(envelope)
            data = pickle.loads(plaintext)
            decrypted_results.append(data)
            print(f"[Server] ✓ Decrypted data from Client {data['client_id']} "
                  f"(itemsets: {len(data['itemsets'])}, txns: {data['num_transactions']})")
        except Exception as e:
            # Decryption failed (AES-GCM invalid tag or parsing) -> tampered
            print(f"[Server] ⚠️ Decryption FAILED for Client {client_rank}: {e}")
            print(f"[Server] !!! ALERT: Client {client_rank} COMPROMISED — Aborting aggregation.")
            return {}, []

    # If we reach here, all client payloads were verified & decrypted successfully
    print("\n[Server] All payloads verified. Merging itemsets...")
    merged = defaultdict(int)
    for r in decrypted_results:
        for items, count in r['itemsets'].items():
            merged[items] += count

    final = dict(sorted(merged.items(), key=lambda x: (len(x[0]), x[1]), reverse=True))
    print(f"[Server] ✓ Merged {len(final)} unique itemsets from {len(decrypted_results)} clients.\n")

    # Display summary
    print("=" * 70)
    print("FINAL AGGREGATED RESULTS")
    print("=" * 70)
    print(f"Total MPI processes (including server): {size}")
    print(f"Clients Contributed: {len(decrypted_results)}")
    print(f"Unique Itemsets: {len(final)}\n")

    grouped = defaultdict(list)
    for items, count in final.items():
        grouped[len(items)].append((items, count))

    for k in sorted(grouped):
        print(f"\n{k}-Itemsets:")
        for items, count in grouped[k][:10]:
            print(f"  {set(items)} -> {count}")

    print("\n[Server] Federated mining complete.\n")
    return final, decrypted_results


# =========================================================
#                MAIN: setup, broadcast, run
# =========================================================
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only one server (rank 0) and at least one client required
    if size < 2:
        if rank == 0:
            print("Error: need at least 2 MPI processes (1 server + >=1 client).")
        return

    # Rank 0 picks a random client to tamper (uniform)
    if rank == 0:
        random.seed(int(time.time()) % 100000)
        tampered_client_id = random.randint(1, size - 1)
        # prepare encryption key
        enc_mgr_tmp = EncryptionManager()
        encryption_key = enc_mgr_tmp.get_key()
        # we'll broadcast both the encryption key and tampered client id as a tuple
        to_bcast = (encryption_key, tampered_client_id)
        print(f"\n[Server] (SIM) Selected Client {tampered_client_id} for tampering simulation.\n")
    else:
        to_bcast = None

    # Broadcast (encryption_key, tampered_client_id) to all ranks
    to_bcast = comm.bcast(to_bcast, root=0)
    encryption_key, tampered_client_id = to_bcast

    if rank == 0:
        # server uses its own EncryptionManager instance (not the temporary one)
        final_itemsets, client_details = central_server(size, tampered_client_id)
        if final_itemsets == {}:
            print("\n[Server] ❌ Aggregation aborted due to compromised client data.\n")
        else:
            print("\n[Server] ✅ Aggregation completed successfully.\n")
    else:
        # client receives encryption_key and tampered id; then runs
        client_node(rank, encryption_key, tampered_client_id)


if __name__ == "__main__":
    main()
