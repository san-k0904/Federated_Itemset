"""
Client and Server Nodes for Federated Itemset Mining
Implements ClientNode and CentralServer classes
"""

import time
import random
from collections import defaultdict
from encryption_manager import EncryptionManager
from mining_algorithm import AprioriMiner, FPGrowthMiner
from encryption_manager import EncryptionManager
from signature_manager import generate_keypair, public_key_bytes, sign, load_public_key, verify
import pickle



class ClientNode:
    """
    Represents an edge device in federated itemset mining

    Each client:
    - Holds local transaction data
    - Runs itemset mining algorithm independently
    - Encrypts results before sending to server
    - Can use different algorithms based on device capability
    """

    def __init__(self, client_id, data, algorithm='apriori', min_support=2, output_saver=None):
        """
        Initialize client node
        Args:
            client_id: Unique identifier for this client
            data: Local transaction data (list of lists)
            algorithm: Mining algorithm to use ('apriori' or 'fp-growth')
            min_support: Minimum support threshold
            output_saver: OutputSaver instance for saving results
        """
        self.client_id = client_id
        self.data = data
        self.algorithm = algorithm
        self.min_support = min_support
        self.encryption_manager = None
        self.output_saver = output_saver
        # create per-client signing keypair (in-memory)
        self.sign_private_key, self.sign_public_key = generate_keypair()
        # store serializable public key bytes so server can register/verify
        self.sign_public_key_bytes = public_key_bytes(self.sign_public_key)

    def set_encryption_key(self, key: bytes):
        """
        Set symmetric encryption key bytes provided by server.
        """
        # EncryptionManager wrapper using provided raw key bytes
        self.encryption_manager = EncryptionManager(key=key)

    def mine_itemsets(self):
        """
        Run itemset mining on local data
        Returns:
            dict: Mined itemsets with frequencies
        """
        print(f"[Client {self.client_id}] Starting itemset mining using {self.algorithm}...")

        # Simulate computation time (varies by algorithm complexity)
        if self.algorithm == 'apriori':
            computation_time = random.uniform(1.5, 3.0)  # 1.5-3 seconds
        else:  # fp-growth (typically faster)
            computation_time = random.uniform(1.0, 2.0)  # 1-2 seconds

        print(f"[Client {self.client_id}] Mining in progress...")
        time.sleep(computation_time)

        if self.algorithm == 'apriori':
            miner = AprioriMiner(min_support=self.min_support)
        else:  # fp-growth
            miner = FPGrowthMiner(min_support=self.min_support)

        itemsets = miner.mine(self.data)

        # Convert frozensets to tuples for serialization
        itemsets_serializable = {tuple(sorted(itemset)): count 
                                for itemset, count in itemsets.items()}

        print(f"[Client {self.client_id}] Found {len(itemsets_serializable)} frequent itemsets "
              f"(computation time: {computation_time:.2f}s)")

        # Save client output
        if self.output_saver:
            self.output_saver.save_client_output(
                self.client_id, 
                self.data, 
                itemsets_serializable, 
                self.algorithm, 
                self.min_support
            )

        return itemsets_serializable

    def process_and_send(self):
        """
        Mine itemsets and return a signed, encrypted envelope as bytes.
        Returns:
            tuple: (envelope_bytes, signature_bytes, public_key_bytes)
                - envelope_bytes: pickled dict {'nonce':..., 'ciphertext':..., 'aad':...}
                - signature_bytes: signature over envelope_bytes created by client's private key
                - public_key_bytes: client's public key bytes (PEM) so server can verify (or server may already have it)
        """
        # Mine itemsets
        itemsets = self.mine_itemsets()

        # Prepare result dict (same as before)
        result = {
            'client_id': self.client_id,
            'itemsets': itemsets,
            'algorithm': self.algorithm,
            'num_transactions': len(self.data)
        }

        # Serialise result to bytes
        plaintext = pickle.dumps(result)

        # Encrypt using AES-GCM (returns envelope dict)
        envelope = self.encryption_manager.encrypt(plaintext, associated_data=None)

        # Serialize envelope to bytes to sign / transmit
        envelope_bytes = pickle.dumps(envelope)

        # Sign envelope bytes with client's signing private key
        signature = sign(envelope_bytes, self.sign_private_key)

        # Optionally simulate network transmission time
        print(f"[Client {self.client_id}] Encrypting & signing results...")
        time.sleep(random.uniform(0.2, 0.5))
        transmission_time = random.uniform(0.5, 1.0)
        print(f"[Client {self.client_id}] Sending to server...")
        time.sleep(transmission_time)

        # Return a tuple containing the envelope bytes, signature and public key bytes
        return envelope_bytes, signature, self.sign_public_key_bytes



class CentralServer:
    """
    Central server for federated itemset mining

    Server responsibilities:
    - Send initialization and encryption keys to clients
    - Collect encrypted results from all clients
    - Decrypt and merge itemsets
    - Generate final aggregated results

    Key advantage: Server only active during initialization and aggregation phases,
    reducing cost and attack surface
    """

    def __init__(self, output_saver=None):
        """Initialize central server"""
        self.encryption_manager = EncryptionManager()
        self.clients = []
        self.results = []
        self.output_saver = output_saver

    def initialize_clients(self, clients):
        """
        Send initialization and encryption key to all clients
        Args:
            clients: List of ClientNode objects
        """
        print("\n" + "="*70)
        print("SERVER: Initializing Federated Itemset Mining")
        print("="*70)

        self.clients = clients
        encryption_key = self.encryption_manager.get_key()

        for client in clients:
            # Simulate network delay for key distribution
            time.sleep(random.uniform(0.1, 0.3))
            client.set_encryption_key(encryption_key)
            print(f"[Server] Client {client.client_id} initialized with encryption key")

        print("[Server] All clients initialized. Beginning mining phase...\n")
        print("[Server] Going offline... Clients mining independently...\n")

        # Simulate server going offline (not actively managing the process)
        time.sleep(1.0)

    def collect_results(self):
        """Collect encrypted results from all clients"""
        print("\n" + "="*70)
        print("SERVER: Coming Online - Collecting Results from Clients")
        print("="*70)

        for client in self.clients:
            encrypted_data = client.process_and_send()
            self.results.append(encrypted_data)
            print(f"[Server] Received encrypted data from Client {client.client_id}")

            # Small delay between receiving results
            time.sleep(0.2)

        print(f"[Server] All {len(self.clients)} clients have submitted results\n")

    # def decrypt_and_merge(self):
    #     """
    #     Decrypt all results and merge itemsets.
    #     Detects and aborts if any client payload appears tampered.
    #     """
    #     print("\n" + "="*70)
    #     print("SERVER: Decrypting and Merging Results")
    #     print("="*70)

    #     decrypted_results = []
    #     compromised_client = None  # Track if any client is compromised

    #     for i, result_blob in enumerate(self.results):
    #         try:
    #             envelope_bytes, signature, client_pubkey_bytes = result_blob
    #         except Exception as e:
    #             print(f"[Server] Invalid result format from index {i}: {e}")
    #             continue

    #         # Load client public key and verify signature
    #         try:
    #             client_pubkey = load_public_key(client_pubkey_bytes)
    #             verify(envelope_bytes, signature, client_pubkey)
    #             print(f"[Server] ✓ Signature verification passed for client payload {i+1}")
    #         except Exception as e:
    #             print(f"[Server] ⚠️ Signature verification FAILED for client payload {i+1}: {e}")
    #             compromised_client = i + 1
    #             print(f"[Server] !!! ALERT: Client {compromised_client} COMPROMISED — Signature invalid.")
    #             print(f"[Server] Aborting aggregation for safety.\n")
    #             # Immediately abort to prevent data contamination
    #             return {}, []
            
    #         # If signature OK, decrypt
    #         try:
    #             envelope = pickle.loads(envelope_bytes)
    #             plaintext = self.encryption_manager.decrypt(envelope)
    #             decrypted = pickle.loads(plaintext)
    #             decrypted_results.append(decrypted)
    #             print(f"[Server] ✓ Decrypted data from Client {decrypted['client_id']}")
    #             print(f"          Algorithm: {decrypted['algorithm']}, "
    #                 f"Transactions: {decrypted['num_transactions']}, "
    #                 f"Itemsets: {len(decrypted['itemsets'])}")
    #         except Exception as e:
    #             print(f"[Server] ⚠️ Decryption/processing failed for client payload {i+1}: {e}")
    #             compromised_client = i + 1
    #             print(f"[Server] !!! ALERT: Client {compromised_client} COMPROMISED — Decryption failed.")
    #             print(f"[Server] Aborting aggregation for safety.\n")
    #             return {}, []

    #     # No compromises detected → proceed to merge
    #     print("\n[Server] Merging itemsets from all clients...")
    #     time.sleep(1.0)

    #     merged_itemsets = defaultdict(int)
    #     for result in decrypted_results:
    #         for itemset_tuple, count in result['itemsets'].items():
    #             merged_itemsets[itemset_tuple] += count

    #     final_itemsets = dict(sorted(
    #         merged_itemsets.items(),
    #         key=lambda x: (len(x[0]), x[1]),
    #         reverse=True
    #     ))

    #     print(f"[Server] ✓ Merged {len(final_itemsets)} unique itemsets across all clients\n")

    #     return final_itemsets, decrypted_results

    def decrypt_and_merge(self, global_min_support=None):  # Add parameter
        """
        Decrypt all results and merge itemsets.
        Detects and aborts if any client payload appears tampered.
        Args:
            global_min_support: Global minimum support threshold to filter aggregated itemsets
        """
        print("\n" + "="*70)
        print("SERVER: Decrypting and Merging Results")
        print("="*70)

        decrypted_results = []
        compromised_client = None

        for i, result_blob in enumerate(self.results):
            try:
                envelope_bytes, signature, client_pubkey_bytes = result_blob
            except Exception as e:
                print(f"[Server] Invalid result format from index {i}: {e}")
                continue

            try:
                client_pubkey = load_public_key(client_pubkey_bytes)
                verify(envelope_bytes, signature, client_pubkey)
                print(f"[Server] ✓ Signature verification passed for client payload {i+1}")
            except Exception as e:
                print(f"[Server] ⚠️ Signature verification FAILED for client payload {i+1}: {e}")
                compromised_client = i + 1
                print(f"[Server] !!! ALERT: Client {compromised_client} COMPROMISED — Signature invalid.")
                print(f"[Server] Aborting aggregation for safety.\n")
                return {}, []
            
            try:
                envelope = pickle.loads(envelope_bytes)
                plaintext = self.encryption_manager.decrypt(envelope)
                decrypted = pickle.loads(plaintext)
                decrypted_results.append(decrypted)
                print(f"[Server] ✓ Decrypted data from Client {decrypted['client_id']}")
                print(f"          Algorithm: {decrypted['algorithm']}, "
                    f"Transactions: {decrypted['num_transactions']}, "
                    f"Itemsets: {len(decrypted['itemsets'])}")
            except Exception as e:
                print(f"[Server] ⚠️ Decryption/processing failed for client payload {i+1}: {e}")
                compromised_client = i + 1
                print(f"[Server] !!! ALERT: Client {compromised_client} COMPROMISED — Decryption failed.")
                print(f"[Server] Aborting aggregation for safety.\n")
                return {}, []

        print("\n[Server] Merging itemsets from all clients...")
        time.sleep(1.0)

        # Sum support counts across all clients
        merged_itemsets = defaultdict(int)
        for result in decrypted_results:
            for itemset_tuple, count in result['itemsets'].items():
                merged_itemsets[itemset_tuple] += count

        # ============================================
        # ADD THIS: Filter by global minimum support
        # ============================================
        if global_min_support is not None:
            print(f"[Server] Applying global min_support filter: {global_min_support}")
            before_count = len(merged_itemsets)
            merged_itemsets = {
                itemset: count 
                for itemset, count in merged_itemsets.items() 
                if count >= global_min_support
            }
            after_count = len(merged_itemsets)
            filtered_count = before_count - after_count
            print(f"[Server] Filtered out {filtered_count} itemsets below global threshold")
            print(f"[Server] Remaining itemsets: {after_count}")

        final_itemsets = dict(sorted(
            merged_itemsets.items(),
            key=lambda x: (len(x[0]), x[1]),
            reverse=True
        ))

        print(f"[Server] ✓ Merged {len(final_itemsets)} unique itemsets across all clients\n")

        return final_itemsets, decrypted_results

    def display_results(self, final_itemsets, client_details,client_count):
        """
        Display the final aggregated results
        Args:
            final_itemsets: Merged itemsets from all clients
            client_details: Details from each client
        Returns:
            dict: Final itemsets
        """
        print("\n" + "="*70)
        print("FINAL AGGREGATED RESULTS")
        print("="*70)

        # Summary statistics
        total_transactions = sum(c['num_transactions'] for c in client_details)
        print(f"\nTotal Clients: {client_count}")
        print(f"Total Transactions: {total_transactions}")
        print(f"Unique Itemsets Found: {len(final_itemsets)}")

        # Group itemsets by size
        itemsets_by_size = defaultdict(list)
        for itemset, count in final_itemsets.items():
            itemsets_by_size[len(itemset)].append((itemset, count))

        print("\n" + "-"*70)
        print("Itemsets by Size:")
        print("-"*70)

        for size in sorted(itemsets_by_size.keys()):
            print(f"\n{size}-Itemsets (Count: {len(itemsets_by_size[size])}):")
            # Show top 15 for each size
            for itemset, count in sorted(itemsets_by_size[size], key=lambda x: x[1], reverse=True)[:15]:
                print(f"  {set(itemset)} -> Frequency: {count}")
            if len(itemsets_by_size[size]) > 15:
                print(f"  ... and {len(itemsets_by_size[size]) - 15} more")

        # Save server output
        if self.output_saver:
            self.output_saver.save_server_output(
                final_itemsets, 
                client_details, 
                len(self.clients), 
                total_transactions
            )

        print("\n" + "="*70)
        print("Federated Itemset Mining Complete!")
        print("="*70)

        return final_itemsets