import multiprocessing as mp
import time
import os
import pandas as pd
from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth

from output_saver import OutputSaver
from federated_nodes import ClientNode, CentralServer



def load_transactions(file_path="store_data.dat"):
    dataset = []
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    with open(file_path, 'r') as f:
        for line in f:
            items = line.strip().split()
            if items:
                dataset.append(items)

    print(f"\n✅ Loaded {len(dataset)} transactions from {file_path}")
    return dataset


def split_for_clients(dataset, num_clients):
    n = len(dataset)
    chunk_size = (n + num_clients - 1) // num_clients
    splits = [dataset[i * chunk_size:(i + 1) * chunk_size] for i in range(num_clients)]
    return splits



def run_centralized_baseline(data, min_support=10):
    print("\n" + "="*70)
    print("RUNNING CENTRALIZED BASELINE (FP-Growth)")
    print("="*70)

    te = TransactionEncoder()
    te_ary = te.fit(data).transform(data)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    min_support_ratio = min_support / len(data)
    frequent_itemsets = fpgrowth(df, min_support=min_support_ratio, use_colnames=True)

    baseline_itemsets = {}
    for _, row in frequent_itemsets.iterrows():
        itemset = tuple(sorted(row['itemsets']))
        support = int(row['support'] * len(data))
        baseline_itemsets[itemset] = support

    print(f"✅ Baseline found {len(baseline_itemsets)} frequent itemsets.")
    return baseline_itemsets




def compare_results_and_calculate_f1(federated_itemsets, baseline_itemsets):
    print("\n" + "="*70)
    print("COMPARING FEDERATED VS BASELINE RESULTS")
    print("="*70)

    fed_set = {tuple(sorted(i)) if not isinstance(i, str) else (i,) for i in federated_itemsets.keys()}
    base_set = set(baseline_itemsets.keys())

    common = fed_set.intersection(base_set)
    only_fed = fed_set - base_set
    only_base = base_set - fed_set

    num_fed = len(fed_set)
    num_base = len(base_set)
    num_common = len(common)

    precision = num_common / num_fed if num_fed else 0
    recall = num_common / num_base if num_base else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n📊 Itemset Counts:")
    print(f"  Baseline (Centralized):  {num_base}")
    print(f"  Federated:               {num_fed}")
    print(f"  Common (Both found):     {num_common}")
    print(f"  Only in Baseline:        {len(only_base)}")
    print(f"  Only in Federated:       {len(only_fed)}")

    print(f"\n📈 Evaluation Metrics:")
    print(f"  Precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall * 100:.2f}%)")
    print(f"  F1 Score:  {f1:.4f} ({f1 * 100:.2f}%)")

    size_distribution = defaultdict(int)
    for itemset in federated_itemsets.keys():
        size_distribution[len(itemset)] += 1

    print(f"\n📊 Itemset Size Distribution (Federated):")
    for size in sorted(size_distribution.keys()):
        print(f"   Size {size}: {size_distribution[size]} itemsets")

    if only_base:
        print(f"\n⚠️  Sample itemsets missed by Federated:")
        for i, itemset in enumerate(list(only_base)[:5]):
            print(f"   {i+1}. {set(itemset)}")

    if only_fed:
        print(f"\n⚠️  Sample itemsets only in Federated:")
        for i, itemset in enumerate(list(only_fed)[:5]):
            print(f"   {i+1}. {set(itemset)}")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'num_common': num_common,
        'num_federated': num_fed,
        'num_baseline': num_base
    }



def client_process(child_conn, client_id, data, algorithm, min_support, encryption_key, use_output_saver):
    try:
        start = time.time()
        output_saver = OutputSaver() if use_output_saver else None
        client = ClientNode(client_id=client_id, data=data, algorithm=algorithm,
                            min_support=min_support, output_saver=output_saver)
        client.set_encryption_key(encryption_key)
        encrypted_blob = client.process_and_send()
        child_conn.send(encrypted_blob)
        child_conn.close()
        print(f"[Client {client_id}] ✅ Completed in {time.time() - start:.2f}s")
    except Exception as e:
        child_conn.send(("error", client_id, str(e)))
        child_conn.close()
        print(f"[Client {client_id}] ❌ Error: {e}")



def run_multiprocess_federated(num_clients=5, use_output_saver=False,
                               dataset_path="store_data.dat", merge_support_factor=0.9):
    """
    merge_support_factor (float): scales the global threshold for merging.
    e.g., 1.0 = strict filtering, 0.9 = slightly relaxed (recover recall).
    """

    print("\n" + "🚀 " * 25)
    print("STARTING FEDERATED ITEMSET MINING (MULTIPROCESS MODE)")
    print("🚀 " * 25)

    output_saver = OutputSaver() if use_output_saver else None
    dataset = load_transactions(dataset_path)
    total_rows = len(dataset)

    SUPPORT_PERCENTAGE = 0.05
    MIN_SUPPORT = max(10, int(total_rows * SUPPORT_PERCENTAGE))
    support_percentage = MIN_SUPPORT / total_rows
    MERGE_THRESHOLD = int(MIN_SUPPORT * merge_support_factor)

    print(f"\n⚙️  Configuration Summary:")
    print(f"   - Dataset: {dataset_path}")
    print(f"   - Total Transactions: {total_rows}")
    print(f"   - Global Min Support: {MIN_SUPPORT} ({SUPPORT_PERCENTAGE * 100:.2f}%)")
    print(f"   - Merge Threshold: {MERGE_THRESHOLD} (Factor={merge_support_factor})")
    print(f"   - Number of Clients: {num_clients}")

    print("\n" + "="*70)
    print("RUNNING CENTRALIZED BASELINE")
    print("="*70)
    baseline_itemsets = run_centralized_baseline(dataset, min_support=MIN_SUPPORT)

    print("\n" + "="*70)
    print("SPLITTING DATA AMONG CLIENTS")
    print("="*70)
    client_datasets = split_for_clients(dataset, num_clients)
    for i, data_chunk in enumerate(client_datasets, 1):
        print(f"  Client {i}: {len(data_chunk)} transactions")

    print("\n" + "="*70)
    print("INITIALIZING SERVER AND CLIENTS")
    print("="*70)
    server = CentralServer(output_saver=output_saver)
    encryption_key = server.encryption_manager.get_key()

    processes = []
    for i, data_chunk in enumerate(client_datasets):
        client_data_size = len(data_chunk)
        client_min_support = max(1, int(client_data_size * support_percentage))
        print(f"   🔹 Configured Client {i+1} with {client_data_size} txns, "
              f"min_support={client_min_support} ({(client_min_support/client_data_size)*100:.2f}%)")

        parent_conn, child_conn = mp.Pipe(duplex=False)
        p = mp.Process(target=client_process, args=(
            child_conn, i + 1, data_chunk, 'fp-growth',
            client_min_support, encryption_key, use_output_saver
        ))
        processes.append((p, parent_conn))

    print("\n" + "="*70)
    print("LAUNCHING CLIENT PROCESSES")
    print("="*70)
    for p, _ in processes:
        p.start()

    encrypted_results = []
    for p, parent_conn in processes:
        msg = parent_conn.recv()
        if isinstance(msg, tuple) and msg[0] == "error":
            _, cid, err = msg
            print(f"❌ Error from Client {cid}: {err}")
        else:
            encrypted_results.append(msg)
        parent_conn.close()

    for p, _ in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    print("\n" + "="*70)
    print("MERGING CLIENT RESULTS ON SERVER")
    print("="*70)
    server.results = encrypted_results
    final_itemsets, client_details = server.decrypt_and_merge(global_min_support=MERGE_THRESHOLD)
    server.display_results(final_itemsets, client_details, num_clients)

    print("\n" + "="*70)
    print("EVALUATING RESULTS AGAINST BASELINE")
    print("="*70)
    metrics = compare_results_and_calculate_f1(final_itemsets, baseline_itemsets)

    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")

    print("\n🎯 Interpretation:")
    if metrics['f1_score'] >= 0.95:
        print("   ✅ Excellent! Federated results closely match centralized baseline.")
    elif metrics['f1_score'] >= 0.80:
        print("   ✅ Good! Federated approach performs well.")
    elif metrics['f1_score'] >= 0.60:
        print("   ⚠️  Moderate. Some discrepancy present.")
    else:
        print("   ⚠️  Low match — consider tuning merge_support_factor or client split.")

    return final_itemsets, server, metrics



if __name__ == "__main__":
    mp.set_start_method('fork', force=True) ##you can chng 'fork' to 'spawn' on windows
    start = time.time()
    # You can tune merge_support_factor between 0.85–1.0
    final_itemsets, server, metrics = run_multiprocess_federated(
        num_clients=5,
        use_output_saver=False,
        dataset_path="store_data.dat",
        merge_support_factor=0.9 
    )
    end = time.time()
    print(f"\n⏱ Total Execution Time: {end - start:.2f} seconds")
