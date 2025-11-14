import time
from collections import defaultdict
from output_saver import OutputSaver
from federated_nodes import ClientNode, CentralServer
import math

def load_data_from_file(file_path, num_rows=None, delimiter=','):
    transactions = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for i, line in enumerate(file):
                if num_rows is not None and i >= num_rows:
                    break

                line = line.strip()
                if not line:
                    continue

                items = [item.strip() for item in line.split(delimiter) if item.strip()]

                if items:
                    transactions.append(items)

        print(f"\n✅ Successfully loaded {len(transactions)} transactions from {file_path}")
        return transactions

    except FileNotFoundError:
        print(f"\n❌ Error: File '{file_path}' not found!")
        return []
    except Exception as e:
        print(f"\n❌ Error reading file: {str(e)}")
        return []


def count_total_rows(file_path):
    try:
        with open(file_path, 'r') as f:
            return sum(1 for line in f if line.strip())
    except Exception as e:
        print(f"Error counting rows: {str(e)}")
        return 0


def split_data_equally(data, num_clients=5):
    rows_per_client = len(data) // num_clients
    client_datasets = []

    for i in range(num_clients):
        start_idx = i * rows_per_client

        if i == num_clients - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + rows_per_client

        client_datasets.append(data[start_idx:end_idx])

    return client_datasets


def run_centralized_baseline(data, min_support=1, algorithm='fp-growth'):
    print("\n" + "="*70)
    print("RUNNING CENTRALIZED BASELINE ALGORITHM")
    print("="*70)
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Total transactions: {len(data)}")
    print(f"Min support (absolute): {min_support}")
    print(f"Min support (percentage): {(min_support/len(data))*100:.2f}%")

    if algorithm == 'fp-growth':
        from mlxtend.preprocessing import TransactionEncoder
        from mlxtend.frequent_patterns import fpgrowth
        import pandas as pd

        print("Encoding transactions...")
        te = TransactionEncoder()
        te_ary = te.fit(data).transform(data)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        print(f"Total unique items: {len(te.columns_)}")

        print("Running FP-Growth algorithm...")
        min_support_ratio = min_support / len(data)
        frequent_itemsets = fpgrowth(df, min_support=min_support_ratio, use_colnames=True)

        baseline_itemsets = {}
        for idx, row in frequent_itemsets.iterrows():
            itemset = tuple(sorted(row['itemsets']))
            support = int(row['support'] * len(data))
            baseline_itemsets[itemset] = support

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"✅ Baseline found {len(baseline_itemsets)} frequent itemsets")

    if baseline_itemsets:
        print(f"\n📋 Sample frequent itemsets (first 10):")
        for i, (itemset, support) in enumerate(list(baseline_itemsets.items())[:10]):
            if len(itemset) == 1:
                print(f"   {i+1}. {{{itemset[0]}}}: support={support}")
            else:
                print(f"   {i+1}. {set(itemset)}: support={support}")

    size_distribution = defaultdict(int)
    for itemset in baseline_itemsets.keys():
        size_distribution[len(itemset)] += 1

    print(f"\n📊 Itemset Size Distribution:")
    for size in sorted(size_distribution.keys()):
        print(f"   Size {size}: {size_distribution[size]} itemsets")

    return baseline_itemsets


def compare_results_and_calculate_f1(federated_itemsets, baseline_itemsets):
    print("\n" + "="*70)
    print("COMPARING FEDERATED VS BASELINE RESULTS")
    print("="*70)

    fed_set = set()
    for itemset in federated_itemsets.keys():
        if isinstance(itemset, str):
            fed_set.add((itemset,))
        else:
            fed_set.add(tuple(sorted(itemset)))

    base_set = set(baseline_itemsets.keys())

    common_itemsets = fed_set.intersection(base_set)
    only_federated = fed_set - base_set
    only_baseline = base_set - fed_set

    num_federated = len(fed_set)
    num_baseline = len(base_set)
    num_common = len(common_itemsets)

    precision = num_common / num_federated if num_federated > 0 else 0
    recall = num_common / num_baseline if num_baseline > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 Itemset Counts:")
    print(f"  Baseline (Centralized):  {num_baseline}")
    print(f"  Federated:               {num_federated}")
    print(f"  Common (Both found):     {num_common}")
    print(f"  Only in Baseline:        {len(only_baseline)}")
    print(f"  Only in Federated:       {len(only_federated)}")

    print(f"\n📈 Evaluation Metrics:")
    print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"  Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"  F1 Score:  {f1_score:.4f} ({f1_score*100:.2f}%)")

    if only_baseline:
        print(f"\n⚠️  Sample itemsets MISSED by Federated (first 5):")
        for i, itemset in enumerate(list(only_baseline)[:5]):
            if len(itemset) == 1:
                print(f"     {i+1}. {{{itemset[0]}}} (support: {baseline_itemsets[itemset]})")
            else:
                print(f"     {i+1}. {set(itemset)} (support: {baseline_itemsets[itemset]})")

    if only_federated:
        print(f"\n⚠️  Sample itemsets ONLY in Federated (first 5):")
        for i, itemset in enumerate(list(only_federated)[:5]):
            fed_support = federated_itemsets.get(itemset, 'N/A')
            if len(itemset) == 1:
                print(f"     {i+1}. {{{itemset[0]}}} (support: {fed_support})")
            else:
                print(f"     {i+1}. {set(itemset)} (support: {fed_support})")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'num_baseline': num_baseline,
        'num_federated': num_federated,
        'num_common': num_common,
        'common_itemsets': common_itemsets,
        'only_baseline': only_baseline,
        'only_federated': only_federated
    }


def run_federated_itemset_mining(file_path, num_rows=None, delimiter=',', 
                                 min_support=1, baseline_algorithm='fp-growth', 
                                 num_clients=5):
    output_saver = OutputSaver()

    print("\n" + "="*70)
    print("LOADING DATA FROM FILE")
    print("="*70)
    all_transactions = load_data_from_file(file_path, num_rows, delimiter)

    if not all_transactions:
        print("❌ No data loaded. Exiting...")
        return None, None, None, None, None

    print(f"Total transactions loaded: {len(all_transactions)}")
    if all_transactions:
        sample_items = all_transactions[0][:10] if len(all_transactions[0]) > 10 else all_transactions[0]
        print(f"Sample transaction (first 10 items): {sample_items}...")
        print(f"Items per transaction: {len(all_transactions[0])} items")

    baseline_itemsets = run_centralized_baseline(
        all_transactions, 
        min_support=min_support, 
        algorithm=baseline_algorithm
    )

    print("\n" + "="*70)
    print("RUNNING FEDERATED APPROACH")
    print("="*70)

    client_datasets = split_data_equally(all_transactions, num_clients=num_clients)

    print(f"\n📊 Data split among {len(client_datasets)} clients:")
    for i, dataset in enumerate(client_datasets, 1):
        print(f"  Client {i}: {len(dataset)} transactions")

    support_percentage = min_support / len(all_transactions)

    print(f"\n🔧 Min support configuration:")
    print(f"   Global min support (absolute): {min_support}")
    print(f"   Global min support (percentage): {(support_percentage)*100:.2f}%")
    print(f"   Strategy: Using PROPORTIONAL min_support per client")

    clients = []
    for i in range(num_clients):
        client_data_size = len(client_datasets[i])

        proportional_threshold = client_data_size * support_percentage
        client_min_support = max(1, int(proportional_threshold * 0.5))  # ← NEW LINE
    
        print(f"   Client {i+1} min_support: {client_min_support} " +
          f"({(client_min_support/client_data_size)*100:.2f}% of {client_data_size}) " +
          f"[Lenient: 50% of {int(proportional_threshold)}]")
        client = ClientNode(
            client_id=i+1,
            data=client_datasets[i],
            algorithm='fp-growth',
            min_support=client_min_support,
            output_saver=output_saver
        )
        clients.append(client)

    # clients = []
    # for i in range(num_clients):
    #     client_data_size = len(client_datasets[i])
    #     client_min_support = max(1, int(client_data_size * support_percentage))
        
    #     # Assign algorithm: First 3 clients use FP-Growth, last 2 use Apriori
    #     if i < 3:
    #         algorithm = 'fp-growth'
    #     else:
    #         algorithm = 'apriori'
        
    #     print(f"   Client {i+1} min_support: {client_min_support} ({(client_min_support/client_data_size)*100:.2f}% of {client_data_size}) - Algorithm: {algorithm.upper()}")
        
    #     client = ClientNode(
    #         client_id=i+1,
    #         data=client_datasets[i],
    #         algorithm=algorithm,  # Use dynamic algorithm assignment
    #         min_support=client_min_support,
    #         output_saver=output_saver
    #     )
    #     clients.append(client)

    print("\n" + "="*70)
    print("CLIENT CONFIGURATION")
    print("="*70)
    for i, client in enumerate(clients):
        sample_items = client.data[0][:5] if len(client.data[0]) > 5 else client.data[0]
        print(f"Client {client.client_id}:")
        print(f"  - Transactions: {len(client.data)}")
        print(f"  - Algorithm: {client.algorithm.upper()}")
        print(f"  - Min Support: {client.min_support}")
        print(f"  - Sample transaction: {sample_items}...")

    server = CentralServer(output_saver=output_saver)

    print("\n" + "="*70)
    print("FEDERATED MINING PHASES")
    print("="*70)
    server.initialize_clients(clients)

    server.collect_results()

    final_itemsets, client_details = server.decrypt_and_merge(
        global_min_support=min_support  # Add this parameter
    )

    server.display_results(final_itemsets, client_details, num_clients)

    comparison_metrics = compare_results_and_calculate_f1(final_itemsets, baseline_itemsets)

    return final_itemsets, server, clients, comparison_metrics, baseline_itemsets


if __name__ == "__main__":
    print("\n" + "🚀 " * 35)
    print("STARTING FEDERATED ITEMSET MINING SIMULATION")
    print("🚀 " * 35)

    FILE_PATH = '/home/shagnik/Documents/MinorSpec/Dynamic Fed Mining 2/mushrooms.txt'
    DELIMITER = ' '
    BASELINE_ALGORITHM = 'fp-growth'
    NUM_CLIENTS = 5

    TOTAL_ROWS = count_total_rows(FILE_PATH)
    NUM_ROWS = TOTAL_ROWS
    MIN_SUPPORT = math.ceil(0.35*NUM_ROWS)

    print(f"\n⚙️  Configuration Summary:")
    print(f"   - Dataset: {FILE_PATH}")
    print(f"   - Total Rows in File: {TOTAL_ROWS}")
    print(f"   - Using Rows: {NUM_ROWS}")
    print(f"   - Min Support (absolute): {MIN_SUPPORT} transaction(s)")
    print(f"   - Baseline Algorithm: {BASELINE_ALGORITHM.upper()}")
    print(f"   - Number of Clients: {NUM_CLIENTS}")
    print(f"   - Delimiter: '{DELIMITER}'")
    print(f"\n   ✅ Using PROPORTIONAL min_support strategy")
    print(f"      - Baseline: {MIN_SUPPORT} out of {NUM_ROWS} ({(MIN_SUPPORT/NUM_ROWS)*100:.2f}%)")
    print(f"      - Each client: Scales based on their data size")

    print("\n" + "="*70)
    print("STARTING ANALYSIS...")
    print("="*70)

    start_time = time.time()
    final_results, server, clients, metrics, baseline = run_federated_itemset_mining(
        FILE_PATH, 
        NUM_ROWS, 
        DELIMITER,
        MIN_SUPPORT,
        BASELINE_ALGORITHM,
        NUM_CLIENTS
    )
    end_time = time.time()

    if final_results is not None:
        print(f"\n\nTotal execution time: {end_time - start_time:.2f} seconds")

        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        print(f"📊 Dataset: {NUM_ROWS} transactions")
        print(f"📊 Min Support (absolute): {MIN_SUPPORT} transaction(s)")
        print(f"📊 Baseline Algorithm: {BASELINE_ALGORITHM.upper()}")
        print(f"📊 Number of Clients: {NUM_CLIENTS}")
        print(f"📊 Baseline Itemsets: {metrics['num_baseline']}")
        print(f"📊 Federated Itemsets: {metrics['num_federated']}")
        print(f"📊 Common Itemsets: {metrics['num_common']}")
        print(f"\n🎯 Performance Metrics:")
        print(f"   F1 Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"   Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")

        print(f"\n💡 Interpretation:")
        if metrics['f1_score'] >= 0.95:
            print("   ✅ Excellent! Federated approach matches centralized baseline almost perfectly.")
        elif metrics['f1_score'] >= 0.80:
            print("   ✅ Good! Federated approach performs well compared to centralized baseline.")
        elif metrics['f1_score'] >= 0.60:
            print("   ⚠️  Moderate. Some discrepancies between federated and centralized results.")
        else:
            print("   ⚠️  Low match. Consider adjusting min support or investigating data distribution.")

        print("\n📈 Detailed Analysis:")
        print(f"   - Items discovered by both: {metrics['num_common']}")
        print(f"   - Items missed by federated: {len(metrics['only_baseline'])}")
        print(f"   - Extra items found by federated: {len(metrics['only_federated'])}")

        print("\n\n" + "✅ " * 35)
        print("SIMULATION COMPLETE")
        print("✅ " * 35)
    else:
        print("\n\n" + "❌ " * 35)
        print("SIMULATION FAILED")
        print("❌ " * 35)
