"""
Itemset Mining Algorithms Module
Implements Apriori and FP-Growth algorithms for frequent itemset mining
"""

from itertools import combinations
from collections import defaultdict


class AprioriMiner:
    """
    Apriori algorithm for frequent itemset mining

    The Apriori algorithm uses a level-wise approach:
    1. Find all frequent 1-itemsets
    2. Use frequent k-itemsets to generate candidate (k+1)-itemsets
    3. Prune candidates that don't meet minimum support
    4. Repeat until no more frequent itemsets are found
    """

    def __init__(self, min_support=2):
        """
        Initialize Apriori miner
        Args:
            min_support: Minimum support count for an itemset to be considered frequent
        """
        self.min_support = min_support

    def mine(self, transactions):
        """
        Mine frequent itemsets using Apriori algorithm
        Args:
            transactions: List of transactions, where each transaction is a list of items
        Returns:
            dict: Dictionary with itemsets as keys (frozenset) and frequencies as values
        """
        # Count individual items
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # Filter items by minimum support
        frequent_items = {item: count for item, count in item_counts.items() 
                         if count >= self.min_support}

        result = {}

        # Add 1-itemsets
        for item, count in frequent_items.items():
            result[frozenset([item])] = count

        # Generate k-itemsets (k=2,3,...)
        k = 2
        current_itemsets = [frozenset([item]) for item in frequent_items.keys()]

        while current_itemsets:
            # Generate candidate itemsets
            candidates = self._generate_candidates(current_itemsets, k)

            # Count candidate support
            candidate_counts = defaultdict(int)
            for transaction in transactions:
                transaction_set = set(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        candidate_counts[candidate] += 1

            # Filter by minimum support
            frequent_k_itemsets = {itemset: count for itemset, count in candidate_counts.items()
                                  if count >= self.min_support}

            if not frequent_k_itemsets:
                break

            result.update(frequent_k_itemsets)
            current_itemsets = list(frequent_k_itemsets.keys())
            k += 1

        return result

    def _generate_candidates(self, itemsets, k):
        """
        Generate candidate k-itemsets from (k-1)-itemsets
        Args:
            itemsets: List of frequent (k-1)-itemsets
            k: Size of itemsets to generate
        Returns:
            set: Set of candidate k-itemsets
        """
        candidates = set()
        itemsets_list = list(itemsets)

        for i in range(len(itemsets_list)):
            for j in range(i + 1, len(itemsets_list)):
                union = itemsets_list[i] | itemsets_list[j]
                if len(union) == k:
                    candidates.add(union)

        return candidates


class FPGrowthMiner:
    """
    FP-Growth algorithm for frequent itemset mining
    
    This is a corrected implementation that properly handles:
    - Conditional pattern bases
    - Item frequency counting
    - Recursive pattern mining
    """

    def __init__(self, min_support=2):
        """
        Initialize FP-Growth miner
        Args:
            min_support: Minimum support count for an itemset to be considered frequent
        """
        self.min_support = min_support

    def mine(self, transactions):
        """
        Mine frequent itemsets using FP-Growth algorithm
        Args:
            transactions: List of transactions, where each transaction is a list of items
        Returns:
            dict: Dictionary with itemsets as keys (frozenset) and frequencies as values
        """
        # Count item frequencies
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1

        # Filter by minimum support
        frequent_items = {item: count for item, count in item_counts.items()
                         if count >= self.min_support}

        if not frequent_items:
            return {}

        result = {}

        # Add 1-itemsets
        for item, count in frequent_items.items():
            result[frozenset([item])] = count

        # Sort items by frequency (descending)
        sorted_items = sorted(frequent_items.items(), key=lambda x: x[1], reverse=True)
        item_order = {item: idx for idx, (item, _) in enumerate(sorted_items)}

        # Order transactions by item frequency
        ordered_transactions = []
        for transaction in transactions:
            ordered = sorted([item for item in transaction if item in frequent_items],
                           key=lambda x: item_order[x])
            if ordered:
                ordered_transactions.append(ordered)

        # Mine patterns using recursive FP-Growth
        self._fp_growth(ordered_transactions, [], result, item_order)

        return result

    def _fp_growth(self, transactions, prefix, result, item_order):
        """
        Recursive FP-Growth pattern mining (CORRECTED VERSION)
        
        Args:
            transactions: List of ordered transactions
            prefix: Current itemset prefix
            result: Dictionary to store results
            item_order: Item ordering dictionary
        """
        # Count items in current transactions
        item_counts = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                # CRITICAL FIX: Don't count items already in prefix
                if item not in prefix:
                    item_counts[item] += 1

        # Sort items by original order
        sorted_items = sorted(item_counts.items(), key=lambda x: item_order.get(x[0], float('inf')))

        # Process each frequent item
        for item, count in sorted_items:
            if count >= self.min_support:
                # Create new itemset with current item
                new_itemset = prefix + [item]
                result[frozenset(new_itemset)] = count

                # Build conditional pattern base for this item
                conditional_transactions = []
                for transaction in transactions:
                    if item in transaction:
                        # Get all items before current item (conditional pattern base)
                        idx = transaction.index(item)
                        if idx > 0:
                            # Filter out items already in prefix
                            pattern = [i for i in transaction[:idx] if i not in prefix and i != item]
                            if pattern:
                                conditional_transactions.append(pattern)

                # Recursively mine with conditional transactions
                if conditional_transactions:
                    self._fp_growth(conditional_transactions, new_itemset, result, item_order)