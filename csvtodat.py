import pickle

def convert_txt_to_dat_with_mapping(input_file, output_dat, output_mapping='item_mapping.pkl'):
    """
    Convert a text file to .dat format, converting strings to integers
    
    Input format (txt):
        burgers meatballs eggs
        chutney
        turkey avocado
    
    Output format (dat):
        1 2 3
        4
        5 6
    """
    
    print("="*80)
    print("CONVERTING TXT TO DAT (STRING TO INTEGER)")
    print("="*80)
    
    # Step 1: Build item vocabulary
    print("\n📋 Step 1: Building item vocabulary...")
    item_to_id = {}
    id_counter = 1
    
    with open(input_file, 'r') as f:
        for line in f:
            items = [item.strip() for item in line.strip().split() if item.strip()]
            for item in items:
                if item not in item_to_id:
                    item_to_id[item] = id_counter
                    id_counter += 1
    
    print(f"✅ Found {len(item_to_id)} unique items")
    
    # Step 2: Convert file
    print(f"\n📦 Step 2: Converting to numeric format...")
    
    transaction_count = 0
    with open(input_file, 'r') as fin, open(output_dat, 'w') as fout:
        for line in fin:
            items = [item.strip() for item in line.strip().split() if item.strip()]
            
            if items:
                # Convert strings to IDs
                item_ids = [str(item_to_id[item]) for item in items]
                # Write as space-separated integers
                fout.write(' '.join(item_ids) + '\n')
                transaction_count += 1
    
    print(f"✅ Converted {transaction_count} transactions")
    
    # Step 3: Save mapping
    print(f"\n💾 Step 3: Saving item mapping...")
    
    with open(output_mapping, 'wb') as f:
        pickle.dump(item_to_id, f)
    
    # Also save reverse mapping
    id_to_item = {v: k for k, v in item_to_id.items()}
    reverse_mapping = output_mapping.replace('.pkl', '_reverse.pkl')
    with open(reverse_mapping, 'wb') as f:
        pickle.dump(id_to_item, f)
    
    print(f"✅ Saved mappings:")
    print(f"   {output_mapping}")
    print(f"   {reverse_mapping}")
    
    # Step 4: Show sample
    print(f"\n📝 Sample item mappings (first 10):")
    for item, id_val in list(item_to_id.items())[:10]:
        print(f"   '{item}' -> {id_val}")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"Input:  {input_file}")
    print(f"Output: {output_dat}")
    print(f"Transactions: {transaction_count}")
    print(f"Unique items: {len(item_to_id)}")
    print("="*80)
    
    return item_to_id, id_to_item


if __name__ == '__main__':
    # Example usage
    INPUT_FILE = '/home/shagnik/Documents/MinorSpec/Dynamic Fed Mining 2/mushrooms.txt'      # Your input file with strings
    OUTPUT_FILE = 'mushroom.dat'     # Output .dat file with integers
    MAPPING_FILE = 'item_mapping.pkl'  # Saves the string->int mapping
    
    # Convert the file
    item_to_id, id_to_item = convert_txt_to_dat_with_mapping(
        INPUT_FILE, 
        OUTPUT_FILE, 
        MAPPING_FILE
    )
    
    # Optional: Print statistics
    print(f"\n📊 Statistics:")
    print(f"   Most common items:")
    
    # Count item frequencies
    from collections import Counter
    item_counts = Counter()
    with open(INPUT_FILE, 'r') as f:
        for line in f:
            items = [item.strip() for item in line.strip().split() if item.strip()]
            item_counts.update(items)
    
    for item, count in item_counts.most_common(5):
        print(f"   '{item}' (ID {item_to_id[item]}): {count} occurrences")
