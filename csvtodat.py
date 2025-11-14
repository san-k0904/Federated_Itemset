import pandas as pd
import csv


def csv_to_dat_transaction_format(csv_file, output_dat_file, delimiter=' ', 
                                   transaction_column=None, item_columns=None,
                                   include_header=False):
    """
    Convert CSV to DAT format for itemset mining (transaction format)
    
    Args:
        csv_file (str): Path to input CSV file
        output_dat_file (str): Path to output DAT file
        delimiter (str): Delimiter for output file (default: space)
        transaction_column (str): Column name containing transaction ID (optional)
        item_columns (list): List of column names to include as items (if None, all columns used)
        include_header (bool): Whether to include header in output (default: False)
    """
    try:
        df = pd.read_csv(csv_file)
        
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        print(f"Columns: {list(df.columns)}")
        
        with open(output_dat_file, 'w') as dat_file:
            if include_header and item_columns:
                dat_file.write(delimiter.join(item_columns) + '\n')
            
            if item_columns:
                for idx, row in df.iterrows():
                    items = [str(row[col]) for col in item_columns if pd.notna(row[col])]
                    if items:
                        dat_file.write(delimiter.join(items) + '\n')
            else:
                for idx, row in df.iterrows():
                    items = [str(val) for val in row if pd.notna(val)]
                    if items:
                        dat_file.write(delimiter.join(items) + '\n')
        
        print(f"✅ Successfully converted to {output_dat_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def csv_to_dat_basket_format(csv_file, output_dat_file, delimiter=' ',
                             group_by_column=None):
    """
    Convert CSV to DAT in basket format (each row = one transaction with multiple items)
    Useful for market basket analysis
    
    Args:
        csv_file (str): Path to input CSV file
        output_dat_file (str): Path to output DAT file
        delimiter (str): Delimiter for output file (default: space)
        group_by_column (str): Column to group by (e.g., transaction ID, customer ID)
    """
    try:
        df = pd.read_csv(csv_file)
        
        print(f"✅ Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        if group_by_column and group_by_column in df.columns:
            item_columns = [col for col in df.columns if col != group_by_column]
            
            with open(output_dat_file, 'w') as dat_file:
                for group_id, group_df in df.groupby(group_by_column):
                    items = []
                    for _, row in group_df.iterrows():
                        for col in item_columns:
                            if pd.notna(row[col]) and str(row[col]).strip():
                                items.append(str(row[col]))
                    
                    if items:
                        dat_file.write(delimiter.join(items) + '\n')
        else:
            csv_to_dat_transaction_format(csv_file, output_dat_file, delimiter)
        
        print(f"✅ Successfully converted to {output_dat_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def csv_to_txt_simple(csv_file, output_txt_file, delimiter=' '):
    """
    Simple CSV to TXT conversion
    
    Args:
        csv_file (str): Path to input CSV file
        output_txt_file (str): Path to output TXT file
        delimiter (str): Delimiter for output file (default: space)
    """
    try:
        df = pd.read_csv(csv_file)
        df.to_csv(output_txt_file, sep=delimiter, index=False, header=False)
        print(f"✅ Successfully converted {csv_file} to {output_txt_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def csv_single_column_to_dat(csv_file, output_dat_file, column_name, 
                             delimiter=' ', split_items=True, item_separator=','):
    """
    Convert single column CSV (containing transaction data) to DAT format
    Common for pre-processed transaction datasets
    
    Args:
        csv_file (str): Path to input CSV file
        output_dat_file (str): Path to output DAT file
        column_name (str): Name of column containing transaction items
        delimiter (str): Delimiter for output file (default: space)
        split_items (bool): Whether items in column are separated (default: True)
        item_separator (str): Separator used within items (default: comma)
    """
    try:
        df = pd.read_csv(csv_file)
        
        if column_name not in df.columns:
            print(f"❌ Column '{column_name}' not found in CSV")
            return
        
        with open(output_dat_file, 'w') as dat_file:
            for idx, row in df.iterrows():
                if pd.notna(row[column_name]):
                    if split_items:
                        items = str(row[column_name]).split(item_separator)
                        items = [item.strip() for item in items if item.strip()]
                        dat_file.write(delimiter.join(items) + '\n')
                    else:
                        dat_file.write(str(row[column_name]) + '\n')
        
        print(f"✅ Successfully converted to {output_dat_file}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def csv_binary_to_dat(csv_file, output_dat_file, delimiter=' ', 
                     transaction_id_column=None):
    """
    Convert binary matrix CSV (1/0 for item presence) to transaction DAT format
    
    Args:
        csv_file (str): Path to input CSV file (binary matrix format)
        output_dat_file (str): Path to output DAT file
        delimiter (str): Delimiter for output file (default: space)
        transaction_id_column (str): Column name for transaction ID (skip if None)
    """
    try:
        df = pd.read_csv(csv_file)
        
        if transaction_id_column:
            df = df.drop(columns=[transaction_id_column])
        
        with open(output_dat_file, 'w') as dat_file:
            for idx, row in df.iterrows():
                items = [col for col in df.columns if row[col] == 1 or row[col] == '1']
                if items:
                    dat_file.write(delimiter.join(items) + '\n')
        
        print(f"✅ Successfully converted binary matrix to {output_dat_file}")
        print(f"   Converted {len(df)} transactions")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("CSV to DAT/TXT Converter for Itemset Mining\n")
    print("="*70)
    
    csv_input = "store_data.csv"
    dat_output = "store_data.dat"
    txt_output = "store_data.txt"
    
    print("\nExample 1: Simple CSV to DAT (all columns as items)")
    csv_to_dat_transaction_format(
        csv_file=csv_input,
        output_dat_file=dat_output,
        delimiter=' '
    )
    
    print("\n" + "="*70)
    print("\nExample 2: CSV to DAT (specific columns only)")
    csv_to_dat_transaction_format(
        csv_file=csv_input,
        output_dat_file="output_selected.dat",
        delimiter=' ',
        item_columns=['item1', 'item2', 'item3']
    )
    
    print("\n" + "="*70)
    print("\nExample 3: CSV to DAT (basket format with grouping)")
    csv_to_dat_basket_format(
        csv_file=csv_input,
        output_dat_file="output_basket.dat",
        delimiter=' ',
        group_by_column='transaction_id'
    )
    
    print("\n" + "="*70)
    print("\nExample 4: Single column CSV to DAT")
    csv_single_column_to_dat(
        csv_file=csv_input,
        output_dat_file="output_single.dat",
        column_name='items',
        delimiter=' ',
        split_items=True,
        item_separator=','
    )
    
    print("\n" + "="*70)
    print("\nExample 5: Binary matrix CSV to DAT")
    csv_binary_to_dat(
        csv_file="binary_matrix.csv",
        output_dat_file="output_binary.dat",
        delimiter=' ',
        transaction_id_column='trans_id'
    )
    
    print("\n" + "="*70)
    print("\nExample 6: Simple CSV to TXT")
    csv_to_txt_simple(
        csv_file=csv_input,
        output_txt_file=txt_output,
        delimiter=' '
    )
    
    print("\n" + "="*70)
    print("\n✅ Conversion complete!")
