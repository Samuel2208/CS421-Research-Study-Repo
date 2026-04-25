import os
import pandas as pd

def find_best_metrics():
    target_files = [
        "qwen_dailydialog_window_sizes.csv",
        "qwen_meld_window_sizes.csv",
        "gemma_dailydialog_window_sizes.csv",
        "gemma_meld_window_sizes.csv"
    ]

    all_best_results = []

    for file in target_files:
        if os.path.exists(file):
            print(f"Processing {file}...")
            df = pd.read_csv(file)

            df['group_value'] = df['group_value'].astype(float)
            df['metric_value'] = df['metric_value'].astype(float)

            dataset_name = file.replace('_window_sizes.csv', '')

            for window_size, group_df in df.groupby('group_value'):
                
                acc_df = group_df[group_df['metric_name'] == 'accuracy']
                if not acc_df.empty:
                    best_acc_idx = acc_df['metric_value'].idxmax()
                    best_acc_row = acc_df.loc[best_acc_idx]
                    best_acc = best_acc_row['metric_value']
                    best_acc_source = best_acc_row['source_file']
                else:
                    best_acc, best_acc_source = None, None

                f1_df = group_df[group_df['metric_name'] == 'macro_f1']
                if not f1_df.empty:
                    best_f1_idx = f1_df['metric_value'].idxmax()
                    best_f1_row = f1_df.loc[best_f1_idx]
                    best_f1 = best_f1_row['metric_value']
                    best_f1_source = best_f1_row['source_file']
                else:
                    best_f1, best_f1_source = None, None

                all_best_results.append({
                    'Table': dataset_name,
                    'Window_Size': window_size,
                    'Best_Accuracy': best_acc,
                    'Accuracy_Source': best_acc_source,
                    'Best_Macro_F1': best_f1,
                    'F1_Source': best_f1_source
                })
        else:
            print(f"File not found: {file}")

    if all_best_results:
        summary_df = pd.DataFrame(all_best_results)
        
        summary_df = summary_df.sort_values(by=['Table', 'Window_Size'])
        
        output_file = "best_window_size_metrics.csv"
        summary_df.to_csv(output_file, index=False)
        
        print("\n" + "="*80)
        print(f"Summary saved to {output_file}")
        print("="*80 + "\n")
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(summary_df.to_string(index=False))
    else:
        print("\nNo data was processed. Make sure the 4 CSV files are in the same folder as this script.")

if __name__ == "__main__":
    find_best_metrics()