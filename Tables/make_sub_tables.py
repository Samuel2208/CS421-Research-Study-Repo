import os
import pandas as pd

CONFIG_DATA = """
# Zero-Shot Tests
qwen2_tests:zero_shot:qwen_zero_shot_structured_lemmatized.py:qwen_zero_shot_structured_lemmatized_results.csv:false
qwen2_tests:zero_shot:qwen_zero_shot_structured.py:qwen_zero_shot_structured_results.csv:false
qwen2_tests:zero_shot:qwen_zeroshot_lemmatized.py:qwen_zero_shot_lemmatized_results.csv:false
qwen2_tests:zero_shot:qwen_zeroshot.py:qwen_zero_shot_results.csv:false

gemma_tests:zero_shot:gemma_zero_shot_structured_lemmatized.py:gemma_zero_shot_structured_lemmatized_results.csv:false
gemma_tests:zero_shot:gemma_zero_shot_structured.py:gemma_zero_shot_structured_results.csv:false
gemma_tests:zero_shot:gemma_zeroshot_lemmatized.py:gemma_zero_shot_lemmatized_results.csv:false
gemma_tests:zero_shot:gemma_zero_shot.py:gemma_zero_shot_results.csv:false

qwen2_tests:zero_shot:qwen_zero_shot_structured_lemmatized_dd.py:qwen_zero_shot_structured_lemmatized_dailydialog_results.csv:false
qwen2_tests:zero_shot:qwen_zero_shot_structured_dd.py:qwen_zero_shot_structured_dailydialog_results.csv:false
qwen2_tests:zero_shot:qwen_zeroshot_lemmatized_dd.py:qwen_zero_shot_lemmatized_dailydialog_results.csv:false
qwen2_tests:zero_shot:qwen_zeroshot_dd.py:qwen_zero_shot_dailydialog_results.csv:false

gemma_tests:zero_shot:gemma_zero_shot_structured_lemmatized_dd.py:gemma_zero_shot_structured_lemmatized_dailydialog_results.csv:false
gemma_tests:zero_shot:gemma_zero_shot_structured_dd.py:gemma_zero_shot_structured_dailydialog_results.csv:false
gemma_tests:zero_shot:gemma_zeroshot_lemmatized_dd.py:gemma_zero_shot_lemmatized_dailydialog_results.csv:false
gemma_tests:zero_shot:gemma_zero_shot_dd.py:gemma_zero_shot_dailydialog_results.csv:false

# Least-to-Most Tests
qwen2_tests:least_to_most:qwen_least_to_most.py:qwen_least_to_most_results.csv:true
qwen2_tests:least_to_most:qwen_least_to_most_structured.py:qwen_least_to_most_structured_results.csv:true
qwen2_tests:least_to_most:qwen_least_to_most_structured_lemmatized.py:qwen_least_to_most_structured_lemmatized_results.csv:true
qwen2_tests:least_to_most:qwen_least_to_most_lemmatized.py:qwen_least_to_most_lemmatized_results.csv:true

gemma_tests:least_to_most:gemma_least_to_most.py:gemma_least_to_most_results.csv:true
gemma_tests:least_to_most:gemma_least_to_most_structured.py:gemma_least_to_most_structured_results.csv:true
gemma_tests:least_to_most:gemma_least_to_most_structured_lemmatized.py:gemma_least_to_most_structured_lemmatized_results.csv:true
gemma_tests:least_to_most:gemma_least_to_most_lemmatized.py:gemma_least_to_most_lemmatized_results.csv:true

qwen2_tests:least_to_most:qwen_least_to_most_dd.py:qwen_least_to_most_dailydialog_results.csv:true
qwen2_tests:least_to_most:qwen_least_to_most_structured_dd.py:qwen_least_to_most_structured_dailydialog_results.csv:true
qwen2_tests:least_to_most:qwen_least_to_most_structured_lemmatized_dd.py:qwen_least_to_most_structured_lemmatized_dailydialog_results.csv:true
qwen2_tests:least_to_most:qwen_least_to_most_lemmatized_dd.py:qwen_least_to_most_lemmatized_dailydialog_results.csv:true

gemma_tests:least_to_most:gemma_least_to_most_dd.py:gemma_least_to_most_dailydialog_results.csv:true
gemma_tests:least_to_most:gemma_least_to_most_structured_dd.py:gemma_least_to_most_structured_dailydialog_results.csv:true
gemma_tests:least_to_most:gemma_least_to_most_structured_lemmatized_dd.py:gemma_least_to_most_structured_lemmatized_dailydialog_results.csv:true
gemma_tests:least_to_most:gemma_least_to_most_lemmatized_dd.py:gemma_least_to_most_lemmatized_dailydialog_results.csv:true

# 2-Shot Tests
qwen2_tests:2_shot:qwen_2shot.py:qwen_2_shot_results.csv:true
qwen2_tests:2_shot:qwen_2shot_structured.py:qwen_2_shot_structured_results.csv:true
qwen2_tests:2_shot:qwen_2shot_structured_lemmatized.py:qwen_2_shot_structured_lemmatized_results.csv:true
qwen2_tests:2_shot:qwen_2shot_lemmatized.py:qwen_2_shot_lemmatized_results.csv:true

gemma_tests:2_shot:gemma_2shot.py:gemma_2_shot_results.csv:true
gemma_tests:2_shot:gemma_2shot_structured.py:gemma_2_shot_structured_results.csv:true
gemma_tests:2_shot:gemma_2shot_structured_lemmatized.py:gemma_2_shot_structured_lemmatized_results.csv:true
gemma_tests:2_shot:gemma_2shot_lemmatized.py:gemma_2_shot_lemmatized_results.csv:true

qwen2_tests:2_shot:qwen_2shot_dd.py:qwen_2_shot_dailydialog_results.csv:true
qwen2_tests:2_shot:qwen_2shot_structured_dd.py:qwen_2_shot_structured_dailydialog_results.csv:true
qwen2_tests:2_shot:qwen_2shot_structured_lemmatized_dd.py:qwen_2_shot_structured_lemmatized_dailydialog_results.csv:true
qwen2_tests:2_shot:qwen_2shot_lemmatized_dd.py:qwen_2_shot_lemmatized_dailydialog_results.csv:true

gemma_tests:2_shot:gemma_2shot_dd.py:gemma_2_shot_dailydialog_results.csv:true
gemma_tests:2_shot:gemma_2shot_structured_dd.py:gemma_2_shot_structured_dailydialog_results.csv:true
gemma_tests:2_shot:gemma_2shot_structured_lemmatized_dd.py:gemma_2_shot_structured_lemmatized_dailydialog_results.csv:true
gemma_tests:2_shot:gemma_2shot_lemmatized_dd.py:gemma_2_shot_lemmatized_dailydialog_results.csv:true
"""

def extract_and_split_window_sizes(config_text, output_directory):
    qwen_dd_dfs = []
    qwen_meld_dfs = []
    gemma_dd_dfs = []
    gemma_meld_dfs = []
    
    for line in config_text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        parts = line.split(':')
        if len(parts) != 5:
            print(f"Skipping malformed line: {line}")
            continue
            
        folder, subfolder, python_script, original_csv, to_transfer = parts
        
        if original_csv.endswith('.csv'):
            eval_csv_name = original_csv.replace('.csv', '_evaluation.csv')
        else:
            eval_csv_name = f"{original_csv}_evaluation.csv"
            
        target_path = os.path.join(folder, subfolder, "results", eval_csv_name)
        
        is_qwen = 'qwen' in folder.lower() or 'qwen' in original_csv.lower()
        is_gemma = 'gemma' in folder.lower() or 'gemma' in original_csv.lower()
        
        is_dd = 'dailydialog' in original_csv.lower() or 'dd' in python_script.lower()
        
        if os.path.exists(target_path):
            try:
                df = pd.read_csv(target_path)
                
                if 'metric_group' in df.columns:
                    window_df = df[df['metric_group'] == 'window_size'].copy()
                    
                    if not window_df.empty:
                        dataset_name = "DailyDialog" if is_dd else "MELD"
                        model_name = "Qwen" if is_qwen else "Gemma"
                        
                        if is_qwen and is_dd:
                            qwen_dd_dfs.append(window_df)
                        elif is_qwen and not is_dd:
                            qwen_meld_dfs.append(window_df)
                        elif is_gemma and is_dd:
                            gemma_dd_dfs.append(window_df)
                        elif is_gemma and not is_dd:
                            gemma_meld_dfs.append(window_df)
                            
                        print(f"Extracted {len(window_df)} rows from {target_path} [{model_name} | {dataset_name}]")
                else:
                    print(f"Warning: 'metric_group' column missing in {target_path}")
            except Exception as e:
                print(f"Error reading {target_path}: {e}")

    def save_csv(df_list, filename):
        if df_list:
            dataset = pd.concat(df_list, ignore_index=True)
            out_path = os.path.join(output_directory, filename)
            dataset.to_csv(out_path, index=False)
            print(f"Saved {len(dataset)} rows to: {out_path}")
        else:
            print(f"No data found for {filename}")

    save_csv(qwen_dd_dfs, "qwen_dailydialog_window_sizes.csv")
    save_csv(qwen_meld_dfs, "qwen_meld_window_sizes.csv")
    save_csv(gemma_dd_dfs, "gemma_dailydialog_window_sizes.csv")
    save_csv(gemma_meld_dfs, "gemma_meld_window_sizes.csv")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parent_dir = os.path.dirname(script_dir)
    
    os.chdir(parent_dir)
    
    extract_and_split_window_sizes(CONFIG_DATA, script_dir)