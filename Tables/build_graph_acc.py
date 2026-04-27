import pandas as pd
import matplotlib.pyplot as plt

def clean_source_name(source_str):
    source_str = str(source_str)
    
    words_to_remove = ['dailydialog', 'results']
    
    parts = source_str.split('_')
    cleaned_parts = [part for part in parts if part not in words_to_remove]
    
    return '_'.join(cleaned_parts)

def create_accuracy_comparison_graph(csv_filepath):
    df = pd.read_csv(csv_filepath)
    
    target_tables = ["gemma_dailydialog", "qwen_dailydialog"]
    target_windows = [1, 3, 5, 7, 9, 11]
    
    filtered_df = df[
        (df['Table'].isin(target_tables)) & 
        (df['Window_Size'].isin(target_windows))
    ].copy()
    
    filtered_df['Accuracy_Source_Clean'] = filtered_df['Accuracy_Source'].apply(clean_source_name)
    
    plt.figure(figsize=(13, 8))
    
    colors = {"gemma_dailydialog": "blue", "qwen_dailydialog": "green"}
    markers = {"gemma_dailydialog": "o", "qwen_dailydialog": "s"}
    
    for table in target_tables:
        subset = filtered_df[filtered_df['Table'] == table].sort_values(by='Window_Size')
        
        if subset.empty:
            continue
            
        plt.plot(
            subset['Window_Size'], 
            subset['Best_Accuracy'], 
            marker=markers[table], 
            color=colors[table], 
            label=table, 
            linewidth=2, 
            markersize=8
        )
        
        for _, row in subset.iterrows():
            plt.annotate(
                row['Accuracy_Source_Clean'],
                xy=(row['Window_Size'], row['Best_Accuracy']),
                xytext=(0, 12),
                textcoords="offset points",
                ha='center',
                fontsize=9,
                color=colors[table],
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7) 
            )

    plt.title('Best Accuracy Comparison: Gemma vs Qwen', fontsize=15, fontweight='bold')
    plt.xlabel('Window Size', fontsize=12)
    plt.ylabel('Best Accuracy', fontsize=12)
    
    plt.xticks(target_windows)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title='Model / Table', loc='best')
    plt.tight_layout()
    
    plt.savefig('accuracy_comparison_graph.png', dpi=300)
    print("Graph saved as 'accuracy_comparison_graph.png'")
    plt.show()

if __name__ == "__main__":
    create_accuracy_comparison_graph('best_window_size_metrics.csv')