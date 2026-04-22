#!/bin/bash
set -e

# Define tests in the format: "tests_folder:sub_folder:script_name:csv_output_name:needs_move(true/false)"
TESTS=(
    # Zero-Shot Tests
    "qwen2_tests:zero_shot:qwen_zero_shot_structured_lemmatized.py:qwen_zero_shot_structured_lemmatized_results.csv:false"
    "qwen2_tests:zero_shot:qwen_zero_shot_structured.py:qwen_zero_shot_structured_results.csv:false"
    "qwen2_tests:zero_shot:qwen_zeroshot_lemmatized.py:qwen_zero_shot_lemmatized_results.csv:false"
    "qwen2_tests:zero_shot:qwen_zeroshot.py:qwen_zero_shot_results.csv:false"

    "qwen3_tests:zero_shot:qwen_zero_shot_structured_lemmatized.py:qwen3_zero_shot_structured_lemmatized_results.csv:false"
    "qwen3_tests:zero_shot:qwen_zero_shot_structured.py:qwen3_zero_shot_structured_results.csv:false"
    "qwen3_tests:zero_shot:qwen_zeroshot_lemmatized.py:qwen3_zero_shot_lemmatized_results.csv:false"
    "qwen3_tests:zero_shot:qwen_zeroshot.py:qwen3_zero_shot_results.csv:false"
    
    "gemma_tests:zero_shot:gemma_zero_shot_structured_lemmatized.py:gemma_zero_shot_structured_lemmatized_results.csv:false"
    "gemma_tests:zero_shot:gemma_zero_shot_structured.py:gemma_zero_shot_structured_results.csv:false"
    "gemma_tests:zero_shot:gemma_zeroshot_lemmatized.py:gemma_zero_shot_lemmatized_results.csv:false"
    "gemma_tests:zero_shot:gemma_zeroshot.py:gemma_zero_shot_results.csv:false"
    
    # Least-to-Most Tests
    "qwen2_tests:least_to_most:qwen_least_to_most.py:qwen_least_to_most_results.csv:true"
    "qwen2_tests:least_to_most:qwen_least_to_most_structured.py:qwen_least_to_most_structured_results.csv:true"
    "qwen2_tests:least_to_most:qwen_least_to_most_structured_lemmatized.py:qwen_least_to_most_structured_lemmatized_results.csv:true"
    "qwen2_tests:least_to_most:qwen_least_to_most_lemmatized.py:qwen_least_to_most_lemmatized_results.csv:true"

    "qwen3_tests:least_to_most:qwen_least_to_most.py:qwen3_least_to_most_results.csv:true"
    "qwen3_tests:least_to_most:qwen_least_to_most_structured.py:qwen3_least_to_most_structured_results.csv:true"
    "qwen3_tests:least_to_most:qwen_least_to_most_structured_lemmatized.py:qwen3_least_to_most_structured_lemmatized_results.csv:true"
    "qwen3_tests:least_to_most:qwen_least_to_most_lemmatized.py:qwen3_least_to_most_lemmatized_results.csv:true"

    "gemma_tests:least_to_most:gemma_least_to_most.py:gemma_least_to_most_results.csv:true"
    "gemma_tests:least_to_most:gemma_least_to_most_structured.py:gemma_least_to_most_structured_results.csv:true"
    "gemma_tests:least_to_most:gemma_least_to_most_structured_lemmatized.py:gemma_least_to_most_structured_lemmatized_results.csv:true"
    "gemma_tests:least_to_most:gemma_least_to_most_lemmatized.py:gemma_least_to_most_lemmatized_results.csv:true"
    
    # 2-Shot Tests
    "qwen2_tests:2_shot:qwen_2shot.py:qwen_2_shot_results.csv:true"
    "qwen2_tests:2_shot:qwen_2shot_structured.py:qwen_2_shot_structured_results.csv:true"
    "qwen2_tests:2_shot:qwen_2shot_structured_lemmatized.py:qwen_2_shot_structured_lemmatized_results.csv:true"
    "qwen2_tests:2_shot:qwen_2shot_lemmatized.py:qwen_2_shot_lemmatized_results.csv:true"

    "qwen3_tests:2_shot:qwen_2shot.py:qwen3_2_shot_results.csv:true"
    "qwen3_tests:2_shot:qwen_2shot_structured.py:qwen3_2_shot_structured_results.csv:true"
    "qwen3_tests:2_shot:qwen_2shot_structured_lemmatized.py:qwen3_2_shot_structured_lemmatized_results.csv:true"
    "qwen3_tests:2_shot:qwen_2shot_lemmatized.py:qwen3_2_shot_lemmatized_results.csv:true"

    "gemma_tests:2_shot:gemma_2shot.py:gemma_2_shot_results.csv:true"
    "gemma_tests:2_shot:gemma_2shot_structured.py:gemma_2_shot_structured_results.csv:true"
    "gemma_tests:2_shot:gemma_2shot_structured_lemmatized.py:gemma_2_shot_structured_lemmatized_results.csv:true"
    "gemma_tests:2_shot:gemma_2shot_lemmatized.py:gemma_2_shot_lemmatized_results.csv:true"

)

echo "Starting all evaluations..."

# Save the original directory so we can easily return to it
ROOT_DIR=$(pwd)

for TEST in "${TESTS[@]}"; do
    # Parse the colon-separated values
    IFS=':' read -r TESTS_FOLDER SUB_FOLDER SCRIPT CSV NEEDS_MOVE <<< "$TEST"
    
    echo "=================================================="
    echo "Starting execution for: $SCRIPT in /$TESTS_FOLDER/$SUB_FOLDER"
    echo "=================================================="
    
    # Navigate into the specific test directory
    cd "$TESTS_FOLDER/$SUB_FOLDER"
    
    # Ensure results directory exists just in case
    mkdir -p results
    
    # Run the inference script
    python "$SCRIPT"
    
    # Move the CSV if required
    if [ "$NEEDS_MOVE" = "true" ]; then
        echo "Moving $CSV to results folder..."
        mv "$CSV" results/
    fi
    
    # Run the evaluation script, passing the CSV name as a command line argument
    echo "Evaluating $CSV..."
    python "evaluating.py" "$CSV"
    
    # Head back to the root directory for the next iteration
    cd "$ROOT_DIR"
    
    echo "Finished $SCRIPT"
    echo ""
done

echo "=================================================="
echo "Completed all tests successfully!"
echo "=================================================="