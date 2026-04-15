
set -e

echo "Starting 2_shot execution..."
cd 2_shot
python qwen_2shot_structured.py
mv qwen_2_shot_structured_results.csv results/
python "evaluating-qwen-meld copy.py"
cd ..

echo "Starting least_to_most execution..."
cd least_to_most
python qwen_least_to_most_structured.py
mv qwen_least_to_most_structured_results.csv results/
python "evaluating-qwen-meld copy.py"
cd ..

echo "Starting zero_shot execution..."
cd zero_shot
python qwen_zero_shot_structured.py
python "evaluating-qwen-meld copy.py"

echo "Completed all tests"