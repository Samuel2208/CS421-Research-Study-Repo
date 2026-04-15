
set -e

echo "Starting 2_shot execution..."
cd 2_shot
python qwen_2shot.py
mv qwen_2_shot_results.csv results/
python "evaluating-qwen-meld.py"
cd ..

echo "Starting least_to_most execution..."
cd least_to_most
python qwen_least_to_most.py
mv qwen_least_to_most_results.csv results/
python "evaluating-qwen-meld.py"
cd ..

echo "Starting zero_shot execution..."
cd zero_shot
python qwen_zeroshot.py
python "evaluating-qwen-meld.py"

echo "Completed all tests"