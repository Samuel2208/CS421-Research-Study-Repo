
set -e

echo "Starting 2_shot execution..."
cd 2_shot
python gemma_2shot.py
mv gemma_2_shot_results.csv results/
python "evaluating-gemma-meld.py"
cd ..

echo "Starting least_to_most execution..."
cd least_to_most
python gemma_least_to_most.py
mv gemma_least_to_most_results.csv results/
python "evaluating-gemma-meld.py"
cd ..

echo "Starting zero_shot execution..."
cd zero_shot
python gemma_zero_shot.py
python "evaluating-gemma-meld.py"

echo "Completed all tests"