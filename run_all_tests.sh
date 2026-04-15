
set -e

echo "Starting qwen tests"
cd qwen_tests
sh qwen_run_all_tests.sh
cd ..
echo "Starting gemma tests"
cd gemma_tests
sh gemma_run_all_tests.sh

echo "Ran all tests successfully."