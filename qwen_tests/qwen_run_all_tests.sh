
set -e

echo "Starting run_tests.sh..."
sh qwen_run_tests.sh

echo "Starting run_tests_structured.sh..."
sh qwen_run_tests_structured.sh

echo "Ran all tests successfully."