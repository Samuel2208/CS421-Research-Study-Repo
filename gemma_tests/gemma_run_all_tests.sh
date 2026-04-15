
set -e

echo "Starting run_tests.sh..."
sh gemma_run_tests.sh

echo "Starting run_tests_structured.sh..."
sh gemma_run_tests_structured.sh

echo "Ran all tests successfully."