#!/bin/bash

set -e  # Exit on error

# Color output for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Running workflow tests..."
echo "========================================="

# Find all test configuration files
TEST_CONFIGS=$(find configs/weather_generators/tests -name "*.yaml" -type f | sort)

if [ -z "$TEST_CONFIGS" ]; then
    echo -e "${RED}No test configuration files found in config/tests/${NC}"
    exit 1
fi

# Count total tests
TOTAL_TESTS=$(echo "$TEST_CONFIGS" | wc -l)
PASSED=0
FAILED=0

# Array to store failed testsq
declare -a FAILED_TESTS

# Clean previous test results (optional)
# rm -rf results/*/

# Disable exit-on-error for the test loop
set +e

# Run each test configuration
for config_file in $TEST_CONFIGS; do
    test_name=$(basename "$config_file" .yaml)
    echo ""
    echo -e "${YELLOW}Testing: $test_name${NC}"
    echo "Config: $config_file"
    echo "-----------------------------------------"

    if snakemake \
        --configfile "$config_file" \
        --profile test; then  # --profile test; then
        echo -e "${GREEN}✓ $test_name PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ $test_name FAILED${NC}"
        ((FAILED++))
        FAILED_TESTS+=("$test_name")
    fi
done

# Re-enable exit-on-error
set -e

# Summary
echo ""
echo "========================================="
echo "Test Summary"
echo "========================================="
echo "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo -e "${RED}Failed tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  - $test"
    done
    exit 1
else
    echo ""
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
