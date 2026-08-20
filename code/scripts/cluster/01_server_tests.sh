#!/bin/bash
# Stage 0: Server Unit Tests
# Server-only script - DO NOT run locally

set -euo pipefail

echo "=== Stage 0: Server Unit Tests ==="
echo "Cycle: cycle-20260818-01"
echo "Experiment: v0.3 MVP"
echo "Date: $(date)"

# Check if running on server
if [[ "${RUN_CONTEXT:-}" != "cluster" ]]; then
    echo "ERROR: This script must run on the server (RUN_CONTEXT=cluster)"
    exit 1
fi

# Parse arguments
ENV_DIR=${1:-"/nfs4/ICLR/envs/mvp-v0.1"}
CODE_DIR=${2:-"/nfs4/ICLR/code/cycle-20260818-01"}
OUTPUT_DIR=${3:-"/nfs4/ICLR/work/cycle-20260818-01"}

echo "Environment: $ENV_DIR"
echo "Code: $CODE_DIR"
echo "Output: $OUTPUT_DIR"

# Activate environment
source "$ENV_DIR/bin/activate"

# Add code directory to PYTHONPATH
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

# Create test output directory
TEST_OUTPUT="$OUTPUT_DIR/tests"
mkdir -p "$TEST_OUTPUT"

echo ""
echo "Running unit tests..."

# Run pytest if tests exist
if [[ -d "$CODE_DIR/tests" ]]; then
    pytest "$CODE_DIR/tests" -v --tb=short | tee "$TEST_OUTPUT/test_results.txt"
    TEST_EXIT_CODE=${PIPESTATUS[0]}
else
    echo "No tests directory found, skipping pytest"
    TEST_EXIT_CODE=0
fi

# Manual critical tests
echo ""
echo "Running critical validation tests..."

# Test 1: L_PE numerical correctness
echo "Test 1: L_PE numerical correctness..."
python -c "
import torch
import sys
sys.path.insert(0, '$CODE_DIR')

from src.model.pe_loss import PELoss

# Test case: p_i should separate into two groups
p_i = torch.tensor([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])

pe_loss = PELoss(epsilon=1e-8)
loss, info = pe_loss(p_i)

print(f'Loss: {loss.item():.6f}')
print(f'e_hat_plus: [{info[\"e_hat_plus_0\"]:.4f}, {info[\"e_hat_plus_1\"]:.4f}]')
print(f'e_hat_minus: [{info[\"e_hat_minus_0\"]:.4f}, {info[\"e_hat_minus_1\"]:.4f}]')

# Check that e_hat_plus is close to [1, 0] and e_hat_minus is close to [0, 1]
assert info['e_hat_plus_0'] > 0.7, 'e_hat_plus[0] should be close to 1'
assert info['e_hat_plus_1'] < 0.3, 'e_hat_plus[1] should be close to 0'
assert info['e_hat_minus_0'] < 0.3, 'e_hat_minus[0] should be close to 0'
assert info['e_hat_minus_1'] > 0.7, 'e_hat_minus[1] should be close to 1'

print('✓ L_PE numerical correctness test passed')
"

# Test 2: Gradient path check
echo ""
echo "Test 2: Gradient path through p_i..."
python -c "
import torch
import sys
sys.path.insert(0, '$CODE_DIR')

from src.model.pe_loss import PELoss

p_i = torch.tensor([0.9, 0.8, 0.7, 0.3, 0.2, 0.1], requires_grad=True)

pe_loss = PELoss(epsilon=1e-8, detach_denominator=False)
loss, info = pe_loss(p_i)

loss.backward()

assert p_i.grad is not None, 'Gradient should flow through p_i'
assert not torch.isnan(p_i.grad).any(), 'Gradient should not be NaN'
assert not torch.isinf(p_i.grad).any(), 'Gradient should not be Inf'

print(f'p_i gradient: {p_i.grad}')
print('✓ Gradient path test passed')
"

# Test 3: Denominator stability
echo ""
echo "Test 3: Denominator stability with different epsilon..."
python -c "
import torch
import sys
sys.path.insert(0, '$CODE_DIR')

from src.model.pe_loss import PELoss

p_i = torch.tensor([0.5, 0.5, 0.5, 0.5])  # Uniform distribution

for eps in [1e-5, 1e-6, 1e-8]:
    pe_loss = PELoss(epsilon=eps)
    loss, info = pe_loss(p_i)

    print(f'epsilon={eps}: loss={loss.item():.6f}, sum_p={info[\"sum_p\"]:.4f}')

    assert not torch.isnan(loss), f'Loss is NaN with epsilon={eps}'
    assert not torch.isinf(loss), f'Loss is Inf with epsilon={eps}'
    assert info['sum_p'] > eps, f'Denominator too small with epsilon={eps}'

print('✓ Denominator stability test passed')
"

# Test 4: Label isolation (mock test)
echo ""
echo "Test 4: Label isolation check..."
python -c "
import sys
sys.path.insert(0, '$CODE_DIR')

# This would be tested with actual data in stage 1
# For now, just check the function exists
from src.data.data_utils import verify_label_isolation

print('✓ Label isolation function exists')
"

# Test 5: DPO loss sanity check
echo ""
echo "Test 5: DPO loss sanity check..."
python -c "
import torch
import sys
sys.path.insert(0, '$CODE_DIR')

from src.model.dpo_loss import DPOLoss

# Chosen should be preferred over rejected
policy_chosen_logps = torch.tensor([0.5, 0.6])
policy_rejected_logps = torch.tensor([0.3, 0.4])
reference_chosen_logps = torch.tensor([0.4, 0.5])
reference_rejected_logps = torch.tensor([0.4, 0.5])

dpo_loss = DPOLoss(beta=0.1)
loss, info = dpo_loss(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps
)

print(f'DPO loss: {loss.item():.6f}')
print(f'Accuracy: {info[\"accuracy\"]:.4f}')

assert not torch.isnan(loss), 'DPO loss is NaN'
assert not torch.isinf(loss), 'DPO loss is Inf'
assert info['accuracy'] == 1.0, 'Should correctly prefer chosen'

print('✓ DPO loss sanity check passed')
"

echo ""
echo "=== All Tests Passed ==="
echo "Test results saved to: $TEST_OUTPUT/test_results.txt"
echo ""
echo "Next: Run Stage 1 (02_prepare_data.sh)"

exit $TEST_EXIT_CODE
