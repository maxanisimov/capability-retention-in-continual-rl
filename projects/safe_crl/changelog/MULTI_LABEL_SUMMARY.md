# Multi-Label Support for _get_min_acc Function

## Summary

Successfully transformed the `_get_min_acc` function to support multi-label problems where multiple ground truth labels are valid for each data sample.

## Changes Made

### 1. Modified `src/interval_utils.py`

**Function Signature:**
```python
def _get_min_acc(bounded_model, X, y, soft=False, multi_label=False, delta=None):
```

**New Parameter:**
- `multi_label` (bool, default=False): When True, treats `y` as multi-label targets where each sample can have multiple correct labels

### 2. Added `src/verification/verify.py`

**New Functions:**
- `bound_multi_label_accuracy(logit_bounds, targets)`: Hard accuracy for multi-label case
- `bound_multi_label_soft_accuracy(logit_bounds, targets)`: Soft accuracy for multi-label case

## Multi-Label Format

For multi-label problems, targets should be formatted as:
```python
# Shape: (batch_size, max_labels)
# Use -1 for padding when samples have different numbers of labels
y_multi = torch.tensor([
    [1, 2, -1, -1],  # Sample 0: classes 1,2 are correct
    [0, -1, -1, -1], # Sample 1: only class 0 is correct  
    [2, 3, -1, -1]   # Sample 2: classes 2,3 are correct
])
```

## Usage Examples

### Single-Label (Original Behavior)
```python
# Traditional single-label classification
y_single = torch.tensor([1, 0, 2])
acc = _get_min_acc(bounded_model, X, y_single, multi_label=False)
# or simply:
acc = _get_min_acc(bounded_model, X, y_single)  # multi_label=False is default
```

### Multi-Label (New Functionality)
```python
# Multi-label classification with padded targets
y_multi = torch.tensor([
    [1, 2, -1, -1],  # Multiple valid labels per sample
    [0, -1, -1, -1],
    [2, 3, -1, -1]
])

# Hard accuracy (prediction must exactly match one of the valid labels)
acc_hard = _get_min_acc(bounded_model, X, y_multi, multi_label=True)

# Soft accuracy (probabilistic matching)
acc_soft = _get_min_acc(bounded_model, X, y_multi, soft=True, multi_label=True) 
```

## Key Features

1. **Backward Compatibility**: All existing code continues to work unchanged
2. **Flexible Target Format**: Supports variable number of labels per sample using -1 padding
3. **Both Hard and Soft Accuracy**: Works with both accuracy computation methods
4. **Interval Bounds**: Maintains certified accuracy bounds for multi-label scenarios

## Testing Results

- ✅ Single-label accuracy: 0.333
- ✅ Multi-label hard accuracy: 0.667  
- ✅ Multi-label soft accuracy: 0.515
- ✅ Backward compatibility maintained

## Applications

This enhancement enables the use of `_get_min_acc` for:
- Multi-label classification problems
- Scenarios where multiple ground truth answers are valid
- Ambiguous classification tasks with acceptable alternative labels
- Continual learning with label uncertainty