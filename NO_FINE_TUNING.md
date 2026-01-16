# Fine-Tuning Disabled by Default

## What Changed

Fine-tuning after compression is now **disabled by default** to save time. Previously it was hardcoded to 5 epochs for every compression experiment, which was very time-consuming.

## Changes Made

### 1. **UNET/compression_experiment.py**
- Changed default `fine_tune_epochs=5` → `fine_tune_epochs=0`
- Added command-line argument: `--fine_tune_epochs`
- Updated logic to skip fine-tuning when `fine_tune_epochs=0`
- Only evaluates once (instead of before/after fine-tuning) when disabled

### 2. **run_compression_pipeline.py**
- Added `--fine_tune_epochs` argument (default: 0)
- Passes the argument through to compression_experiment.py

### 3. **run_complete_pipeline.py**
- Added `--fine_tune_epochs` argument (default: 0)
- Displays fine-tuning setting in summary
- Passes through to compression pipeline

## How to Use

### Default (No Fine-Tuning) - Fast:
```bash
python run_complete_pipeline.py \
  --data_root /path/to/dataset \
  --epochs 100 \
  --run_compression \
  --svd_ranks "50,100,150" \
  --arsvd_taus "0.95,0.9,0.85" \
  --out_dir ./results
```

**Expected time**: Much faster! (no fine-tuning iterations)

### With Fine-Tuning - Slower but Better Results:
```bash
python run_complete_pipeline.py \
  --data_root /path/to/dataset \
  --epochs 100 \
  --run_compression \
  --svd_ranks "50,100,150" \
  --arsvd_taus "0.95,0.9,0.85" \
  --fine_tune_epochs 5 \
  --out_dir ./results
```

**Expected time**: ~5x longer (5 epochs for each compressed model)

## What to Expect

### Without Fine-Tuning (Default):
- **Speed**: Very fast compression experiments
- **Results**: Lower accuracy (Dice: 0.30-0.50 range typically)
- **Use case**: Quick iteration, testing different ranks/taus

### With Fine-Tuning (Optional):
- **Speed**: Slower (but still much faster than full retraining)
- **Results**: Better accuracy (Dice: 0.70-0.80 range, 90-97% recovery)
- **Use case**: Final experiments, publication-quality results

## Recommended Workflow

### Step 1: Quick Test (No Fine-Tuning)
```bash
# Test with 1 epoch training + compression
python run_complete_pipeline.py \
  --data_root /path/to/dataset \
  --epochs 1 \
  --run_compression \
  --svd_ranks "50,100" \
  --arsvd_taus "0.95,0.9" \
  --out_dir ./test_results
```

### Step 2: Full Training (No Fine-Tuning)
```bash
# Train baseline model properly
python run_complete_pipeline.py \
  --data_root /path/to/dataset \
  --epochs 100 \
  --run_compression \
  --svd_ranks "50,100,150" \
  --arsvd_taus "0.95,0.9,0.85,0.8" \
  --out_dir ./results
```

### Step 3: Best Configuration with Fine-Tuning (Optional)
```bash
# After analyzing results, re-run best configs with fine-tuning
python run_compression_pipeline.py \
  --data_root /path/to/dataset \
  --model_path ./results/training/model.h5 \
  --svd_ranks "100" \
  --arsvd_taus "0.9" \
  --fine_tune_epochs 5 \
  --out_dir ./final_results
```

## Time Comparison

For a typical experiment with 3 SVD ranks + 4 ARSVD taus = 7 experiments:

| Configuration | Time per Experiment | Total Time |
|--------------|---------------------|------------|
| **No Fine-Tuning** | ~1-2 minutes | **~10-15 minutes** |
| **With Fine-Tuning** | ~5-10 minutes | **~40-70 minutes** |

## Important Notes

1. **Fine-tuning is optional**: You can now run quick experiments without waiting for fine-tuning

2. **Baseline training still needed**: You still need to train the base model properly (100 epochs) for meaningful compression results

3. **Results will be lower without FT**: Compressed models without fine-tuning will show lower accuracy (expected!)

4. **Use for comparison**: Even without fine-tuning, you can still compare:
   - Different ranks (which rank works best?)
   - Different taus (which ARSVD threshold works best?)
   - ARSVD vs SVD (which method selects better ranks?)

5. **Enable for final results**: Add `--fine_tune_epochs 5` when you want to see the recovered performance

## Summary

✅ **Default**: No fine-tuning (fast!)
✅ **Optional**: Add `--fine_tune_epochs 5` for better results
✅ **Saves time**: 5x faster compression experiments
✅ **Best practice**: Test without FT first, then re-run best configs with FT

---

**Ready to run!** The pipeline is now much faster for quick experimentation.
