# ARSVD vs SVD Compression Experiments

This document describes the implementation and usage of **Adaptive-Rank SVD (ARSVD)** and **standard SVD** compression experiments for the Brain Tumor Segmentation U-Net model.

## Overview

The compression experiments compare two methods:
1. **ARSVD** - Entropy-guided adaptive rank selection per layer
2. **SVD** - Fixed-rank compression across all layers

## File Structure

```
UNET/
├── compression_utils.py         # ARSVD and SVD algorithms
├── compression_experiment.py    # Main experiment runner
├── plot_results.py              # Visualization module
└── metrics.py                   # Evaluation metrics

run_compression_pipeline.py      # Wrapper script
```

## Key Features

### Compression Algorithms

**ARSVD (Adaptive-Rank SVD)**
- Uses spectral entropy to adaptively select rank per layer
- Tau parameter controls information retention (0.8-0.95)
- Higher tau = more information retained = higher rank
- Layer-specific compression based on information content

**Standard SVD**
- Fixed rank across all Conv2D layers
- Ranks tested: 50, 100, 150, 200, etc.
- Uniform compression strategy
- Simpler but less adaptive

### Metrics Tracked

For each experiment, we calculate:
- **Dice Score** - Overlap between prediction and ground truth
- **IoU (Jaccard)** - Intersection over Union
- **F1 Score** - Harmonic mean of precision and recall
- **Accuracy** - Pixel-wise accuracy

### Outputs

The pipeline generates:
1. **compression_summary.csv** - All results in table format
2. **compression_results.json** - Detailed experiment data
3. **plots/** directory with:
   - IoU vs SVD Ranks
   - IoU vs ARSVD Tau values
   - Per-layer ranks for each method
   - Comprehensive comparison summary

## Usage

### Command Line

```bash
python run_compression_pipeline.py \
  --data_root /path/to/dataset \
  --model_path /path/to/trained_model.h5 \
  --svd_ranks "50,100,150,200" \
  --arsvd_taus "0.95,0.9,0.85,0.8" \
  --batch_size 16 \
  --img_size 256 \
  --out_dir ./compression_results
```

### Parameters

- `--data_root`: Path to dataset directory (required)
- `--model_path`: Path to pre-trained model (optional - will train if not provided)
- `--svd_ranks`: Comma-separated ranks for SVD (default: "100,150,200")
- `--arsvd_taus`: Comma-separated taus for ARSVD (default: "0.95,0.9,0.85,0.8")
- `--epochs`: Training epochs if no model provided (default: 50)
- `--batch_size`: Batch size for evaluation (default: 16)
- `--img_size`: Image size (default: 256)
- `--out_dir`: Output directory (default: ./compression_results)

### Google Colab

The notebook `Brain_Tumor_Segmentation_Colab.ipynb` includes compression experiment cells:

1. Run main training pipeline first
2. Run compression experiments with pre-trained model
3. View comparison plots and results
4. Results automatically saved to Google Drive

## Example Results

### Expected Output Structure

```
compression_results/
├── compression_summary.csv
├── compression_results.json
├── base_model.h5
└── plots/
    ├── iou_vs_svd_ranks.png
    ├── iou_vs_arsvd_taus.png
    ├── per_layer_ranks_arsvd.png
    ├── per_layer_ranks_svd.png
    └── comparison_summary.png
```

### Summary Table Format

| experiment    | method | param_value | dice  | iou   | f1    | accuracy |
|---------------|--------|-------------|-------|-------|-------|----------|
| base_model    | none   | None        | 0.850 | 0.750 | 0.850 | 0.990    |
| SVD_rank_100  | svd    | 100         | 0.840 | 0.735 | 0.840 | 0.988    |
| ARSVD_tau_0.9 | arsvd  | 0.9         | 0.848 | 0.748 | 0.848 | 0.989    |

## Technical Details

### Compression Process

1. **Load Trained Model**: U-Net with pre-trained weights
2. **Identify Conv2D Layers**: All convolutional layers are compressible
3. **Apply Compression**:
   - Reshape kernel to 2D matrix (in_channels * h * w, out_channels)
   - Compute SVD
   - Apply ARSVD or SVD truncation
   - Reconstruct compressed kernel
4. **Evaluate**: Test compressed model on validation set
5. **Compare**: Aggregate results across all experiments

### ARSVD Algorithm

```python
def arsvd_compress(W, tau):
    # 1. Compute SVD
    U, S, Vt = svd(W)

    # 2. Normalize singular values
    p = S / sum(S)

    # 3. Compute total entropy
    H_total = -sum(p * log(p))

    # 4. Find smallest k where partial_entropy >= tau * H_total
    H_partial = 0
    for i in range(len(S)):
        H_partial -= p[i] * log(p[i])
        if H_partial >= tau * H_total:
            k = i + 1
            break

    # 5. Truncate and reconstruct
    W_compressed = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]

    return W_compressed, k
```

## Research Context

This implementation is based on the ARSVD paper:
```
"Low-Rank Matrix Approximation for Neural Network Compression"
Kalyan Cherukuri, Aarav Lala
arXiv:2504.20078v2 [cs.LG] 11 May 2025
```

### Key Innovations from Paper

1. **Entropy-guided selection**: Adaptively chooses rank based on information content
2. **Per-layer optimization**: Each layer retains different amount of information
3. **Improved compression**: 20-30% better parameter reduction vs fixed-rank SVD
4. **Accuracy preservation**: Maintains or improves performance

## Future Enhancements

Possible extensions:
1. **Fine-tuning after compression**: Retrain compressed models
2. **Layer-wise sensitivity analysis**: Identify which layers can be compressed more
3. **Dynamic tau selection**: Automatically choose optimal tau per layer
4. **Structured compression**: Preserve network structure during compression
5. **Multi-stage compression**: Progressive compression with intermediate fine-tuning

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or use smaller ranks
2. **No improvement**: Ensure base model is well-trained before compression
3. **Slow evaluation**: Reduce number of test samples or ranks/taus to test

### Tips for Best Results

1. Train base model to convergence first (50+ epochs)
2. Use diverse tau values (0.8-0.95) for ARSVD
3. Test ranks around 50-200 for SVD depending on layer size
4. Always compare with baseline (uncompressed) model
5. Use multiple random seeds for robustness

## Citation

If you use this code or the ARSVD method, please cite:
```bibtex
@article{cherukuri2025arsvd,
  title={Low-Rank Matrix Approximation for Neural Network Compression},
  author={Cherukuri, Kalyan and Lala, Aarav},
  journal={arXiv preprint arXiv:2504.20078},
  year={2025}
}
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
