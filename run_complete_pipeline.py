"""
Complete Pipeline: Train U-Net + Compression Experiments
Runs training, evaluation, and ARSVD vs SVD compression experiments
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import argparse
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*80}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        sys.exit(1)
    print(f"✅ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Complete Pipeline: Train + Compression Experiments'
    )

    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset directory')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='ReduceLROnPlateau patience')

    # Compression parameters
    parser.add_argument('--svd_ranks', type=str, default="50,100,150",
                        help='Comma-separated SVD ranks (e.g., "50,100,150")')
    parser.add_argument('--arsvd_taus', type=str, default="0.95,0.9,0.85,0.8",
                        help='Comma-separated ARSVD taus (e.g., "0.95,0.9,0.85,0.8")')
    parser.add_argument('--fine_tune_epochs', type=int, default=0,
                        help='Number of epochs to fine-tune after compression (default: 0, disabled)')
    parser.add_argument('--run_compression', action='store_true',
                        help='Run compression experiments after training')

    # Output parameters
    parser.add_argument('--out_dir', type=str, default='./results',
                        help='Output directory for all results')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COMPLETE PIPELINE: TRAINING + COMPRESSION EXPERIMENTS")
    print("="*80)
    print(f"Dataset: {args.data_root}")
    print(f"Training Epochs: {args.epochs}")
    print(f"Compression: {'YES' if args.run_compression else 'NO (use --run_compression to enable)'}")
    if args.run_compression:
        print(f"  SVD Ranks: {args.svd_ranks}")
        print(f"  ARSVD Taus: {args.arsvd_taus}")
        print(f"  Fine-tune: {args.fine_tune_epochs} epochs")
    print(f"Output: {args.out_dir}")
    print("="*80)

    # Step 1: Training and Testing
    train_cmd = (
        f"python run_pipeline.py "
        f"--data_root {args.data_root} "
        f"--batch_size {args.batch_size} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr} "
        f"--img_size {args.img_size} "
        f"--patience {args.patience} "
        f"--lr_patience {args.lr_patience} "
        f"--seed {args.seed} "
        f"--out_dir {args.out_dir}/training"
    )

    run_command(train_cmd, "STEP 1/3: TRAINING AND EVALUATION")

    # Step 2: Compression Experiments (if enabled)
    if args.run_compression:
        compression_cmd = (
            f"python run_compression_pipeline.py "
            f"--data_root {args.data_root} "
            f"--model_path {args.out_dir}/training/model.h5 "
            f"--svd_ranks \"{args.svd_ranks}\" "
            f"--arsvd_taus \"{args.arsvd_taus}\" "
            f"--batch_size {args.batch_size} "
            f"--img_size {args.img_size} "
            f"--fine_tune_epochs {args.fine_tune_epochs} "
            f"--out_dir {args.out_dir}/compression"
        )

        run_command(compression_cmd, "STEP 2/3: COMPRESSION EXPERIMENTS (ARSVD vs SVD)")

        # Step 3: Generate comparison plots
        plot_cmd = f"python UNET/plot_results.py {args.out_dir}/compression"
        run_command(plot_cmd, "STEP 3/3: GENERATING COMPARISON PLOTS")
    else:
        print("\n" + "="*80)
        print("⏭️  Compression experiments skipped (use --run_compression to enable)")
        print("="*80)

    print("\n" + "="*80)
    print("🎉 COMPLETE PIPELINE FINISHED!")
    print("="*80)
    print(f"All results saved to: {args.out_dir}")
    if args.run_compression:
        print(f"\nTraining results:")
        print(f"  - {args.out_dir}/training/model.h5")
        print(f"  - {args.out_dir}/training/score.csv")
        print(f"  - {args.out_dir}/training/results/")
        print(f"\nCompression results:")
        print(f"  - {args.out_dir}/compression/compression_summary.csv")
        print(f"  - {args.out_dir}/compression/compression_results.json")
        print(f"  - {args.out_dir}/compression/plots/")
    else:
        print(f"\nTraining results:")
        print(f"  - {args.out_dir}/training/model.h5")
        print(f"  - {args.out_dir}/training/score.csv")
        print(f"  - {args.out_dir}/training/results/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
