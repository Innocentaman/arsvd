"""
Wrapper script to run compression experiments and generate plots
"""

import argparse
import subprocess
import sys
from datetime import datetime


def run_command(cmd, description):
    """Run a shell command and print status"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ Error in {description}")
        sys.exit(1)
    print(f"✅ {description} completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description='Run Compression Experiments: ARSVD vs SVD'
    )

    # Dataset parameters
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to dataset directory')

    # Compression parameters
    parser.add_argument('--svd_ranks', type=str, default="100,150,200",
                        help='Comma-separated SVD ranks (e.g., "100,150,200")')
    parser.add_argument('--arsvd_taus', type=str, default="0.95,0.9,0.85,0.8",
                        help='Comma-separated ARSVD taus (e.g., "0.95,0.9,0.85,0.8")')

    # Model parameters
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to pre-trained model (optional)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train if no pre-trained model provided')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--fine_tune_epochs', type=int, default=0,
                        help='Number of epochs to fine-tune after compression (default: 0, disabled)')

    # Output parameters
    parser.add_argument('--out_dir', type=str, default='./compression_results',
                        help='Output directory for results')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("COMPRESSION PIPELINE: ARSVD vs SVD")
    print("="*80)
    print(f"Dataset: {args.data_root}")
    print(f"SVD Ranks: {args.svd_ranks}")
    print(f"ARSVD Taus: {args.arsvd_taus}")
    print(f"Fine-tune: {args.fine_tune_epochs} epochs")
    print(f"Output: {args.out_dir}")
    print("="*80)

    # Step 1: Run compression experiments
    exp_cmd = (
        f"python UNET/compression_experiment.py "
        f"--data_root {args.data_root} "
        f"--svd_ranks \"{args.svd_ranks}\" "
        f"--arsvd_taus \"{args.arsvd_taus}\" "
        f"--batch_size {args.batch_size} "
        f"--epochs {args.epochs} "
        f"--lr {args.lr} "
        f"--img_size {args.img_size} "
        f"--fine_tune_epochs {args.fine_tune_epochs} "
        f"--out_dir {args.out_dir}"
    )

    if args.model_path:
        exp_cmd += f" --model_path {args.model_path}"

    run_command(exp_cmd, "COMPRESSION EXPERIMENTS")

    # Step 2: Generate plots
    plot_cmd = f"python UNET/plot_results.py {args.out_dir}"
    run_command(plot_cmd, "GENERATING PLOTS")

    print("\n" + "="*80)
    print("🎉 COMPRESSION PIPELINE COMPLETED!")
    print("="*80)
    print(f"Results saved to: {args.out_dir}")
    print(f"  - compression_summary.csv")
    print(f"  - compression_results.json")
    print(f"  - plots/")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
