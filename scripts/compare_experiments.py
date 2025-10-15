#!/usr/bin/env python3
"""
Experiment Comparison Script.

Compares results from two experiments:
- Experiment 1: Camera-only
- Experiment 2: Multimodal (Camera + LiDAR)

두 실험의 결과를 비교합니다:
- 실험 1: 카메라 전용
- 실험 2: 멀티모달 (카메라 + LiDAR)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ExperimentComparator:
    """Compare results from two experiments.
    
    두 실험의 결과를 비교합니다.
    
    Args:
        exp1_dir: Directory for experiment 1 (camera-only)
        exp2_dir: Directory for experiment 2 (multimodal)
        output_dir: Directory to save comparison results
    """
    
    def __init__(
        self,
        exp1_dir: str,
        exp2_dir: str,
        output_dir: str,
    ):
        self.exp1_dir = Path(exp1_dir)
        self.exp2_dir = Path(exp2_dir)
        self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_tensorboard_logs(self, log_dir: Path) -> Dict[str, List]:
        """Load training logs from TensorBoard.
        
        TensorBoard에서 학습 로그를 로드합니다.
        
        Args:
            log_dir: Directory containing TensorBoard logs
            
        Returns:
            Dictionary with scalar data
        """
        log_files = list(log_dir.glob("events.out.tfevents.*"))
        if not log_files:
            logger.warning(f"No TensorBoard logs found in {log_dir}")
            return {}
            
        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()
        
        data = {}
        
        # Load scalar tags
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            data[tag] = {
                "steps": [e.step for e in events],
                "values": [e.value for e in events],
            }
            
        return data
        
    def plot_training_curves(self):
        """Plot and compare training curves.
        
        학습 곡선을 플롯하고 비교합니다.
        """
        logger.info("Plotting training curves...")
        
        # Load logs
        exp1_logs = self.load_tensorboard_logs(self.exp1_dir / "logs")
        exp2_logs = self.load_tensorboard_logs(self.exp2_dir / "logs")
        
        if not exp1_logs or not exp2_logs:
            logger.warning("Missing logs, skipping training curve comparison")
            return
            
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot training loss
        if "loss/train" in exp1_logs and "loss/train" in exp2_logs:
            ax = axes[0]
            ax.plot(
                exp1_logs["loss/train"]["steps"],
                exp1_logs["loss/train"]["values"],
                label="Experiment 1: Camera-only",
                linewidth=2,
            )
            ax.plot(
                exp2_logs["loss/train"]["steps"],
                exp2_logs["loss/train"]["values"],
                label="Experiment 2: Multimodal",
                linewidth=2,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Training Loss")
            ax.set_title("Training Loss Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        # Plot validation loss
        if "loss/val" in exp1_logs and "loss/val" in exp2_logs:
            ax = axes[1]
            ax.plot(
                exp1_logs["loss/val"]["steps"],
                exp1_logs["loss/val"]["values"],
                label="Experiment 1: Camera-only",
                linewidth=2,
            )
            ax.plot(
                exp2_logs["loss/val"]["steps"],
                exp2_logs["loss/val"]["values"],
                label="Experiment 2: Multimodal",
                linewidth=2,
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation Loss")
            ax.set_title("Validation Loss Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        output_path = self.output_dir / "training_curves.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training curves to {output_path}")
        plt.close()
        
    def create_metrics_table(self) -> pd.DataFrame:
        """Create comparison table of metrics.
        
        메트릭 비교 테이블을 생성합니다.
        
        Returns:
            DataFrame with comparison metrics
        """
        logger.info("Creating metrics comparison table...")
        
        metrics_data = []
        
        for exp_name, exp_dir in [
            ("Camera-only", self.exp1_dir),
            ("Multimodal", self.exp2_dir),
        ]:
            # Try to load metrics from multiple possible locations
            metrics = None
            
            # Check for evaluation metrics
            eval_metrics_path = exp_dir / "metrics.json"
            if eval_metrics_path.exists():
                with open(eval_metrics_path, "r") as f:
                    metrics = json.load(f)
                    
            # If no evaluation metrics, try to get from config
            if metrics is None:
                config_path = exp_dir / "config.yaml"
                if config_path.exists():
                    # Create placeholder metrics
                    metrics = {
                        "note": "No evaluation metrics found",
                    }
                    
            if metrics:
                metrics["experiment"] = exp_name
                metrics_data.append(metrics)
                
        if not metrics_data:
            logger.warning("No metrics found for comparison")
            return pd.DataFrame()
            
        df = pd.DataFrame(metrics_data)
        
        # Save as CSV
        output_path = self.output_dir / "metrics_comparison.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved metrics comparison to {output_path}")
        
        return df
        
    def plot_metrics_comparison(self, df: pd.DataFrame):
        """Plot bar chart comparing metrics.
        
        메트릭을 비교하는 막대 그래프를 플롯합니다.
        
        Args:
            df: DataFrame with metrics
        """
        if df.empty:
            logger.warning("No metrics to plot")
            return
            
        logger.info("Plotting metrics comparison...")
        
        # Filter numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            logger.warning("No numeric metrics to plot")
            return
            
        # Create bar plots
        n_metrics = len(numeric_cols)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            experiments = df["experiment"].values
            values = df[col].values
            
            colors = ["#3498db", "#e74c3c"]
            ax.bar(experiments, values, color=colors[:len(experiments)])
            ax.set_ylabel(col)
            ax.set_title(f"{col} Comparison")
            ax.grid(True, alpha=0.3, axis="y")
            
        # Hide unused subplots
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis("off")
            
        plt.tight_layout()
        output_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metrics comparison plot to {output_path}")
        plt.close()
        
    def generate_summary_report(self, df: pd.DataFrame):
        """Generate text summary report.
        
        텍스트 요약 보고서를 생성합니다.
        
        Args:
            df: DataFrame with metrics
        """
        logger.info("Generating summary report...")
        
        report_lines = [
            "=" * 80,
            "EXPERIMENT COMPARISON REPORT",
            "실험 비교 보고서",
            "=" * 80,
            "",
        ]
        
        if not df.empty:
            report_lines.append("METRICS SUMMARY / 메트릭 요약")
            report_lines.append("-" * 80)
            report_lines.append(df.to_string(index=False))
            report_lines.append("")
            
        # Add conclusions section
        report_lines.extend([
            "",
            "CONCLUSIONS / 결론",
            "-" * 80,
            "",
            "Experiment 1 (Camera-only):",
            "  - Simpler architecture",
            "  - Faster inference",
            "  - Baseline performance",
            "",
            "Experiment 2 (Multimodal):",
            "  - More complex architecture",
            "  - Enhanced spatial awareness",
            "  - Potentially better robustness",
            "",
            "Recommendation:",
            "  Compare the validation losses and choose the model that best",
            "  fits your requirements for accuracy vs. computational cost.",
            "",
            "권장사항:",
            "  검증 손실을 비교하고 정확도 대 계산 비용에 대한 요구사항에",
            "  가장 적합한 모델을 선택하세요.",
            "",
            "=" * 80,
        ])
        
        # Save report
        output_path = self.output_dir / "comparison_report.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        logger.info(f"Saved summary report to {output_path}")
        
        # Print to console
        print("\n".join(report_lines))
        
    def compare(self):
        """Run full comparison.
        
        전체 비교를 실행합니다.
        """
        logger.info("Starting experiment comparison...")
        
        # Plot training curves
        self.plot_training_curves()
        
        # Create metrics table
        df = self.create_metrics_table()
        
        # Plot metrics comparison
        self.plot_metrics_comparison(df)
        
        # Generate summary report
        self.generate_summary_report(df)
        
        logger.info("✅ Comparison complete!")
        logger.info(f"Results saved to {self.output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--exp1-dir",
        type=str,
        required=True,
        help="Directory for experiment 1 (camera-only)"
    )
    
    parser.add_argument(
        "--exp2-dir",
        type=str,
        required=True,
        help="Directory for experiment 2 (multimodal)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for comparison results"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    try:
        comparator = ExperimentComparator(
            exp1_dir=args.exp1_dir,
            exp2_dir=args.exp2_dir,
            output_dir=args.output_dir,
        )
        
        comparator.compare()
        
    except KeyboardInterrupt:
        logger.info("Comparison interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise


if __name__ == "__main__":
    main()
