import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import logging
import math
from pathlib import Path
from data_process import data_path

plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


class ModelComparison:
    def __init__(self, results_path=None):
        """Initialize model comparison class"""
        if results_path is None:
            results_path = data_path + "results/all_models_summary.json"

        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = Path(data_path) / "analysis_plots"
        self.output_dir.mkdir(exist_ok=True)

    def load_results(self):
        """Load all model results - 支持baseline和改进模型"""
        results = {}

        # 尝试加载主要汇总文件（包括baseline和改进版本）
        summary_files = [
            self.results_path,
            data_path + "results/all_models_summary_with_baseline.json",
            data_path + "results/all_models_summary_with_scheduler.json",
            data_path + "results/all_models_summary.json",
        ]

        for file_path in summary_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    loaded_results = json.load(f)
                    results.update(loaded_results)
                    print(f"✅ 成功加载: {file_path}")
                    break
            except FileNotFoundError:
                continue

        if not results:
            print(
                f"Warning: No summary file found, trying to load individual result files..."
            )
            # 尝试加载单独的结果文件（包括baseline）
            results_dir = Path(data_path) / "results"
            if results_dir.exists():
                for result_file in results_dir.glob("results_*.json"):
                    try:
                        with open(result_file, "r", encoding="utf-8") as f:
                            result = json.load(f)
                            model_type = result.get("model_type", result_file.stem)
                            results[model_type] = result
                            print(f"✅ 加载单独文件: {result_file}")
                    except Exception as e:
                        print(f"❌ 加载失败 {result_file}: {e}")

        return results

    def plot_mmoe_detailed_training_curves(self):
        """专门为MMOE模型绘制详细的三阶段训练曲线"""
        # 找到MMOE模型
        mmoe_results = None
        for model_type, results in self.results.items():
            if (
                "MMoE" in model_type
                or "mmoe" in model_type.lower()
                or "TwoStage" in model_type
            ):
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMOE model results not found")
            return

        print("🔍 Found MMOE model, creating detailed training analysis...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(
            "MMOE Three-Stage Training Analysis (90 Epochs Total)",
            fontsize=16,
            fontweight="bold",
        )

        history = mmoe_results["training_history"]

        # 检查训练历史数据
        if "train_losses" not in history or not history["train_losses"]:
            print("❌ MMOE training history data not found")
            return

        total_epochs = len(history["train_losses"])
        print(f"📊 MMOE total training epochs: {total_epochs}")

        # 分割三个阶段 (假设每个阶段30轮)
        stage_epochs = 30
        stages = {
            "Stage 1: Temporal Modeling": (0, stage_epochs),
            "Stage 2: CF Modeling": (stage_epochs, 2 * stage_epochs),
            "Stage 3: MMoE Fusion": (
                2 * stage_epochs,
                min(total_epochs, 3 * stage_epochs),
            ),
        }

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # 红、青、蓝

        # 绘制每个阶段的训练损失
        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            ax = axes[idx, 0]

            if end > start and end <= len(history["train_losses"]):
                stage_epochs_range = range(start + 1, end + 1)
                stage_train_losses = history["train_losses"][start:end]

                ax.plot(
                    stage_epochs_range,
                    stage_train_losses,
                    color=colors[idx],
                    linewidth=2,
                    label="Training Loss",
                )

                # 如果有验证损失，也绘制
                if (
                    "val_losses" in history
                    and history["val_losses"]
                    and end <= len(history["val_losses"])
                ):
                    stage_val_losses = history["val_losses"][start:end]
                    ax.plot(
                        stage_epochs_range,
                        stage_val_losses,
                        color=colors[idx],
                        linewidth=2,
                        linestyle="--",
                        label="Validation Loss",
                        alpha=0.8,
                    )

                ax.set_title(f"{stage_name}\n(Epochs {start+1}-{end})")
                ax.set_xlabel("Training Epochs")
                ax.set_ylabel("MSE Loss")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # 添加统计信息
                final_loss = stage_train_losses[-1] if stage_train_losses else 0
                min_loss = min(stage_train_losses) if stage_train_losses else 0
                ax.text(
                    0.05,
                    0.95,
                    f"Final: {final_loss:.4f}\nMin: {min_loss:.4f}",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        # 绘制每个阶段的学习率变化
        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            ax = axes[idx, 1]

            if (
                "learning_rates" in history
                and history["learning_rates"]
                and end > start
                and end <= len(history["learning_rates"])
            ):

                stage_epochs_range = range(start + 1, end + 1)
                stage_learning_rates = history["learning_rates"][start:end]

                ax.plot(
                    stage_epochs_range,
                    stage_learning_rates,
                    color=colors[idx],
                    linewidth=2,
                    marker="o",
                    markersize=3,
                )

                ax.set_title(f"{stage_name}\nLearning Rate Evolution")
                ax.set_xlabel("Training Epochs")
                ax.set_ylabel("Learning Rate")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.3)

                # 添加学习率变化信息
                initial_lr = stage_learning_rates[0] if stage_learning_rates else 0
                final_lr = stage_learning_rates[-1] if stage_learning_rates else 0
                lr_change = (
                    ((initial_lr - final_lr) / initial_lr * 100)
                    if initial_lr > 0
                    else 0
                )

                ax.text(
                    0.05,
                    0.95,
                    f"Initial: {initial_lr:.6f}\nFinal: {final_lr:.6f}\nDecay: {lr_change:.1f}%",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "mmoe_detailed_stage_training.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"✅ MMOE detailed training curves saved to: {self.output_dir / 'mmoe_detailed_stage_training.png'}"
        )

    def plot_training_curves_with_mmoe_comparison(self):
        """绘制训练曲线对比，特别处理MMOE的90轮训练 - 显示所有模型"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))  # 增加图像宽度
        fig.suptitle(
            "Complete Training Process Comparison (All Models)",
            fontsize=16,
            fontweight="bold",
        )

        # 使用更多颜色，确保每个模型都有独特颜色
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        # 分别处理MMOE和其他模型，并收集所有模型信息
        mmoe_data = None
        other_models = []
        all_models_info = []

        for model_type, results in self.results.items():
            model_info = {
                "type": model_type,
                "results": results,
                "name": results["model_name"],
                "is_mmoe": False,
            }

            if (
                "MMoE" in model_type
                or "mmoe" in model_type.lower()
                or "TwoStage" in model_type
            ):
                mmoe_data = (model_type, results)
                model_info["is_mmoe"] = True
            else:
                other_models.append((model_type, results))

            all_models_info.append(model_info)

        print(f"🔍 Found {len(all_models_info)} models for comparison:")
        for model_info in all_models_info:
            model_type_label = (
                " (MMOE - 90 epochs)" if model_info["is_mmoe"] else " (30 epochs)"
            )
            print(f"  - {model_info['name']}{model_type_label}")

        # 1. 训练损失对比 - 显示所有模型
        ax = axes[0, 0]

        # 绘制其他模型（30轮）
        for idx, (model_type, results) in enumerate(other_models):
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs,
                    history["train_losses"],
                    label=f"{results['model_name']} (30ep)",
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                )

        # 绘制MMOE（90轮），使用特殊样式
        if mmoe_data:
            _, mmoe_results = mmoe_data
            history = mmoe_results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)

                # MMOE用粗线和特殊颜色
                ax.plot(
                    epochs,
                    history["train_losses"],
                    label=f"{mmoe_results['model_name']} (90ep)",
                    color="#FF1744",
                    linewidth=3,
                    alpha=0.9,
                )

                # 添加阶段分隔线
                stage_boundaries = [30, 60]
                for boundary in stage_boundaries:
                    if boundary < len(history["train_losses"]):
                        ax.axvline(
                            x=boundary, color="#FF1744", linestyle=":", alpha=0.5
                        )

                # 标注三个阶段
                stage_labels = ["Temporal", "CF", "MMoE"]
                stage_positions = [15, 45, 75]
                for pos, label in zip(stage_positions, stage_labels):
                    if pos < len(history["train_losses"]):
                        ax.text(
                            pos,
                            max(history["train_losses"]) * 0.9,
                            label,
                            ha="center",
                            va="center",
                            bbox=dict(
                                boxstyle="round,pad=0.3", facecolor="white", alpha=0.7
                            ),
                            fontsize=9,
                            color="#FF1744",
                        )

        ax.set_title("Training Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. 验证损失对比 - 显示所有模型
        ax = axes[0, 1]

        # 绘制其他模型的验证损失
        for idx, (model_type, results) in enumerate(other_models):
            history = results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs,
                    history["val_losses"],
                    label=f"{results['model_name']} (30ep)",
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                )

        # 绘制MMOE验证损失
        if mmoe_data:
            _, mmoe_results = mmoe_data
            history = mmoe_results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                ax.plot(
                    epochs,
                    history["val_losses"],
                    label=f"{mmoe_results['model_name']} (90ep)",
                    color="#FF1744",
                    linewidth=3,
                    alpha=0.9,
                )

                # 添加阶段分隔线
                for boundary in [30, 60]:
                    if boundary < len(history["val_losses"]):
                        ax.axvline(
                            x=boundary, color="#FF1744", linestyle=":", alpha=0.5
                        )

        ax.set_title("Validation Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. 学习率演化对比 - 显示所有模型
        ax = axes[1, 0]

        # 绘制其他模型的学习率
        for idx, (model_type, results) in enumerate(other_models):
            history = results["training_history"]
            if "learning_rates" in history and history["learning_rates"]:
                epochs = range(1, len(history["learning_rates"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs,
                    history["learning_rates"],
                    label=f"{results['model_name']}",
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                )

        # 绘制MMOE学习率
        if mmoe_data:
            _, mmoe_results = mmoe_data
            history = mmoe_results["training_history"]
            if "learning_rates" in history and history["learning_rates"]:
                epochs = range(1, len(history["learning_rates"]) + 1)
                ax.plot(
                    epochs,
                    history["learning_rates"],
                    label=f"{mmoe_results['model_name']}",
                    color="#FF1744",
                    linewidth=3,
                    alpha=0.9,
                )

                # 添加阶段分隔线
                for boundary in [30, 60]:
                    if boundary < len(history["learning_rates"]):
                        ax.axvline(
                            x=boundary, color="#FF1744", linestyle=":", alpha=0.5
                        )

        ax.set_title("Learning Rate Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. 最终30轮公平对比（所有模型标准化到30轮）
        ax = axes[1, 1]

        # 绘制其他模型（原始30轮）
        for idx, (model_type, results) in enumerate(other_models):
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, min(31, len(history["train_losses"]) + 1))
                train_losses = history["train_losses"][:30]
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs,
                    train_losses,
                    label=f"{results['model_name']}",
                    color=color,
                    linewidth=2,
                    alpha=0.8,
                )

        # 绘制MMOE的最后30轮（61-90轮）
        if mmoe_data:
            _, mmoe_results = mmoe_data
            history = mmoe_results["training_history"]
            if "train_losses" in history and len(history["train_losses"]) >= 90:
                # 取最后30轮 (61-90)
                mmoe_last_30 = history["train_losses"][60:90]
                epochs = range(1, len(mmoe_last_30) + 1)
                ax.plot(
                    epochs,
                    mmoe_last_30,
                    label=f"{mmoe_results['model_name']} (Stage 3)",
                    color="#FF1744",
                    linewidth=3,
                    alpha=0.9,
                )

        ax.set_title(
            "Fair 30-Epoch Comparison\n(MMOE: Final Stage vs Others: Full Training)"
        )
        ax.set_xlabel("Epoch (within 30-epoch window)")
        ax.set_ylabel("Training Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_comparison_all_models_with_mmoe.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"✅ Complete model training comparison saved to: {self.output_dir / 'training_comparison_all_models_with_mmoe.png'}"
        )

    def plot_mmoe_stage_performance_analysis(self):
        """分析MMOE各阶段的性能变化"""
        # 找到MMOE模型
        mmoe_results = None
        for model_type, results in self.results.items():
            if (
                "MMoE" in model_type
                or "mmoe" in model_type.lower()
                or "TwoStage" in model_type
            ):
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMOE model results not found")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "MMOE Stage-by-Stage Performance Analysis", fontsize=16, fontweight="bold"
        )

        history = mmoe_results["training_history"]

        if "train_losses" not in history or len(history["train_losses"]) < 90:
            print("❌ MMOE training history insufficient for stage analysis")
            return

        # 定义三个阶段
        stages = {
            "Stage 1: Temporal Modeling": (0, 30),
            "Stage 2: CF Modeling": (30, 60),
            "Stage 3: MMoE Fusion": (60, 90),
        }

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # 1. 各阶段损失对比
        ax = axes[0, 0]
        stage_avg_losses = []
        stage_names = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                avg_loss = np.mean(stage_losses)
                stage_avg_losses.append(avg_loss)
                stage_names.append(stage_name.split(":")[0])  # 只取Stage部分

        bars = ax.bar(stage_names, stage_avg_losses, color=colors[: len(stage_names)])
        ax.set_title("Average Training Loss by Stage")
        ax.set_ylabel("Average MSE Loss")

        for bar, loss in zip(bars, stage_avg_losses):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{loss:.4f}",
                ha="center",
                va="bottom",
            )

        # 2. 损失减少量分析
        ax = axes[0, 1]
        loss_reductions = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                initial_loss = stage_losses[0]
                final_loss = stage_losses[-1]
                reduction = (
                    ((initial_loss - final_loss) / initial_loss * 100)
                    if initial_loss > 0
                    else 0
                )
                loss_reductions.append(reduction)

        bars = ax.bar(stage_names, loss_reductions, color=colors[: len(stage_names)])
        ax.set_title("Loss Reduction by Stage")
        ax.set_ylabel("Loss Reduction (%)")

        for bar, reduction in zip(bars, loss_reductions):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{reduction:.1f}%",
                ha="center",
                va="bottom",
            )

        # 3. 学习率衰减分析
        ax = axes[0, 2]
        if "learning_rates" in history and len(history["learning_rates"]) >= 90:
            lr_changes = []

            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                stage_lrs = history["learning_rates"][start:end]
                if stage_lrs:
                    initial_lr = stage_lrs[0]
                    final_lr = stage_lrs[-1]
                    change = (
                        ((initial_lr - final_lr) / initial_lr * 100)
                        if initial_lr > 0
                        else 0
                    )
                    lr_changes.append(change)

            bars = ax.bar(stage_names, lr_changes, color=colors[: len(stage_names)])
            ax.set_title("Learning Rate Decay by Stage")
            ax.set_ylabel("LR Decay (%)")

            for bar, change in zip(bars, lr_changes):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + height * 0.01,
                    f"{change:.1f}%",
                    ha="center",
                    va="bottom",
                )

        # 4. 训练稳定性分析（损失方差）
        ax = axes[1, 0]
        loss_stds = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                loss_std = np.std(stage_losses)
                loss_stds.append(loss_std)

        bars = ax.bar(stage_names, loss_stds, color=colors[: len(stage_names)])
        ax.set_title("Training Stability by Stage")
        ax.set_ylabel("Loss Standard Deviation")

        for bar, std in zip(bars, loss_stds):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{std:.4f}",
                ha="center",
                va="bottom",
            )

        # 5. 阶段间损失变化
        ax = axes[1, 1]
        stage_endpoints = [30, 60, 90]
        endpoint_losses = [
            history["train_losses"][ep - 1]
            for ep in stage_endpoints
            if ep <= len(history["train_losses"])
        ]

        x_pos = range(len(endpoint_losses))
        ax.plot(
            x_pos, endpoint_losses, "o-", linewidth=3, markersize=8, color="#FF1744"
        )
        ax.set_title("Loss Evolution Across Stages")
        ax.set_ylabel("Training Loss at Stage End")
        ax.set_xlabel("Stage")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_names)
        ax.grid(True, alpha=0.3)

        for i, loss in enumerate(endpoint_losses):
            ax.text(
                i,
                loss + max(endpoint_losses) * 0.01,
                f"{loss:.4f}",
                ha="center",
                va="bottom",
            )

        # 6. 最终测试性能
        ax = axes[1, 2]
        test_metrics = mmoe_results["test_metrics"]
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        values = [test_metrics[metric] for metric in metrics if metric in test_metrics]
        metrics = [metric for metric in metrics if metric in test_metrics]

        bars = ax.bar(metrics, values, color="#9C27B0")
        ax.set_title("Final MMOE Test Performance")
        ax.set_ylabel("Metric Value")

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "mmoe_stage_performance_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"✅ MMOE stage performance analysis saved to: {self.output_dir / 'mmoe_stage_performance_analysis.png'}"
        )

    def plot_performance_comparison(self):
        """Plot performance metrics comparison - 确保所有模型都清晰显示"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))  # 增加图像尺寸
        fig.suptitle(
            "Comprehensive Model Performance Comparison (All Models)",
            fontsize=16,
            fontweight="bold",
        )

        # 准备数据，确保baseline排在第一位，MMOE排在最后
        baseline_item = None
        mmoe_item = None
        other_items = []

        for model_type, results in self.results.items():
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_item = (model_type, results)
            elif (
                "MMoE" in model_type
                or "mmoe" in model_type.lower()
                or "TwoStage" in model_type
            ):
                mmoe_item = (model_type, results)
            else:
                other_items.append((model_type, results))

        # 重新排序：baseline -> 其他模型 -> MMOE
        sorted_items = []
        if baseline_item:
            sorted_items.append(baseline_item)
        sorted_items.extend(other_items)
        if mmoe_item:
            sorted_items.append(mmoe_item)

        # 重新组织数据
        model_names = [results["model_name"] for _, results in sorted_items]

        print(f"📊 Performance comparison includes {len(model_names)} models:")
        for i, name in enumerate(model_names):
            model_type = (
                "Baseline"
                if i == 0
                else ("MMOE" if i == len(model_names) - 1 else "Time-Aware")
            )
            print(f"  {i+1}. {name} ({model_type})")

        # 主要性能指标
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        metric_values = {metric: [] for metric in metrics}

        for _, results in sorted_items:
            test_metrics = results["test_metrics"]
            for metric in metrics:
                metric_values[metric].append(test_metrics[metric])

        # 绘制主要指标
        positions = np.arange(len(model_names))

        # 使用渐变颜色，特别突出baseline和MMOE
        colors = sns.color_palette("husl", len(model_names))
        if baseline_item:
            colors[0] = "#FF6B6B"  # 红色突出baseline
        if mmoe_item:
            colors[-1] = "#9C27B0"  # 紫色突出MMOE

        # RMSE对比
        ax = axes[0, 0]
        bars = ax.bar(positions, metric_values["RMSE"], color=colors)
        ax.set_title("RMSE Comparison (Lower is Better)", fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        # 添加数值标签
        for bar, value in zip(bars, metric_values["RMSE"]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # MAE对比
        ax = axes[0, 1]
        bars = ax.bar(positions, metric_values["MAE"], color=colors)
        ax.set_title("MAE Comparison (Lower is Better)", fontweight="bold")
        ax.set_ylabel("MAE")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, metric_values["MAE"]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Correlation coefficient
        ax = axes[0, 2]
        bars = ax.bar(positions, metric_values["Correlation"], color=colors)
        ax.set_title("Prediction Correlation (Higher is Better)", fontweight="bold")
        ax.set_ylabel("Correlation")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1)

        for bar, value in zip(bars, metric_values["Correlation"]):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Model complexity comparison
        ax = axes[1, 0]
        param_counts = [
            results["model_params"]["total_params"] for _, results in sorted_items
        ]
        bars = ax.bar(positions, param_counts, color=colors)
        ax.set_title("Model Parameter Count", fontweight="bold")
        ax.set_ylabel("Parameter Count")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value//1000}K" if value > 1000 else f"{value}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Training time comparison (考虑MMOE的90轮 vs 其他30轮)
        ax = axes[1, 1]
        training_times = [
            results["training_history"]["total_training_time"]
            for _, results in sorted_items
        ]
        bars = ax.bar(positions, training_times, color=colors)
        ax.set_title(
            "Total Training Time\n(MMOE: 90ep vs Others: 30ep)", fontweight="bold"
        )
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.1f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Inference time comparison
        ax = axes[1, 2]
        inference_times = [
            results["test_metrics"]["Inference_Time"] for _, results in sorted_items
        ]
        bars = ax.bar(positions, inference_times, color=colors)
        ax.set_title("Inference Time Comparison", fontweight="bold")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + height * 0.01,
                f"{value:.3f}s",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "comprehensive_performance_comparison_all_models.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        print(
            f"✅ Comprehensive performance comparison saved to: {self.output_dir / 'comprehensive_performance_comparison_all_models.png'}"
        )

    def run_complete_analysis(self):
        """Run complete analysis workflow - enhanced version with MMOE focus"""
        print("Starting enhanced model comparison analysis with MMOE stage analysis...")

        if not self.results:
            print("No model result data found. Please run training scripts first.")
            return

        print(f"Found results for {len(self.results)} models")

        # 检查是否有MMOE模型
        has_mmoe = any(
            "MMoE" in model_type
            or "mmoe" in model_type.lower()
            or "TwoStage" in model_type
            for model_type in self.results.keys()
        )

        if has_mmoe:
            print("🔍 MMOE model detected, generating specialized MMOE analysis...")

        # 生成所有图表
        print("1. Plotting MMOE detailed stage training curves...")
        self.plot_mmoe_detailed_training_curves()

        print("2. Plotting training curves comparison (MMOE vs Others)...")
        self.plot_training_curves_with_mmoe_comparison()

        print("3. Plotting MMOE stage performance analysis...")
        self.plot_mmoe_stage_performance_analysis()

        print("4. Plotting performance metrics comparison...")
        self.plot_performance_comparison()

        print("5. Generating enhanced summary table...")
        summary_df = self.generate_summary_table()

        print(f"\n✅ All analysis charts saved to: {self.output_dir}")
        print("🎯 Enhanced MMOE-focused analysis completed!")

        return summary_df

    def generate_summary_table(self):
        """Generate enhanced model performance summary table - 显示所有模型"""
        summary_data = []

        # 按类型排序：baseline -> 时间感知模型 -> MMOE
        baseline_results = []
        time_aware_results = []
        mmoe_results = []

        for model_type, results in self.results.items():
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_results.append((model_type, results))
            elif (
                "MMoE" in model_type
                or "mmoe" in model_type.lower()
                or "TwoStage" in model_type
            ):
                mmoe_results.append((model_type, results))
            else:
                time_aware_results.append((model_type, results))

        # 合并所有结果，按逻辑顺序排列
        all_results = baseline_results + time_aware_results + mmoe_results

        print(f"📋 Generating summary table for {len(all_results)} models:")

        for model_type, results in all_results:
            test_metrics = results["test_metrics"]
            training_history = results["training_history"]
            model_params = results["model_params"]

            # 计算学习率变化信息
            lr_info = "N/A"
            if (
                "learning_rates" in training_history
                and training_history["learning_rates"]
            ):
                initial_lr = training_history["learning_rates"][0]
                final_lr = training_history["learning_rates"][-1]
                lr_reduction = (
                    ((initial_lr - final_lr) / initial_lr * 100)
                    if initial_lr > 0
                    else 0
                )
                lr_info = f"{initial_lr:.6f}→{final_lr:.6f} (-{lr_reduction:.1f}%)"

            # 确定模型类别和训练信息
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                model_category = "Baseline CF"
                training_info = f"{training_history['total_epochs']} epochs"
            elif (
                "MMoE" in model_type
                or "mmoe" in model_type.lower()
                or "TwoStage" in model_type
            ):
                model_category = "MMoE (Multi-task)"
                training_info = "90 epochs (3×30 stages)"
            else:
                model_category = "Time-Aware CF"
                training_info = f"{training_history['total_epochs']} epochs"

            print(f"  - {results['model_name']} ({model_category})")

            summary_data.append(
                {
                    "Model Name": results["model_name"],
                    "Category": model_category,
                    "Model Type": model_type,
                    "RMSE": f"{test_metrics['RMSE']:.4f}",
                    "MAE": f"{test_metrics['MAE']:.4f}",
                    "MAPE (%)": f"{test_metrics['MAPE']:.2f}",
                    "Correlation": f"{test_metrics['Correlation']:.4f}",
                    "Parameter Count": f"{model_params['total_params']:,}",
                    "Training Info": training_info,
                    "Training Time (s)": f"{training_history['total_training_time']:.1f}",
                    "Inference Time (s)": f"{test_metrics['Inference_Time']:.3f}",
                    "Learning Rate Change": lr_info,
                    "Best Epoch": training_history.get("best_epoch", 0) + 1,
                }
            )

        df = pd.DataFrame(summary_data)

        # 保存为CSV
        csv_path = self.output_dir / "complete_model_comparison_summary.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        # 打印表格
        print("\n" + "=" * 150)
        print("Complete Model Performance Summary Table (All Models)")
        print("=" * 150)
        print(df.to_string(index=False))
        print("=" * 150)

        print(f"\n📊 Summary Statistics:")
        print(f"  - Total Models: {len(df)}")
        print(f"  - Baseline Models: {len(baseline_results)}")
        print(f"  - Time-Aware Models: {len(time_aware_results)}")
        print(f"  - MMOE Models: {len(mmoe_results)}")

        return df


def main():
    """Main function"""
    # Create model comparison analyzer
    analyzer = ModelComparison()

    # Run complete analysis
    summary_df = analyzer.run_complete_analysis()

    return analyzer, summary_df


if __name__ == "__main__":
    analyzer, summary = main()
