import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import logging
import math
from pathlib import Path
from data_process import data_path

# 🔧 Fix Chinese font display issues - Use English only
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")
sns.set_palette("husl")

class ModelComparison:
    def __init__(self, results_path=None):
        """Initialize model comparison class"""
        if results_path is None:
            # Try multiple possible summary file locations
            possible_paths = [
                data_path + "results/all_models_summary_with_baseline_new.json",
                data_path + "results/all_models_summary_with_baseline.json",
                data_path + "results/all_models_summary_with_scheduler.json",
                data_path + "results/all_models_summary.json"
            ]
            
            results_path = None
            for path in possible_paths:
                if Path(path).exists():
                    results_path = path
                    break
            
            if results_path is None:
                results_path = possible_paths[0]  # Default to first path

        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = Path(data_path) / "analysis_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Model name mapping for consistent English display
        self.model_name_mapping = {
            "UserTimeModel": "User Time-Aware Model",
            "IndependentTimeModel": "Independent Time Feature Model", 
            "UMTimeModel": "User-Movie Time-Aware Model",
            "TwoStageMMoE": "Two-Stage MMoE Model",
            "CFModel_Baseline": "Baseline Collaborative Filtering"
        }

    def load_results(self):
        """Load all model results - support baseline and improved models"""
        results = {}

        # Try to load main summary file
        try:
            print(f"Looking for results from: {self.results_path}")
            with open(self.results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"✅ Successfully loaded {len(results)} model results")
            return results
        except FileNotFoundError:
            print(f"⚠️  Summary file not found: {self.results_path}")

        # Try to load individual result files
        print("Looking for individual result files...")
        results_dir = Path(data_path) / "results"
        if results_dir.exists():
            individual_files = list(results_dir.glob("results_*.json"))
            print(f"Found {len(individual_files)} individual result files")
            
            for result_file in individual_files:
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        model_type = file_data.get("model_type", result_file.stem)
                        results[model_type] = file_data
                        print(f"✅ Loaded: {file_data.get('model_name', model_type)}")
                except Exception as e:
                    print(f"❌ Failed to load {result_file}: {e}")

        if not results:
            print("❌ No model results found! Please run training first.")
            
        return results

    def get_display_name(self, model_type, results):
        """Get consistent English display name for model"""
        if model_type in self.model_name_mapping:
            return self.model_name_mapping[model_type]
        
        # Fallback to model_name from results
        model_name = results.get("model_name", model_type)
        
        # Clean up any Chinese characters or problematic strings
        english_mapping = {
            "Baseline Collaborative Filtering": "Baseline Collaborative Filtering",
            "User Time-Aware Model": "User Time-Aware Model",
            "Independent Time Feature Model": "Independent Time Feature Model", 
            "User-Movie Time-Aware Model": "User-Movie Time-Aware Model",
            "Two-Stage MMoE Model (Optimized)": "Two-Stage MMoE Model",
            "Two-Stage MMoE Model": "Two-Stage MMoE Model"
        }
        
        return english_mapping.get(model_name, model_name.replace("模型", "Model").replace("基线", "Baseline"))

    def organize_models(self):
        """Organize models by type: baseline -> time-aware -> MMOE"""
        baseline_models = []
        time_aware_models = []
        mmoe_models = []
        
        for model_type, results in self.results.items():
            display_name = self.get_display_name(model_type, results)
            
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_models.append((model_type, results, display_name))
            elif "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_models.append((model_type, results, display_name))
            else:
                time_aware_models.append((model_type, results, display_name))
        
        # Combine in logical order
        organized_models = baseline_models + time_aware_models + mmoe_models
        return organized_models

    def plot_training_loss_comparison(self):
        """Plot 1: Training Loss Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        colors = plt.cm.Set3(np.linspace(0, 1, len(organized_models)))
        
        print(f"Plotting training loss for {len(organized_models)} models...")
        
        for idx, (model_type, results, display_name) in enumerate(organized_models):
            history = results.get("training_history", {})
            train_losses = history.get("train_losses", [])
            
            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                
                # Special styling for MMOE (90 epochs) vs others (30 epochs)
                if "mmoe" in model_type.lower():
                    plt.plot(epochs, train_losses, 
                           label=f"{display_name} (90 epochs)", 
                           color=colors[idx], linewidth=3, alpha=0.9,
                           linestyle='-')
                else:
                    plt.plot(epochs, train_losses, 
                           label=f"{display_name} (30 epochs)", 
                           color=colors[idx], linewidth=2, alpha=0.8,
                           linestyle='-')
                
                print(f"  ✅ {display_name}: {len(train_losses)} epochs")
            else:
                print(f"  ⚠️  {display_name}: No training loss data")
        
        plt.title("Training Loss Evolution Comparison (All Models)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "01_training_loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Training loss comparison saved to: {save_path}")

    def plot_validation_loss_comparison(self):
        """Plot 2: Validation Loss Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        colors = plt.cm.Set3(np.linspace(0, 1, len(organized_models)))
        
        print(f"Plotting validation loss for {len(organized_models)} models...")
        
        for idx, (model_type, results, display_name) in enumerate(organized_models):
            history = results.get("training_history", {})
            val_losses = history.get("val_losses", [])
            
            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                
                if "mmoe" in model_type.lower():
                    plt.plot(epochs, val_losses, 
                           label=f"{display_name} (90 epochs)", 
                           color=colors[idx], linewidth=3, alpha=0.9,
                           linestyle='-')
                else:
                    plt.plot(epochs, val_losses, 
                           label=f"{display_name} (30 epochs)", 
                           color=colors[idx], linewidth=2, alpha=0.8,
                           linestyle='-')
                
                print(f"  ✅ {display_name}: {len(val_losses)} epochs")
            else:
                print(f"  ⚠️  {display_name}: No validation loss data")
        
        plt.title("Validation Loss Evolution Comparison (All Models)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "02_validation_loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Validation loss comparison saved to: {save_path}")

    def plot_rmse_comparison(self):
        """Plot 3: RMSE Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        rmse_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            rmse = test_metrics.get("RMSE", 0)
            rmse_values.append(rmse)
        
        # Color mapping: baseline=red, time-aware=blues, mmoe=purple
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#2196F3")  # Blue
        
        bars = plt.bar(range(len(model_names)), rmse_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("RMSE Comparison (Lower is Better)", fontsize=16, fontweight='bold')
        plt.ylabel("RMSE", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "03_rmse_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ RMSE comparison saved to: {save_path}")

    def plot_mae_comparison(self):
        """Plot 4: MAE Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        mae_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            mae = test_metrics.get("MAE", 0)
            mae_values.append(mae)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#4CAF50")  # Green for time-aware
        
        bars = plt.bar(range(len(model_names)), mae_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, mae_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("MAE Comparison (Lower is Better)", fontsize=16, fontweight='bold')
        plt.ylabel("MAE", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "04_mae_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MAE comparison saved to: {save_path}")

    def plot_correlation_comparison(self):
        """Plot 5: Correlation Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        correlation_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            correlation = test_metrics.get("Correlation", 0)
            correlation_values.append(correlation)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#FF9800")  # Orange for time-aware
        
        bars = plt.bar(range(len(model_names)), correlation_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, correlation_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("Prediction Correlation (Higher is Better)", fontsize=16, fontweight='bold')
        plt.ylabel("Correlation Coefficient", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "05_correlation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Correlation comparison saved to: {save_path}")

    def plot_parameter_count_comparison(self):
        """Plot 6: Parameter Count Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        param_counts = []
        
        for model_type, results, display_name in organized_models:
            model_params = results.get("model_params", {})
            total_params = model_params.get("total_params", 0)
            param_counts.append(total_params)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#607D8B")  # Blue Grey for time-aware
        
        bars = plt.bar(range(len(model_names)), param_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels (show in K format)
        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            label = f'{value//1000}K' if value >= 1000 else str(value)
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.title("Model Parameter Count Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Parameter Count", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "06_parameter_count_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Parameter count comparison saved to: {save_path}")

    def plot_training_time_comparison(self):
        """Plot 7: Training Time Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        training_times = []
        
        for model_type, results, display_name in organized_models:
            training_history = results.get("training_history", {})
            training_time = training_history.get("total_training_time", 0)
            training_times.append(training_time)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#795548")  # Brown for time-aware
        
        bars = plt.bar(range(len(model_names)), training_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.title("Training Time Comparison\n(MMoE: 90 epochs vs Others: 30 epochs)", fontsize=16, fontweight='bold')
        plt.ylabel("Training Time (seconds)", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "07_training_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Training time comparison saved to: {save_path}")

    def plot_inference_time_comparison(self):
        """Plot 8: Inference Time Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        inference_times = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            inference_time = test_metrics.get("Inference_Time", 0)
            inference_times.append(inference_time)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#009688")  # Teal for time-aware
        
        bars = plt.bar(range(len(model_names)), inference_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.title("Inference Time Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Inference Time (seconds)", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "08_inference_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Inference time comparison saved to: {save_path}")

    def plot_mmoe_stage_analysis(self):
        """Plot 9: MMoE Stage Analysis - Single chart"""
        # Find MMoE model
        mmoe_results = None
        mmoe_display_name = ""
        
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                mmoe_display_name = self.get_display_name(model_type, results)
                break
        
        if mmoe_results is None:
            print("⚠️  No MMoE model found for stage analysis")
            return
        
        history = mmoe_results.get("training_history", {})
        train_losses = history.get("train_losses", [])
        
        if len(train_losses) < 90:
            print(f"⚠️  MMoE training history insufficient: {len(train_losses)} epochs")
            return
        
        plt.figure(figsize=(16, 10))
        
        # Define stages
        stage_names = ["Stage 1: Temporal\nModeling", "Stage 2: Collaborative\nFiltering", "Stage 3: MMoE\nFusion"]
        stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        # Plot complete training curve with stage annotations
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, linewidth=3, color='#333333', alpha=0.8, label='Training Loss')
        
        # Add stage backgrounds
        stage_boundaries = [0, 30, 60, 90]
        for i in range(3):
            start_epoch = stage_boundaries[i]
            end_epoch = stage_boundaries[i+1]
            plt.axvspan(start_epoch, end_epoch, alpha=0.2, color=stage_colors[i])
            
            # Add stage labels
            mid_epoch = (start_epoch + end_epoch) / 2
            plt.text(mid_epoch, max(train_losses) * 0.95, stage_names[i], 
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=stage_colors[i], alpha=0.7))
        
        # Add vertical lines at stage boundaries
        for boundary in [30, 60]:
            plt.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Calculate and display stage statistics
        stage_stats = []
        for i in range(3):
            start_idx = stage_boundaries[i]
            end_idx = stage_boundaries[i+1]
            stage_losses = train_losses[start_idx:end_idx]
            
            if stage_losses:
                avg_loss = np.mean(stage_losses)
                loss_reduction = ((stage_losses[0] - stage_losses[-1]) / stage_losses[0] * 100) if stage_losses[0] > 0 else 0
                stage_stats.append(f"Avg: {avg_loss:.4f}, Reduction: {loss_reduction:.1f}%")
            else:
                stage_stats.append("No data")
        
        plt.title(f"{mmoe_display_name} - Three-Stage Training Analysis\n(90 Epochs Total)", 
                 fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = "\n".join([f"{stage_names[i].split(':')[0]}: {stage_stats[i]}" 
                               for i in range(3)])
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        save_path = self.output_dir / "09_mmoe_stage_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MMoE stage analysis saved to: {save_path}")

    def create_summary_table_image(self):
        """Plot 10: Summary Table as Image"""
        organized_models = self.organize_models()
        
        # Prepare table data
        table_data = []
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            training_history = results.get("training_history", {})
            model_params = results.get("model_params", {})
            
            # Determine epochs info
            if "mmoe" in model_type.lower():
                epochs_info = "90 (30×3)"
            else:
                epochs_info = "30"
            
            row = [
                display_name,
                f"{test_metrics.get('RMSE', 0):.4f}",
                f"{test_metrics.get('MAE', 0):.4f}",
                f"{test_metrics.get('MAPE', 0):.2f}%",
                f"{test_metrics.get('Correlation', 0):.4f}",
                f"{model_params.get('total_params', 0):,}",
                epochs_info,
                f"{training_history.get('total_training_time', 0):.1f}s",
                f"{test_metrics.get('Inference_Time', 0):.3f}s"
            ]
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Table headers
        headers = ["Model Name", "RMSE", "MAE", "MAPE", "Correlation", 
                  "Parameters", "Epochs", "Training Time", "Inference Time"]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by model type
        for i, (model_type, _, _) in enumerate(organized_models):
            row_idx = i + 1
            if "baseline" in model_type.lower():
                color = '#FFE5E5'  # Light red
            elif "mmoe" in model_type.lower():
                color = '#F3E5F5'  # Light purple
            else:
                color = '#E3F2FD'  # Light blue
            
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor(color)
        
        # Highlight best values
        rmse_values = [float(row[1]) for row in table_data]
        mae_values = [float(row[2]) for row in table_data]
        corr_values = [float(row[4]) for row in table_data]
        
        best_rmse_idx = rmse_values.index(min(rmse_values))
        best_mae_idx = mae_values.index(min(mae_values))
        best_corr_idx = corr_values.index(max(corr_values))
        
        # Highlight best RMSE
        table[(best_rmse_idx + 1, 1)].set_facecolor('#C8E6C9')
        table[(best_rmse_idx + 1, 1)].set_text_props(weight='bold')
        
        # Highlight best MAE
        table[(best_mae_idx + 1, 2)].set_facecolor('#C8E6C9')
        table[(best_mae_idx + 1, 2)].set_text_props(weight='bold')
        
        # Highlight best Correlation
        table[(best_corr_idx + 1, 4)].set_facecolor('#C8E6C9')
        table[(best_corr_idx + 1, 4)].set_text_props(weight='bold')
        
        plt.title("Complete Model Performance Summary\n(Green highlights indicate best performance)", 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "10_summary_table.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Summary table saved to: {save_path}")
        
        return table_data, headers

    def plot_mmoe_detailed_training_curves(self):
        """Plot 11: MMoE Detailed Stage Training - Single chart"""
        # Find MMoE model
        mmoe_results = None
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMoE model results not found")
            return

        print("Found MMoE model, creating detailed training analysis...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(
            "MMoE Three-Stage Training Analysis (90 Epochs Total)",
            fontsize=16,
            fontweight="bold",
        )

        history = mmoe_results["training_history"]

        # Check training history data
        if "train_losses" not in history or not history["train_losses"]:
            print("❌ MMoE training history data not found")
            return

        total_epochs = len(history["train_losses"])
        print(f"MMoE total training epochs: {total_epochs}")

        # Split into three stages (assume 30 epochs each)
        stage_epochs = 30
        stages = {
            "Stage 1: Temporal Modeling": (0, stage_epochs),
            "Stage 2: CF Modeling": (stage_epochs, 2 * stage_epochs),
            "Stage 3: MMoE Fusion": (2 * stage_epochs, min(total_epochs, 3 * stage_epochs)),
        }

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Plot each stage's training loss
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

                # If validation loss exists, plot it too
                if "val_losses" in history and history["val_losses"] and end <= len(history["val_losses"]):
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

                # Add statistics
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

        # Plot learning rate changes for each stage
        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            ax = axes[idx, 1]

            if ("learning_rates" in history and history["learning_rates"] and 
                end > start and end <= len(history["learning_rates"])):

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

                # Add learning rate change info
                initial_lr = stage_learning_rates[0] if stage_learning_rates else 0
                final_lr = stage_learning_rates[-1] if stage_learning_rates else 0
                lr_change = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0

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
        
        save_path = self.output_dir / "11_mmoe_detailed_stage_training.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MMoE detailed training curves saved to: {save_path}")

    def plot_training_curves_with_mmoe_comparison(self):
        """Plot 12: Training Curves with MMoE Comparison - Single chart"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(
            "Complete Training Process Comparison (All Models)",
            fontsize=16,
            fontweight="bold",
        )

        # Use more colors to ensure each model has unique color
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        # Process MMoE and other models separately
        mmoe_data = None
        other_models = []
        all_models_info = []

        for model_type, results in self.results.items():
            display_name = self.get_display_name(model_type, results)
            model_info = {
                "type": model_type,
                "results": results,
                "name": display_name,
                "is_mmoe": False,
            }

            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_data = (model_type, results)
                model_info["is_mmoe"] = True
            else:
                other_models.append((model_type, results))

            all_models_info.append(model_info)

        print(f"Found {len(all_models_info)} models for comparison:")
        for model_info in all_models_info:
            model_type_label = " (MMoE - 90 epochs)" if model_info["is_mmoe"] else " (30 epochs)"
            print(f"  - {model_info['name']}{model_type_label}")

        # 1. Training loss comparison - show all models
        ax = axes[0, 0]

        # Plot other models (30 epochs)
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["train_losses"],
                    label=f"{display_name} (30ep)",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE (90 epochs) with special styling
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)

                # MMoE with thick line and special color
                ax.plot(
                    epochs, history["train_losses"],
                    label=f"{mmoe_display_name} (90ep)",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # Add stage dividers
                stage_boundaries = [30, 60]
                for boundary in stage_boundaries:
                    if boundary < len(history["train_losses"]):
                        ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

                # Annotate three stages
                stage_labels = ["Temporal", "CF", "MMoE"]
                stage_positions = [15, 45, 75]
                for pos, label in zip(stage_positions, stage_labels):
                    if pos < len(history["train_losses"]):
                        ax.text(
                            pos, max(history["train_losses"]) * 0.9, label,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                            fontsize=9, color="#FF1744",
                        )

        ax.set_title("Training Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. Validation loss comparison - show all models
        ax = axes[0, 1]

        # Plot other models' validation loss
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["val_losses"],
                    label=f"{display_name} (30ep)",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE validation loss
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                ax.plot(
                    epochs, history["val_losses"],
                    label=f"{mmoe_display_name} (90ep)",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # Add stage dividers
                for boundary in [30, 60]:
                    if boundary < len(history["val_losses"]):
                        ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

        ax.set_title("Validation Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Learning rate evolution comparison - show all models
        ax = axes[1, 0]

        # Plot other models' learning rates
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "learning_rates" in history and history["learning_rates"]:
                epochs = range(1, len(history["learning_rates"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["learning_rates"],
                    label=f"{display_name}",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE learning rate
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "learning_rates" in history and history["learning_rates"]:
                epochs = range(1, len(history["learning_rates"]) + 1)
                ax.plot(
                    epochs, history["learning_rates"],
                    label=f"{mmoe_display_name}",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # Add stage dividers
                for boundary in [30, 60]:
                    if boundary < len(history["learning_rates"]):
                        ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

        ax.set_title("Learning Rate Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. Fair 30-epoch comparison (all models normalized to 30 epochs)
        ax = axes[1, 1]

        # Plot other models (original 30 epochs)
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, min(31, len(history["train_losses"]) + 1))
                train_losses = history["train_losses"][:30]
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, train_losses,
                    label=f"{display_name}",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE's last 30 epochs (61-90 epochs)
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "train_losses" in history and len(history["train_losses"]) >= 90:
                # Take last 30 epochs (61-90)
                mmoe_last_30 = history["train_losses"][60:90]
                epochs = range(1, len(mmoe_last_30) + 1)
                ax.plot(
                    epochs, mmoe_last_30,
                    label=f"{mmoe_display_name} (Stage 3)",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

        ax.set_title("Fair 30-Epoch Comparison\n(MMoE: Final Stage vs Others: Full Training)")
        ax.set_xlabel("Epoch (within 30-epoch window)")
        ax.set_ylabel("Training Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        save_path = self.output_dir / "12_training_comparison_all_models_with_mmoe.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Complete model training comparison saved to: {save_path}")

    def plot_mmoe_stage_performance_analysis(self):
        """Plot 13: MMoE Stage Performance Analysis - Single chart"""
        # Find MMoE model
        mmoe_results = None
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMoE model results not found")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("MMoE Stage-by-Stage Performance Analysis", fontsize=16, fontweight="bold")

        history = mmoe_results["training_history"]

        if "train_losses" not in history or len(history["train_losses"]) < 90:
            print("❌ MMoE training history insufficient for stage analysis")
            return

        # Define three stages
        stages = {
            "Stage 1: Temporal Modeling": (0, 30),
            "Stage 2: CF Modeling": (30, 60),
            "Stage 3: MMoE Fusion": (60, 90),
        }

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # 1. Average loss by stage
        ax = axes[0, 0]
        stage_avg_losses = []
        stage_names = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                avg_loss = np.mean(stage_losses)
                stage_avg_losses.append(avg_loss)
                stage_names.append(stage_name.split(":")[0])

        bars = ax.bar(stage_names, stage_avg_losses, color=colors[:len(stage_names)])
        ax.set_title("Average Training Loss by Stage")
        ax.set_ylabel("Average MSE Loss")

        for bar, loss in zip(bars, stage_avg_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{loss:.4f}", ha="center", va="bottom")

        # 2. Loss reduction analysis
        ax = axes[0, 1]
        loss_reductions = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                initial_loss = stage_losses[0]
                final_loss = stage_losses[-1]
                reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                loss_reductions.append(reduction)

        bars = ax.bar(stage_names, loss_reductions, color=colors[:len(stage_names)])
        ax.set_title("Loss Reduction by Stage")
        ax.set_ylabel("Loss Reduction (%)")

        for bar, reduction in zip(bars, loss_reductions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{reduction:.1f}%", ha="center", va="bottom")

        # 3. Learning rate decay analysis
        ax = axes[0, 2]
        if "learning_rates" in history and len(history["learning_rates"]) >= 90:
            lr_changes = []

            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                stage_lrs = history["learning_rates"][start:end]
                if stage_lrs:
                    initial_lr = stage_lrs[0]
                    final_lr = stage_lrs[-1]
                    change = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
                    lr_changes.append(change)

            bars = ax.bar(stage_names, lr_changes, color=colors[:len(stage_names)])
            ax.set_title("Learning Rate Decay by Stage")
            ax.set_ylabel("LR Decay (%)")

            for bar, change in zip(bars, lr_changes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                       f"{change:.1f}%", ha="center", va="bottom")

        # 4. Training stability analysis (loss variance)
        ax = axes[1, 0]
        loss_stds = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                loss_std = np.std(stage_losses)
                loss_stds.append(loss_std)

        bars = ax.bar(stage_names, loss_stds, color=colors[:len(stage_names)])
        ax.set_title("Training Stability by Stage")
        ax.set_ylabel("Loss Standard Deviation")

        for bar, std in zip(bars, loss_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{std:.4f}", ha="center", va="bottom")

        # 5. Stage transition loss changes
        ax = axes[1, 1]
        stage_endpoints = [30, 60, 90]
        endpoint_losses = [
            history["train_losses"][ep - 1]
            for ep in stage_endpoints
            if ep <= len(history["train_losses"])
        ]

        x_pos = range(len(endpoint_losses))
        ax.plot(x_pos, endpoint_losses, "o-", linewidth=3, markersize=8, color="#FF1744")
        ax.set_title("Loss Evolution Across Stages")
        ax.set_ylabel("Training Loss at Stage End")
        ax.set_xlabel("Stage")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_names)
        ax.grid(True, alpha=0.3)

        for i, loss in enumerate(endpoint_losses):
            ax.text(i, loss + max(endpoint_losses)*0.01, f"{loss:.4f}", ha="center", va="bottom")

        # 6. Final test performance
        ax = axes[1, 2]
        test_metrics = mmoe_results["test_metrics"]
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        values = [test_metrics[metric] for metric in metrics if metric in test_metrics]
        metrics = [metric for metric in metrics if metric in test_metrics]

        bars = ax.bar(metrics, values, color="#9C27B0")
        ax.set_title("Final MMoE Test Performance")
        ax.set_ylabel("Metric Value")

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.4f}", ha="center", va="bottom")

        plt.tight_layout()
        
        save_path = self.output_dir / "13_mmoe_stage_performance_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MMoE stage performance analysis saved to: {save_path}")

    def plot_performance_comparison(self):
        """Plot 14: Performance Metrics Comparison - Single chart"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(
            "Comprehensive Model Performance Comparison (All Models)",
            fontsize=16,
            fontweight="bold",
        )

        # Organize data, ensuring baseline first, MMoE last
        baseline_item = None
        mmoe_item = None
        other_items = []

        for model_type, results in self.results.items():
            display_name = self.get_display_name(model_type, results)
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_item = (model_type, results, display_name)
            elif "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_item = (model_type, results, display_name)
            else:
                other_items.append((model_type, results, display_name))

        # Reorder: baseline -> other models -> MMoE
        sorted_items = []
        if baseline_item:
            sorted_items.append(baseline_item)
        sorted_items.extend(other_items)
        if mmoe_item:
            sorted_items.append(mmoe_item)

        # Reorganize data
        model_names = [display_name for _, _, display_name in sorted_items]

        print(f"Performance comparison includes {len(model_names)} models:")
        for i, name in enumerate(model_names):
            model_type = ("Baseline" if i == 0 else 
                         ("MMoE" if i == len(model_names) - 1 else "Time-Aware"))
            print(f"  {i+1}. {name} ({model_type})")

        # Main performance indicators
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        metric_values = {metric: [] for metric in metrics}

        for _, results, _ in sorted_items:
            test_metrics = results["test_metrics"]
            for metric in metrics:
                metric_values[metric].append(test_metrics[metric])

        # Plot main indicators
        positions = np.arange(len(model_names))

        # Use gradient colors, especially highlight baseline and MMoE
        colors = sns.color_palette("husl", len(model_names))
        if baseline_item:
            colors[0] = "#FF6B6B"  # Red highlight baseline
        if mmoe_item:
            colors[-1] = "#9C27B0"  # Purple highlight MMoE

        # RMSE comparison
        ax = axes[0, 0]
        bars = ax.bar(positions, metric_values["RMSE"], color=colors)
        ax.set_title("RMSE Comparison (Lower is Better)", fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        # Add numeric labels
        for bar, value in zip(bars, metric_values["RMSE"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.4f}", ha="center", va="bottom", fontsize=9)

        # MAE comparison
        ax = axes[0, 1]
        bars = ax.bar(positions, metric_values["MAE"], color=colors)
        ax.set_title("MAE Comparison (Lower is Better)", fontweight="bold")
        ax.set_ylabel("MAE")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, metric_values["MAE"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.4f}", ha="center", va="bottom", fontsize=9)

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
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.01,
                   f"{value:.4f}", ha="center", va="bottom", fontsize=9)

        # Model complexity comparison
        ax = axes[1, 0]
        param_counts = [results["model_params"]["total_params"] for _, results, _ in sorted_items]
        bars = ax.bar# filepath: g:\Personal\1C401\CS182_proj\torch_ver\model_comparison.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import logging
import math
from pathlib import Path
from data_process import data_path

# 🔧 Fix Chinese font display issues - Use English only
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
sns.set_style("whitegrid")
sns.set_palette("husl")

class ModelComparison:
    def __init__(self, results_path=None):
        """Initialize model comparison class"""
        if results_path is None:
            # Try multiple possible summary file locations
            possible_paths = [
                data_path + "results/all_models_summary_with_baseline_new.json",
                data_path + "results/all_models_summary_with_baseline.json",
                data_path + "results/all_models_summary_with_scheduler.json",
                data_path + "results/all_models_summary.json"
            ]
            
            results_path = None
            for path in possible_paths:
                if Path(path).exists():
                    results_path = path
                    break
            
            if results_path is None:
                results_path = possible_paths[0]  # Default to first path

        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = Path(data_path) / "analysis_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        # Model name mapping for consistent English display
        self.model_name_mapping = {
            "UserTimeModel": "User Time-Aware Model",
            "IndependentTimeModel": "Independent Time Feature Model", 
            "UMTimeModel": "User-Movie Time-Aware Model",
            "TwoStageMMoE": "Two-Stage MMoE Model",
            "CFModel_Baseline": "Baseline Collaborative Filtering"
        }

    def load_results(self):
        """Load all model results - support baseline and improved models"""
        results = {}

        # Try to load main summary file
        try:
            print(f"Looking for results from: {self.results_path}")
            with open(self.results_path, "r", encoding="utf-8") as f:
                results = json.load(f)
            print(f"✅ Successfully loaded {len(results)} model results")
            return results
        except FileNotFoundError:
            print(f"⚠️  Summary file not found: {self.results_path}")

        # Try to load individual result files
        print("Looking for individual result files...")
        results_dir = Path(data_path) / "results"
        if results_dir.exists():
            individual_files = list(results_dir.glob("results_*.json"))
            print(f"Found {len(individual_files)} individual result files")
            
            for result_file in individual_files:
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        file_data = json.load(f)
                        model_type = file_data.get("model_type", result_file.stem)
                        results[model_type] = file_data
                        print(f"✅ Loaded: {file_data.get('model_name', model_type)}")
                except Exception as e:
                    print(f"❌ Failed to load {result_file}: {e}")

        if not results:
            print("❌ No model results found! Please run training first.")
            
        return results

    def get_display_name(self, model_type, results):
        """Get consistent English display name for model"""
        if model_type in self.model_name_mapping:
            return self.model_name_mapping[model_type]
        
        # Fallback to model_name from results
        model_name = results.get("model_name", model_type)
        
        # Clean up any Chinese characters or problematic strings
        english_mapping = {
            "Baseline Collaborative Filtering": "Baseline Collaborative Filtering",
            "User Time-Aware Model": "User Time-Aware Model",
            "Independent Time Feature Model": "Independent Time Feature Model", 
            "User-Movie Time-Aware Model": "User-Movie Time-Aware Model",
            "Two-Stage MMoE Model (Optimized)": "Two-Stage MMoE Model",
            "Two-Stage MMoE Model": "Two-Stage MMoE Model"
        }
        
        return english_mapping.get(model_name, model_name.replace("模型", "Model").replace("基线", "Baseline"))

    def organize_models(self):
        """Organize models by type: baseline -> time-aware -> MMOE"""
        baseline_models = []
        time_aware_models = []
        mmoe_models = []
        
        for model_type, results in self.results.items():
            display_name = self.get_display_name(model_type, results)
            
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_models.append((model_type, results, display_name))
            elif "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_models.append((model_type, results, display_name))
            else:
                time_aware_models.append((model_type, results, display_name))
        
        # Combine in logical order
        organized_models = baseline_models + time_aware_models + mmoe_models
        return organized_models

    def plot_training_loss_comparison(self):
        """Plot 1: Training Loss Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        colors = plt.cm.Set3(np.linspace(0, 1, len(organized_models)))
        
        print(f"Plotting training loss for {len(organized_models)} models...")
        
        for idx, (model_type, results, display_name) in enumerate(organized_models):
            history = results.get("training_history", {})
            train_losses = history.get("train_losses", [])
            
            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                
                # Special styling for MMOE (90 epochs) vs others (30 epochs)
                if "mmoe" in model_type.lower():
                    plt.plot(epochs, train_losses, 
                           label=f"{display_name} (90 epochs)", 
                           color=colors[idx], linewidth=3, alpha=0.9,
                           linestyle='-')
                else:
                    plt.plot(epochs, train_losses, 
                           label=f"{display_name} (30 epochs)", 
                           color=colors[idx], linewidth=2, alpha=0.8,
                           linestyle='-')
                
                print(f"  ✅ {display_name}: {len(train_losses)} epochs")
            else:
                print(f"  ⚠️  {display_name}: No training loss data")
        
        plt.title("Training Loss Evolution Comparison (All Models)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "01_training_loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Training loss comparison saved to: {save_path}")

    def plot_validation_loss_comparison(self):
        """Plot 2: Validation Loss Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        colors = plt.cm.Set3(np.linspace(0, 1, len(organized_models)))
        
        print(f"Plotting validation loss for {len(organized_models)} models...")
        
        for idx, (model_type, results, display_name) in enumerate(organized_models):
            history = results.get("training_history", {})
            val_losses = history.get("val_losses", [])
            
            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                
                if "mmoe" in model_type.lower():
                    plt.plot(epochs, val_losses, 
                           label=f"{display_name} (90 epochs)", 
                           color=colors[idx], linewidth=3, alpha=0.9,
                           linestyle='-')
                else:
                    plt.plot(epochs, val_losses, 
                           label=f"{display_name} (30 epochs)", 
                           color=colors[idx], linewidth=2, alpha=0.8,
                           linestyle='-')
                
                print(f"  ✅ {display_name}: {len(val_losses)} epochs")
            else:
                print(f"  ⚠️  {display_name}: No validation loss data")
        
        plt.title("Validation Loss Evolution Comparison (All Models)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "02_validation_loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Validation loss comparison saved to: {save_path}")

    def plot_rmse_comparison(self):
        """Plot 3: RMSE Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        rmse_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            rmse = test_metrics.get("RMSE", 0)
            rmse_values.append(rmse)
        
        # Color mapping: baseline=red, time-aware=blues, mmoe=purple
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#2196F3")  # Blue
        
        bars = plt.bar(range(len(model_names)), rmse_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels on bars
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("RMSE Comparison (Lower is Better)", fontsize=16, fontweight='bold')
        plt.ylabel("RMSE", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "03_rmse_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ RMSE comparison saved to: {save_path}")

    def plot_mae_comparison(self):
        """Plot 4: MAE Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        mae_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            mae = test_metrics.get("MAE", 0)
            mae_values.append(mae)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#4CAF50")  # Green for time-aware
        
        bars = plt.bar(range(len(model_names)), mae_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, mae_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("MAE Comparison (Lower is Better)", fontsize=16, fontweight='bold')
        plt.ylabel("MAE", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "04_mae_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MAE comparison saved to: {save_path}")

    def plot_correlation_comparison(self):
        """Plot 5: Correlation Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        correlation_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            correlation = test_metrics.get("Correlation", 0)
            correlation_values.append(correlation)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#FF9800")  # Orange for time-aware
        
        bars = plt.bar(range(len(model_names)), correlation_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, correlation_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.title("Prediction Correlation (Higher is Better)", fontsize=16, fontweight='bold')
        plt.ylabel("Correlation Coefficient", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "05_correlation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Correlation comparison saved to: {save_path}")

    def plot_parameter_count_comparison(self):
        """Plot 6: Parameter Count Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        param_counts = []
        
        for model_type, results, display_name in organized_models:
            model_params = results.get("model_params", {})
            total_params = model_params.get("total_params", 0)
            param_counts.append(total_params)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#607D8B")  # Blue Grey for time-aware
        
        bars = plt.bar(range(len(model_names)), param_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels (show in K format)
        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            label = f'{value//1000}K' if value >= 1000 else str(value)
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        plt.title("Model Parameter Count Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Parameter Count", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "06_parameter_count_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Parameter count comparison saved to: {save_path}")

    def plot_training_time_comparison(self):
        """Plot 7: Training Time Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        training_times = []
        
        for model_type, results, display_name in organized_models:
            training_history = results.get("training_history", {})
            training_time = training_history.get("total_training_time", 0)
            training_times.append(training_time)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#795548")  # Brown for time-aware
        
        bars = plt.bar(range(len(model_names)), training_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.title("Training Time Comparison\n(MMoE: 90 epochs vs Others: 30 epochs)", fontsize=16, fontweight='bold')
        plt.ylabel("Training Time (seconds)", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "07_training_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Training time comparison saved to: {save_path}")

    def plot_inference_time_comparison(self):
        """Plot 8: Inference Time Comparison - Single chart"""
        plt.figure(figsize=(12, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        inference_times = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            inference_time = test_metrics.get("Inference_Time", 0)
            inference_times.append(inference_time)
        
        # Color mapping
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")
            else:
                colors.append("#009688")  # Teal for time-aware
        
        bars = plt.bar(range(len(model_names)), inference_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.title("Inference Time Comparison", fontsize=16, fontweight='bold')
        plt.ylabel("Inference Time (seconds)", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "08_inference_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Inference time comparison saved to: {save_path}")

    def plot_mmoe_stage_analysis(self):
        """Plot 9: MMoE Stage Analysis - Single chart"""
        # Find MMoE model
        mmoe_results = None
        mmoe_display_name = ""
        
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                mmoe_display_name = self.get_display_name(model_type, results)
                break
        
        if mmoe_results is None:
            print("⚠️  No MMoE model found for stage analysis")
            return
        
        history = mmoe_results.get("training_history", {})
        train_losses = history.get("train_losses", [])
        
        if len(train_losses) < 90:
            print(f"⚠️  MMoE training history insufficient: {len(train_losses)} epochs")
            return
        
        plt.figure(figsize=(16, 10))
        
        # Define stages
        stage_names = ["Stage 1: Temporal\nModeling", "Stage 2: Collaborative\nFiltering", "Stage 3: MMoE\nFusion"]
        stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        # Plot complete training curve with stage annotations
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, linewidth=3, color='#333333', alpha=0.8, label='Training Loss')
        
        # Add stage backgrounds
        stage_boundaries = [0, 30, 60, 90]
        for i in range(3):
            start_epoch = stage_boundaries[i]
            end_epoch = stage_boundaries[i+1]
            plt.axvspan(start_epoch, end_epoch, alpha=0.2, color=stage_colors[i])
            
            # Add stage labels
            mid_epoch = (start_epoch + end_epoch) / 2
            plt.text(mid_epoch, max(train_losses) * 0.95, stage_names[i], 
                    ha='center', va='top', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=stage_colors[i], alpha=0.7))
        
        # Add vertical lines at stage boundaries
        for boundary in [30, 60]:
            plt.axvline(x=boundary, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        # Calculate and display stage statistics
        stage_stats = []
        for i in range(3):
            start_idx = stage_boundaries[i]
            end_idx = stage_boundaries[i+1]
            stage_losses = train_losses[start_idx:end_idx]
            
            if stage_losses:
                avg_loss = np.mean(stage_losses)
                loss_reduction = ((stage_losses[0] - stage_losses[-1]) / stage_losses[0] * 100) if stage_losses[0] > 0 else 0
                stage_stats.append(f"Avg: {avg_loss:.4f}, Reduction: {loss_reduction:.1f}%")
            else:
                stage_stats.append("No data")
        
        plt.title(f"{mmoe_display_name} - Three-Stage Training Analysis\n(90 Epochs Total)", 
                 fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics text box
        stats_text = "\n".join([f"{stage_names[i].split(':')[0]}: {stage_stats[i]}" 
                               for i in range(3)])
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        save_path = self.output_dir / "09_mmoe_stage_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MMoE stage analysis saved to: {save_path}")

    def create_summary_table_image(self):
        """Plot 10: Summary Table as Image"""
        organized_models = self.organize_models()
        
        # Prepare table data
        table_data = []
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            training_history = results.get("training_history", {})
            model_params = results.get("model_params", {})
            
            # Determine epochs info
            if "mmoe" in model_type.lower():
                epochs_info = "90 (30×3)"
            else:
                epochs_info = "30"
            
            row = [
                display_name,
                f"{test_metrics.get('RMSE', 0):.4f}",
                f"{test_metrics.get('MAE', 0):.4f}",
                f"{test_metrics.get('MAPE', 0):.2f}%",
                f"{test_metrics.get('Correlation', 0):.4f}",
                f"{model_params.get('total_params', 0):,}",
                epochs_info,
                f"{training_history.get('total_training_time', 0):.1f}s",
                f"{test_metrics.get('Inference_Time', 0):.3f}s"
            ]
            table_data.append(row)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(18, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Table headers
        headers = ["Model Name", "RMSE", "MAE", "MAPE", "Correlation", 
                  "Parameters", "Epochs", "Training Time", "Inference Time"]
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by model type
        for i, (model_type, _, _) in enumerate(organized_models):
            row_idx = i + 1
            if "baseline" in model_type.lower():
                color = '#FFE5E5'  # Light red
            elif "mmoe" in model_type.lower():
                color = '#F3E5F5'  # Light purple
            else:
                color = '#E3F2FD'  # Light blue
            
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor(color)
        
        # Highlight best values
        rmse_values = [float(row[1]) for row in table_data]
        mae_values = [float(row[2]) for row in table_data]
        corr_values = [float(row[4]) for row in table_data]
        
        best_rmse_idx = rmse_values.index(min(rmse_values))
        best_mae_idx = mae_values.index(min(mae_values))
        best_corr_idx = corr_values.index(max(corr_values))
        
        # Highlight best RMSE
        table[(best_rmse_idx + 1, 1)].set_facecolor('#C8E6C9')
        table[(best_rmse_idx + 1, 1)].set_text_props(weight='bold')
        
        # Highlight best MAE
        table[(best_mae_idx + 1, 2)].set_facecolor('#C8E6C9')
        table[(best_mae_idx + 1, 2)].set_text_props(weight='bold')
        
        # Highlight best Correlation
        table[(best_corr_idx + 1, 4)].set_facecolor('#C8E6C9')
        table[(best_corr_idx + 1, 4)].set_text_props(weight='bold')
        
        plt.title("Complete Model Performance Summary\n(Green highlights indicate best performance)", 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "10_summary_table.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Summary table saved to: {save_path}")
        
        return table_data, headers

    def plot_mmoe_detailed_training_curves(self):
        """Plot 11: MMoE Detailed Stage Training - Single chart"""
        # Find MMoE model
        mmoe_results = None
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMoE model results not found")
            return

        print("Found MMoE model, creating detailed training analysis...")

        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(
            "MMoE Three-Stage Training Analysis (90 Epochs Total)",
            fontsize=16,
            fontweight="bold",
        )

        history = mmoe_results["training_history"]

        # Check training history data
        if "train_losses" not in history or not history["train_losses"]:
            print("❌ MMoE training history data not found")
            return

        total_epochs = len(history["train_losses"])
        print(f"MMoE total training epochs: {total_epochs}")

        # Split into three stages (assume 30 epochs each)
        stage_epochs = 30
        stages = {
            "Stage 1: Temporal Modeling": (0, stage_epochs),
            "Stage 2: CF Modeling": (stage_epochs, 2 * stage_epochs),
            "Stage 3: MMoE Fusion": (2 * stage_epochs, min(total_epochs, 3 * stage_epochs)),
        }

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # Plot each stage's training loss
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

                # If validation loss exists, plot it too
                if "val_losses" in history and history["val_losses"] and end <= len(history["val_losses"]):
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

                # Add statistics
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

        # Plot learning rate changes for each stage
        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            ax = axes[idx, 1]

            if ("learning_rates" in history and history["learning_rates"] and 
                end > start and end <= len(history["learning_rates"])):

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

                # Add learning rate change info
                initial_lr = stage_learning_rates[0] if stage_learning_rates else 0
                final_lr = stage_learning_rates[-1] if stage_learning_rates else 0
                lr_change = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0

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
        
        save_path = self.output_dir / "11_mmoe_detailed_stage_training.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MMoE detailed training curves saved to: {save_path}")

    def plot_training_curves_with_mmoe_comparison(self):
        """Plot 12: Training Curves with MMoE Comparison - Single chart"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 14))
        fig.suptitle(
            "Complete Training Process Comparison (All Models)",
            fontsize=16,
            fontweight="bold",
        )

        # Use more colors to ensure each model has unique color
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        # Process MMoE and other models separately
        mmoe_data = None
        other_models = []
        all_models_info = []

        for model_type, results in self.results.items():
            display_name = self.get_display_name(model_type, results)
            model_info = {
                "type": model_type,
                "results": results,
                "name": display_name,
                "is_mmoe": False,
            }

            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_data = (model_type, results)
                model_info["is_mmoe"] = True
            else:
                other_models.append((model_type, results))

            all_models_info.append(model_info)

        print(f"Found {len(all_models_info)} models for comparison:")
        for model_info in all_models_info:
            model_type_label = " (MMoE - 90 epochs)" if model_info["is_mmoe"] else " (30 epochs)"
            print(f"  - {model_info['name']}{model_type_label}")

        # 1. Training loss comparison - show all models
        ax = axes[0, 0]

        # Plot other models (30 epochs)
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["train_losses"],
                    label=f"{display_name} (30ep)",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE (90 epochs) with special styling
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)

                # MMoE with thick line and special color
                ax.plot(
                    epochs, history["train_losses"],
                    label=f"{mmoe_display_name} (90ep)",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # Add stage dividers
                stage_boundaries = [30, 60]
                for boundary in stage_boundaries:
                    if boundary < len(history["train_losses"]):
                        ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

                # Annotate three stages
                stage_labels = ["Temporal", "CF", "MMoE"]
                stage_positions = [15, 45, 75]
                for pos, label in zip(stage_positions, stage_labels):
                    if pos < len(history["train_losses"]):
                        ax.text(
                            pos, max(history["train_losses"]) * 0.9, label,
                            ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                            fontsize=9, color="#FF1744",
                        )

        ax.set_title("Training Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 2. Validation loss comparison - show all models
        ax = axes[0, 1]

        # Plot other models' validation loss
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["val_losses"],
                    label=f"{display_name} (30ep)",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE validation loss
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                ax.plot(
                    epochs, history["val_losses"],
                    label=f"{mmoe_display_name} (90ep)",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # Add stage dividers
                for boundary in [30, 60]:
                    if boundary < len(history["val_losses"]):
                        ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

        ax.set_title("Validation Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Learning rate evolution comparison - show all models
        ax = axes[1, 0]

        # Plot other models' learning rates
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "learning_rates" in history and history["learning_rates"]:
                epochs = range(1, len(history["learning_rates"]) + 1)
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["learning_rates"],
                    label=f"{display_name}",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE learning rate
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "learning_rates" in history and history["learning_rates"]:
                epochs = range(1, len(history["learning_rates"]) + 1)
                ax.plot(
                    epochs, history["learning_rates"],
                    label=f"{mmoe_display_name}",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # Add stage dividers
                for boundary in [30, 60]:
                    if boundary < len(history["learning_rates"]):
                        ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

        ax.set_title("Learning Rate Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. Fair 30-epoch comparison (all models normalized to 30 epochs)
        ax = axes[1, 1]

        # Plot other models (original 30 epochs)
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, min(31, len(history["train_losses"]) + 1))
                train_losses = history["train_losses"][:30]
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, train_losses,
                    label=f"{display_name}",
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE's last 30 epochs (61-90 epochs)
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "train_losses" in history and len(history["train_losses"]) >= 90:
                # Take last 30 epochs (61-90)
                mmoe_last_30 = history["train_losses"][60:90]
                epochs = range(1, len(mmoe_last_30) + 1)
                ax.plot(
                    epochs, mmoe_last_30,
                    label=f"{mmoe_display_name} (Stage 3)",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

        ax.set_title("Fair 30-Epoch Comparison\n(MMoE: Final Stage vs Others: Full Training)")
        ax.set_xlabel("Epoch (within 30-epoch window)")
        ax.set_ylabel("Training Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        save_path = self.output_dir / "12_training_comparison_all_models_with_mmoe.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Complete model training comparison saved to: {save_path}")

    def plot_mmoe_stage_performance_analysis(self):
        """Plot 13: MMoE Stage Performance Analysis - Single chart"""
        # Find MMoE model
        mmoe_results = None
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMoE model results not found")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("MMoE Stage-by-Stage Performance Analysis", fontsize=16, fontweight="bold")

        history = mmoe_results["training_history"]

        if "train_losses" not in history or len(history["train_losses"]) < 90:
            print("❌ MMoE training history insufficient for stage analysis")
            return

        # Define three stages
        stages = {
            "Stage 1: Temporal Modeling": (0, 30),
            "Stage 2: CF Modeling": (30, 60),
            "Stage 3: MMoE Fusion": (60, 90),
        }

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

        # 1. Average loss by stage
        ax = axes[0, 0]
        stage_avg_losses = []
        stage_names = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                avg_loss = np.mean(stage_losses)
                stage_avg_losses.append(avg_loss)
                stage_names.append(stage_name.split(":")[0])

        bars = ax.bar(stage_names, stage_avg_losses, color=colors[:len(stage_names)])
        ax.set_title("Average Training Loss by Stage")
        ax.set_ylabel("Average MSE Loss")

        for bar, loss in zip(bars, stage_avg_losses):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{loss:.4f}", ha="center", va="bottom")

        # 2. Loss reduction analysis
        ax = axes[0, 1]
        loss_reductions = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                initial_loss = stage_losses[0]
                final_loss = stage_losses[-1]
                reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                loss_reductions.append(reduction)

        bars = ax.bar(stage_names, loss_reductions, color=colors[:len(stage_names)])
        ax.set_title("Loss Reduction by Stage")
        ax.set_ylabel("Loss Reduction (%)")

        for bar, reduction in zip(bars, loss_reductions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{reduction:.1f}%", ha="center", va="bottom")

        # 3. Learning rate decay analysis
        ax = axes[0, 2]
        if "learning_rates" in history and len(history["learning_rates"]) >= 90:
            lr_changes = []

            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                stage_lrs = history["learning_rates"][start:end]
                if stage_lrs:
                    initial_lr = stage_lrs[0]
                    final_lr = stage_lrs[-1]
                    change = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
                    lr_changes.append(change)

            bars = ax.bar(stage_names, lr_changes, color=colors[:len(stage_names)])
            ax.set_title("Learning Rate Decay by Stage")
            ax.set_ylabel("LR Decay (%)")

            for bar, change in zip(bars, lr_changes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                       f"{change:.1f}%", ha="center", va="bottom")

        # 4. Training stability analysis (loss variance)
        ax = axes[1, 0]
        loss_stds = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                loss_std = np.std(stage_losses)
                loss_stds.append(loss_std)

        bars = ax.bar(stage_names, loss_stds, color=colors[:len(stage_names)])
        ax.set_title("Training Stability by Stage")
        ax.set_ylabel("Loss Standard Deviation")

        for bar, std in zip(bars, loss_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{std:.4f}", ha="center", va="bottom")

        # 5. Stage transition loss changes
        ax = axes[1, 1]
        stage_endpoints = [30, 60, 90]
        endpoint_losses = [
            history["train_losses"][ep - 1]
            for ep in stage_endpoints
            if ep <= len(history["train_losses"])
        ]

        x_pos = range(len(endpoint_losses))
        ax.plot(x_pos, endpoint_losses, "o-", linewidth=3, markersize=8, color="#FF1744")
        ax.set_title("Loss Evolution Across Stages")
        ax.set_ylabel("Training Loss at Stage End")
        ax.set_xlabel("Stage")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(stage_names)
        ax.grid(True, alpha=0.3)

        for i, loss in enumerate(endpoint_losses):
            ax.text(i, loss + max(endpoint_losses)*0.01, f"{loss:.4f}", ha="center", va="bottom")

        # 6. Final test performance
        ax = axes[1, 2]
        test_metrics = mmoe_results["test_metrics"]
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        values = [test_metrics[metric] for metric in metrics if metric in test_metrics]
        metrics = [metric for metric in metrics if metric in test_metrics]

        bars = ax.bar(metrics, values, color="#9C27B0")
        ax.set_title("Final MMoE Test Performance")
        ax.set_ylabel("Metric Value")

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.4f}", ha="center", va="bottom")

        plt.tight_layout()
        
        save_path = self.output_dir / "13_mmoe_stage_performance_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ MMoE stage performance analysis saved to: {save_path}")

    def plot_performance_comparison(self):
        """Plot 14: Performance Metrics Comparison - Single chart"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        fig.suptitle(
            "Comprehensive Model Performance Comparison (All Models)",
            fontsize=16,
            fontweight="bold",
        )

        # Organize data, ensuring baseline first, MMoE last
        baseline_item = None
        mmoe_item = None
        other_items = []

        for model_type, results in self.results.items():
            display_name = self.get_display_name(model_type, results)
            if "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_item = (model_type, results, display_name)
            elif "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_item = (model_type, results, display_name)
            else:
                other_items.append((model_type, results, display_name))

        # Reorder: baseline -> other models -> MMoE
        sorted_items = []
        if baseline_item:
            sorted_items.append(baseline_item)
        sorted_items.extend(other_items)
        if mmoe_item:
            sorted_items.append(mmoe_item)

        # Reorganize data
        model_names = [display_name for _, _, display_name in sorted_items]

        print(f"Performance comparison includes {len(model_names)} models:")
        for i, name in enumerate(model_names):
            model_type = ("Baseline" if i == 0 else 
                         ("MMoE" if i == len(model_names) - 1 else "Time-Aware"))
            print(f"  {i+1}. {name} ({model_type})")

        # Main performance indicators
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        metric_values = {metric: [] for metric in metrics}

        for _, results, _ in sorted_items:
            test_metrics = results["test_metrics"]
            for metric in metrics:
                metric_values[metric].append(test_metrics[metric])

        # Plot main indicators
        positions = np.arange(len(model_names))

        # Use gradient colors, especially highlight baseline and MMoE
        colors = sns.color_palette("husl", len(model_names))
        if baseline_item:
            colors[0] = "#FF6B6B"  # Red highlight baseline
        if mmoe_item:
            colors[-1] = "#9C27B0"  # Purple highlight MMoE

        # RMSE comparison
        ax = axes[0, 0]
        bars = ax.bar(positions, metric_values["RMSE"], color=colors)
        ax.set_title("RMSE Comparison (Lower is Better)", fontweight="bold")
        ax.set_ylabel("RMSE")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        # Add numeric labels
        for bar, value in zip(bars, metric_values["RMSE"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.4f}", ha="center", va="bottom", fontsize=9)

        # MAE comparison
        ax = axes[0, 1]
        bars = ax.bar(positions, metric_values["MAE"], color=colors)
        ax.set_title("MAE Comparison (Lower is Better)", fontweight="bold")
        ax.set_ylabel("MAE")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, metric_values["MAE"]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.4f}", ha="center", va="bottom", fontsize=9)

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
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 0.01,
                   f"{value:.4f}", ha="center", va="bottom", fontsize=9)

        # Model complexity comparison
        ax = axes[1, 0]
        param_counts = [results["model_params"]["total_params"] for _, results, _ in sorted_items]
        bars = ax.bar(positions, param_counts, color=colors)
        ax.set_title("Model Parameter Count", fontweight="bold")
        ax.set_ylabel("Parameters")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            label = f'{value//1000}K' if value >= 1000 else str(value)
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   label, ha="center", va="bottom", fontsize=9)

        # Training time comparison
        ax = axes[1, 1]
        training_times = [results["training_history"]["total_training_time"] for _, results, _ in sorted_items]
        bars = ax.bar(positions, training_times, color=colors)
        ax.set_title("Training Time", fontweight="bold")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.1f}s", ha="center", va="bottom", fontsize=9)

        # Inference time comparison
        ax = axes[1, 2]
        inference_times = [results["test_metrics"]["Inference_Time"] for _, results, _ in sorted_items]
        bars = ax.bar(positions, inference_times, color=colors)
        ax.set_title("Inference Time", fontweight="bold")
        ax.set_ylabel("Time (seconds)")
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha="right", fontsize=10)

        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                   f"{value:.3f}s", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        
        save_path = self.output_dir / "14_performance_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Performance comparison saved to: {save_path}")

    def plot_efficiency_analysis(self):
        """Plot 15: Model Efficiency Analysis - Single chart"""
        plt.figure(figsize=(16, 10))
        
        organized_models = self.organize_models()
        
        # Extract data for efficiency analysis
        rmse_values = []
        param_counts = []
        training_times = []
        model_names = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            model_params = results.get("model_params", {})
            training_history = results.get("training_history", {})
            
            rmse = test_metrics.get("RMSE", 0)
            params = model_params.get("total_params", 0)
            train_time = training_history.get("total_training_time", 0)
            
            rmse_values.append(rmse)
            param_counts.append(params)
            training_times.append(train_time)
            model_names.append(display_name)
        
        # Create efficiency scatter plot
        # Bubble size represents training time, x-axis is parameters, y-axis is RMSE (lower is better)
        colors = []
        for model_type, _, _ in organized_models:
            if "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#2196F3")  # Blue
        
        # Normalize bubble sizes
        max_time = max(training_times)
        bubble_sizes = [(time / max_time * 1000 + 100) for time in training_times]
        
        scatter = plt.scatter(param_counts, rmse_values, s=bubble_sizes, c=colors, alpha=0.7, edgecolors='black')
        
        # Add model labels
        for i, name in enumerate(model_names):
            plt.annotate(name, (param_counts[i], rmse_values[i]), 
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, ha='left')
        
        plt.xlabel("Model Parameters", fontsize=12)
        plt.ylabel("RMSE (Lower is Better)", fontsize=12)
        plt.title("Model Efficiency Analysis\n(Bubble size represents training time)", fontsize=16, fontweight='bold')
        
        # Add legend for bubble sizes
        legend_sizes = [100, 500, 1000]
        legend_labels = ["Low", "Medium", "High"]
        legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.6) for size in legend_sizes]
        plt.legend(legend_elements, legend_labels, title="Training Time", loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "15_efficiency_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✅ Efficiency analysis saved to: {save_path}")

    def run_complete_analysis(self):
        """Run complete analysis workflow - Generate all individual charts"""
        print("🚀 Starting comprehensive model comparison analysis...")
        print(f"📁 Output directory: {self.output_dir}")
        
        if not self.results:
            print("❌ No model results found. Please run training scripts first.")
            return None
        
        organized_models = self.organize_models()
        print(f"📊 Found {len(organized_models)} models for analysis:")
        for i, (_, _, display_name) in enumerate(organized_models, 1):
            print(f"  {i}. {display_name}")
        
        print("\n" + "="*60)
        print("🎨 Generating individual comparison charts...")
        print("="*60)
        
        # Generate all individual charts
        charts = [
            ("Training Loss", self.plot_training_loss_comparison),
            ("Validation Loss", self.plot_validation_loss_comparison),
            ("RMSE", self.plot_rmse_comparison),
            ("MAE", self.plot_mae_comparison),
            ("Correlation", self.plot_correlation_comparison),
            ("Parameter Count", self.plot_parameter_count_comparison),
            ("Training Time", self.plot_training_time_comparison),
            ("Inference Time", self.plot_inference_time_comparison),
            ("MMoE Stage Analysis", self.plot_mmoe_stage_analysis),
            ("Summary Table", self.create_summary_table_image),
            ("MMoE Detailed Training", self.plot_mmoe_detailed_training_curves),
            ("Training Comparison with MMoE", self.plot_training_curves_with_mmoe_comparison),
            ("MMoE Stage Performance", self.plot_mmoe_stage_performance_analysis),
            ("Performance Comparison", self.plot_performance_comparison),
            ("Efficiency Analysis", self.plot_efficiency_analysis)
        ]
        
        successful_charts = 0
        for chart_name, chart_function in charts:
            try:
                print(f"\n📈 Generating {chart_name} chart...")
                chart_function()
                successful_charts += 1
            except Exception as e:
                print(f"❌ Failed to generate {chart_name} chart: {e}")
        
        print(f"\n" + "="*60)
        print(f"✅ Analysis completed!")
        print(f"📊 Successfully generated {successful_charts}/{len(charts)} charts")
        print(f"📁 All charts saved to: {self.output_dir}")
        print("="*60)
        
        # List all generated files
        generated_files = list(self.output_dir.glob("*.png"))
        generated_files.sort()
        
        print(f"\n📋 Generated files:")
        for file_path in generated_files:
            print(f"  📄 {file_path.name}")
        
        return successful_charts

def main():
    """Main function"""
    print("🔍 Starting model comparison analysis...")
    
    # Create model comparison analyzer
    analyzer = ModelComparison()
    
    if not analyzer.results:
        print("❌ No results found. Please run model training first.")
        return None, None
    
    # Run complete analysis
    chart_count = analyzer.run_complete_analysis()
    
    print(f"\n🎉 Model comparison analysis completed!")
    print(f"📊 Generated {chart_count} charts")
    
    return analyzer, chart_count

if __name__ == "__main__":
    analyzer, summary = main()