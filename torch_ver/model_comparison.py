import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
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
                results_path = possible_paths[0]

        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = Path(data_path) / "analysis_plots"
        self.output_dir.mkdir(exist_ok=True)
        
        self.split_mmoe_stages = True
        
        # 🔧 修复：更准确的模型名称映射
        self.model_name_mapping = {
            "UserTimeModel": "User Time-Aware Model",
            "UserTime": "User Time-Aware Model",
            "IndependentTimeModel": "Independent Time Feature Model", 
            "IndependentTime": "Independent Time Feature Model",  # 修复映射
            "UMTimeModel": "User-Movie Time-Aware Model",
            "UMTime": "User-Movie Time-Aware Model",
            "TwoStageMMoE": "Two-Stage MMoE Model",
            "TwoStage_MMoE": "Two-Stage MMoE Model",
            "CFModel_Baseline": "Baseline Collaborative Filtering",
            # MMoE阶段映射
            "MMoE_Stage1": "MMoE Stage 1: Temporal",
            "MMoE_Stage2": "MMoE Stage 2: CF",
            "MMoE_Stage3": "MMoE Stage 3: Fusion"
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
            print("🔍 Loaded models:")
            for model_type, result_data in results.items():
                model_name = result_data.get("model_name", "Unknown")
                epochs = len(result_data.get("training_history", {}).get("train_losses", []))
                print(f"  - {model_type}: '{model_name}' ({epochs} epochs)")
                
                # 检查名称不匹配的情况
                if model_type == "IndependentTime" and model_name != "Independent Time Feature Model":
                    print(f"    ⚠️  名称不匹配! 类型={model_type}, 名称={model_name}")
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

    def process_mmoe_stages(self):
        """🔧 修改：使用MMoE训练历史中的阶段信息来分割，而不是固定轮数"""
        if not self.split_mmoe_stages:
            return self.results
        
        processed_results = {}
        
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                # 处理MMoE模型，使用阶段信息来拆分
                history = results.get("training_history", {})
                train_losses = history.get("train_losses", [])
                val_losses = history.get("val_losses", [])
                learning_rates = history.get("learning_rates", [])
                epoch_times = history.get("epoch_times", [])
                stage_info = history.get("stage_info", [])  # 🔧 新增：获取阶段信息
                
                total_epochs = len(train_losses)
                print(f"Processing MMoE model with {total_epochs} epochs into stages...")
                
                # 🔧 修改：优先使用阶段信息，如果没有则回退到固定边界
                if stage_info and len(stage_info) > 0:
                    print(f"  Found {len(stage_info)} stage info entries, using stage boundaries")
                    
                    # 使用阶段信息来确定边界
                    stage_boundaries = []
                    stage_names = []
                    stage_display_names = []
                    
                    for i, stage in enumerate(stage_info):
                        start_epoch = stage.get("start_epoch", 0)
                        end_epoch = stage.get("end_epoch", start_epoch + stage.get("epochs", 10) - 1)
                        stage_name = stage.get("stage_name", f"Stage{i+1}")
                        
                        # 转换为实际的边界格式 (start, end+1)，因为切片是左闭右开
                        boundary_start = start_epoch
                        boundary_end = end_epoch + 1
                        
                        # 确保边界不超过实际数据长度
                        boundary_end = min(boundary_end, total_epochs)
                        
                        if boundary_end > boundary_start:
                            stage_boundaries.append((boundary_start, boundary_end))
                            
                            # 生成阶段名称
                            if "temporal" in stage_name.lower():
                                display_name = "MMoE Stage 1: Temporal"
                                type_name = "MMoE_Stage1"
                            elif "cf" in stage_name.lower():
                                display_name = "MMoE Stage 2: CF"
                                type_name = "MMoE_Stage2"
                            elif "mmoe" in stage_name.lower() or "fusion" in stage_name.lower():
                                display_name = "MMoE Stage 3: Fusion"
                                type_name = "MMoE_Stage3"
                            else:
                                display_name = f"MMoE {stage_name}"
                                type_name = f"MMoE_Stage{i+1}"
                            
                            stage_names.append(type_name)
                            stage_display_names.append(display_name)
                            
                            print(f"    Stage {i+1}: {stage_name} -> Epochs {boundary_start+1}-{boundary_end} ({boundary_end-boundary_start} epochs)")
                    
                else:
                    print("  No stage info found, using fixed epoch boundaries")
                    # 🔧 回退：如果没有阶段信息，使用原来的固定边界逻辑
                    if total_epochs >= 90:
                        stage_boundaries = [(0, 30), (30, 60), (60, 90)]
                        stage_names = ["MMoE_Stage1", "MMoE_Stage2", "MMoE_Stage3"]
                        stage_display_names = [
                            "MMoE Stage 1: Temporal", 
                            "MMoE Stage 2: CF", 
                            "MMoE Stage 3: Fusion"
                        ]
                    elif total_epochs >= 60:
                        stage_boundaries = [(0, 30), (30, 60), (60, total_epochs)]
                        stage_names = ["MMoE_Stage1", "MMoE_Stage2", "MMoE_Stage3"]
                        stage_display_names = [
                            "MMoE Stage 1: Temporal", 
                            "MMoE Stage 2: CF", 
                            "MMoE Stage 3: Fusion (Early Stop)"
                        ]
                    elif total_epochs >= 30:
                        stage_boundaries = [(0, 30), (30, total_epochs)]
                        stage_names = ["MMoE_Stage1", "MMoE_Stage2"]
                        stage_display_names = [
                            "MMoE Stage 1: Temporal", 
                            "MMoE Stage 2: CF (Early Stop)"
                        ]
                    else:
                        stage_boundaries = [(0, total_epochs)]
                        stage_names = ["MMoE_Stage1"]
                        stage_display_names = ["MMoE Stage 1: Temporal (Early Stop)"]
                
                # 为每个阶段创建独立的"模型"结果
                for i, ((start, end), stage_name, display_name) in enumerate(zip(stage_boundaries, stage_names, stage_display_names)):
                    if end > start:
                        # 提取该阶段的训练历史
                        stage_train_losses = train_losses[start:end] if train_losses else []
                        stage_val_losses = val_losses[start:end] if val_losses and end <= len(val_losses) else []
                        stage_learning_rates = learning_rates[start:end] if learning_rates and end <= len(learning_rates) else []
                        stage_epoch_times = epoch_times[start:end] if epoch_times and end <= len(epoch_times) else []
                        
                        # 计算阶段统计
                        stage_epochs = end - start
                        stage_training_time = sum(stage_epoch_times) if stage_epoch_times else 0
                        
                        # 🔧 优先使用阶段信息中的训练时间
                        if stage_info and i < len(stage_info):
                            stage_training_time = stage_info[i].get("training_time", stage_training_time)
                        
                        # 找到该阶段的最佳验证损失轮次
                        if stage_val_losses:
                            best_epoch_in_stage = stage_val_losses.index(min(stage_val_losses))
                        else:
                            best_epoch_in_stage = len(stage_train_losses) - 1 if stage_train_losses else 0
                        
                        # 创建阶段专用的训练历史
                        stage_history = {
                            "train_losses": stage_train_losses,
                            "val_losses": stage_val_losses,
                            "train_rmse": [math.sqrt(loss) for loss in stage_train_losses] if stage_train_losses else [],
                            "val_rmse": [math.sqrt(loss) for loss in stage_val_losses] if stage_val_losses else [],
                            "learning_rates": stage_learning_rates,
                            "epoch_times": stage_epoch_times,
                            "best_epoch": best_epoch_in_stage,
                            "total_epochs": stage_epochs,
                            "total_training_time": stage_training_time,
                        }
                        
                        # 为每个阶段使用原始MMoE的测试结果
                        stage_results = {
                            "model_name": display_name,
                            "model_type": stage_name,
                            "training_history": stage_history,
                            "test_metrics": results.get("test_metrics", {}),
                            "model_params": results.get("model_params", {}),
                            "training_config": results.get("training_config", {}),
                            # 标记这是MMoE的某个阶段
                            "is_mmoe_stage": True,
                            "stage_number": i + 1,
                            "stage_range": f"Epochs {start+1}-{end}",
                            "original_mmoe_results": results
                        }
                        
                        processed_results[stage_name] = stage_results
                        print(f"  Created {display_name}: {stage_epochs} epochs ({start+1}-{end}), training time: {stage_training_time:.1f}s")
                
                print(f"MMoE model split into {len(stage_boundaries)} independent stages")
                
            else:
                # 非MMoE模型保持不变
                processed_results[model_type] = results
        
        return processed_results

    def organize_models(self):
        """🔧 修改：组织模型，包含MMoE阶段"""
        # Process MMoE stages first
        results_to_use = self.process_mmoe_stages() if self.split_mmoe_stages else self.results
        
        baseline_models = []
        time_aware_models = []
        mmoe_models = []
        mmoe_stages = []
        
        for model_type, results in results_to_use.items():
            display_name = self.get_display_name(model_type, results)
            
            if results.get("is_mmoe_stage", False):
                # MMoE stage models
                mmoe_stages.append((model_type, results, display_name))
            elif "baseline" in model_type.lower() or "cfmodel" in model_type.lower():
                baseline_models.append((model_type, results, display_name))
            elif "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                # Original MMoE model (if not splitting stages)
                mmoe_models.append((model_type, results, display_name))
            else:
                time_aware_models.append((model_type, results, display_name))
        
        # Combine order: baseline -> time-aware -> mmoe stages -> original mmoe
        if self.split_mmoe_stages:
            # If stages are split, use stages instead of original MMoE
            organized_models = baseline_models + time_aware_models + mmoe_stages
        else:
            # If not split, use original MMoE
            organized_models = baseline_models + time_aware_models + mmoe_models
        
        return organized_models
    
    def get_display_name(self, model_type, results):
        """🔧 修改：获取显示名称，支持MMoE阶段"""
        if model_type in self.model_name_mapping:
            return self.model_name_mapping[model_type]
        
        # 检查是否是MMoE阶段
        if results.get("is_mmoe_stage", False):
            return results.get("model_name", model_type)
        
        # 获取结果中的模型名称
        model_name = results.get("model_name", model_type)
        
        # 🔧 修复：针对错误名称的特殊处理
        name_corrections = {
            "User Time-Aware Model": {
                "IndependentTime": "Independent Time Feature Model",
                "IndependentTimeModel": "Independent Time Feature Model"
            }
        }
        
        # 如果模型名称和类型不匹配，使用正确的名称
        if model_name in name_corrections:
            if model_type in name_corrections[model_name]:
                corrected_name = name_corrections[model_name][model_type]
                print(f"🔧 修正模型名称: {model_type} -> {corrected_name} (原为: {model_name})")
                return corrected_name
        
        # 增强的英文映射
        english_mapping = {
            "Baseline Collaborative Filtering": "Baseline Collaborative Filtering",
            "User Time-Aware Model": "User Time-Aware Model",
            "Independent Time Feature Model": "Independent Time Feature Model", 
            "User-Movie Time-Aware Model": "User-Movie Time-Aware Model",
            "Two-Stage MMoE Model (Optimized)": "Two-Stage MMoE Model",
            "Two-Stage MMoE Model": "Two-Stage MMoE Model",
        }
        
        clean_name = english_mapping.get(model_name, model_name)
        
        # 进一步清理
        clean_name = (clean_name.replace("模型", "Model")
                                .replace("基线", "Baseline")
                                .replace("优化", "Optimized")
                                .replace("协同过滤", "Collaborative Filtering")
                                .replace("时间感知", "Time-Aware")
                                .replace("用户", "User")
                                .replace("电影", "Movie")
                                .replace("独立", "Independent")
                                .replace("特征", "Feature")
                                .replace("两阶段", "Two-Stage"))
        
        return clean_name


    def plot_training_loss_comparison(self):
        """Plot 1: Training Loss Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        colors = plt.cm.Set3(np.linspace(0, 1, len(organized_models)))
        
        # 🔧 MMoE阶段专用颜色
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]  # 红、青绿、蓝
        
        print(f"Plotting training loss for {len(organized_models)} models...")
        
        mmoe_stage_count = 0
        for idx, (model_type, results, display_name) in enumerate(organized_models):
            history = results.get("training_history", {})
            train_losses = history.get("train_losses", [])
            
            if train_losses:
                # 🔧 所有模型都从epoch 1开始
                epochs = range(1, len(train_losses) + 1)
                
                # 🔧 为MMoE阶段分配特殊颜色
                if results.get("is_mmoe_stage", False):
                    color = mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)]
                    linewidth = 2.5
                    alpha = 0.9
                    mmoe_stage_count += 1
                    stage_info = f" ({len(train_losses)} epochs)"
                else:
                    color = colors[idx]
                    linewidth = 2
                    alpha = 0.8
                    if "mmoe" in model_type.lower():
                        stage_info = f" (90 epochs)"
                    else:
                        stage_info = f" (30 epochs)"
                
                plt.plot(epochs, train_losses, 
                       label=f"{display_name}{stage_info}", 
                       color=color, linewidth=linewidth, alpha=alpha,
                       linestyle='-')
                
                print(f"  ✅ {display_name}: {len(train_losses)} epochs")
            else:
                print(f"  ⚠️  {display_name}: No training loss data")
        
        plt.title("Training Loss Evolution Comparison (All Models + MMoE Stages)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs (Relative to Each Model/Stage)", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "01_training_loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training loss comparison saved to: {save_path}")

    def plot_validation_loss_comparison(self):
        """Plot 2: Validation Loss Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        colors = plt.cm.Set3(np.linspace(0, 1, len(organized_models)))
        
        # MMoE阶段专用颜色
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        
        print(f"Plotting validation loss for {len(organized_models)} models...")
        
        mmoe_stage_count = 0
        for idx, (model_type, results, display_name) in enumerate(organized_models):
            history = results.get("training_history", {})
            val_losses = history.get("val_losses", [])
            
            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                
                if results.get("is_mmoe_stage", False):
                    color = mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)]
                    linewidth = 2.5
                    alpha = 0.9
                    mmoe_stage_count += 1
                    stage_info = f" ({len(val_losses)} epochs)"
                else:
                    color = colors[idx]
                    linewidth = 2
                    alpha = 0.8
                    if "mmoe" in model_type.lower():
                        stage_info = f" (90 epochs)"
                    else:
                        stage_info = f" (30 epochs)"
                
                plt.plot(epochs, val_losses, 
                       label=f"{display_name}{stage_info}", 
                       color=color, linewidth=linewidth, alpha=alpha,
                       linestyle='-')
                
                print(f"  ✅ {display_name}: {len(val_losses)} epochs")
            else:
                print(f"  ⚠️  {display_name}: No validation loss data")
        
        plt.title("Validation Loss Evolution Comparison (All Models + MMoE Stages)", fontsize=16, fontweight='bold')
        plt.xlabel("Training Epochs (Relative to Each Model/Stage)", fontsize=12)
        plt.ylabel("MSE Loss", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "02_validation_loss_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Validation loss comparison saved to: {save_path}")

    def plot_rmse_comparison(self):
        """Plot 3: RMSE Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
        
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        rmse_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            rmse = test_metrics.get("RMSE", 0)
            rmse_values.append(rmse)
        
        # 🔧 颜色映射：基线=红色，时间感知=蓝色，MMoE阶段=渐变色，原始MMoE=紫色
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
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
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.title("RMSE Comparison (Lower is Better) - Including MMoE Stages", fontsize=16, fontweight='bold')
        plt.ylabel("RMSE", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "03_rmse_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ RMSE comparison saved to: {save_path}")

    def plot_mae_comparison(self):
        """Plot 4: MAE Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
    
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        mae_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            mae = test_metrics.get("MAE", 0)
            mae_values.append(mae)
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#4CAF50")  # Green for time-aware
        
        bars = plt.bar(range(len(model_names)), mae_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, mae_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.title("MAE Comparison (Lower is Better) - Including MMoE Stages", fontsize=16, fontweight='bold')
        plt.ylabel("MAE", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "04_mae_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ MAE comparison saved to: {save_path}")

    def plot_correlation_comparison(self):
        """Plot 5: Correlation Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
    
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        correlation_values = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            correlation = test_metrics.get("Correlation", 0)
            correlation_values.append(correlation)
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#FF9800")  # Orange for time-aware
        
        bars = plt.bar(range(len(model_names)), correlation_values, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, correlation_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.title("Prediction Correlation (Higher is Better) - Including MMoE Stages", fontsize=16, fontweight='bold')
        plt.ylabel("Correlation Coefficient", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "05_correlation_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Correlation comparison saved to: {save_path}")

    def plot_parameter_count_comparison(self):
        """Plot 6: Parameter Count Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
    
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        param_counts = []
        
        for model_type, results, display_name in organized_models:
            model_params = results.get("model_params", {})
            total_params = model_params.get("total_params", 0)
            param_counts.append(total_params)
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#607D8B")  # Blue Grey for time-aware
        
        bars = plt.bar(range(len(model_names)), param_counts, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels (show in K format)
        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            label = f'{value//1000}K' if value >= 1000 else str(value)
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    label, ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.title("Model Parameter Count Comparison - Including MMoE Stages", fontsize=16, fontweight='bold')
        plt.ylabel("Parameter Count", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "06_parameter_count_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Parameter count comparison saved to: {save_path}")

    def plot_training_time_comparison(self):
        """Plot 7: Training Time Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
    
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        training_times = []
        
        for model_type, results, display_name in organized_models:
            training_history = results.get("training_history", {})
            training_time = training_history.get("total_training_time", 0)
            training_times.append(training_time)
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#795548")  # Brown for time-aware
        
        bars = plt.bar(range(len(model_names)), training_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels with epoch info
        for i, (bar, value) in enumerate(zip(bars, training_times)):
            height = bar.get_height()
            model_type, results, _ = organized_models[i]
            
            # 添加轮数信息
            if results.get("is_mmoe_stage", False):
                epoch_info = f"\n({results.get('training_history', {}).get('total_epochs', 0)}ep)"
            elif "mmoe" in model_type.lower():
                epoch_info = "\n(90ep)"
            else:
                epoch_info = "\n(30ep)"
            
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.1f}s{epoch_info}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        plt.title("Training Time Comparison - Including MMoE Stages\n(Each stage shows its individual training time)", 
                fontsize=16, fontweight='bold')
        plt.ylabel("Training Time (seconds)", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "07_training_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Training time comparison saved to: {save_path}")

    def plot_inference_time_comparison(self):
        """Plot 8: Inference Time Comparison - Single chart"""
        plt.figure(figsize=(14, 8))
    
        organized_models = self.organize_models()
        model_names = [display_name for _, _, display_name in organized_models]
        inference_times = []
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            inference_time = test_metrics.get("Inference_Time", 0)
            inference_times.append(inference_time)
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#009688")  # Teal for time-aware
        
        bars = plt.bar(range(len(model_names)), inference_times, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.title("Inference Time Comparison - Including MMoE Stages\n(All stages use same final model for inference)", 
                fontsize=16, fontweight='bold')
        plt.ylabel("Inference Time (seconds)", fontsize=12)
        plt.xticks(range(len(model_names)), model_names, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        save_path = self.output_dir / "08_inference_time_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Inference time comparison saved to: {save_path}")

    def plot_mmoe_stage_analysis(self):
        """Plot 9: MMoE Stage Analysis - Single chart"""
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
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, _ in organized_models:
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#2196F3")  # Blue
        
        # Normalize bubble sizes (使用训练时间作为气泡大小)
        if max(training_times) > 0:
            bubble_sizes = [(time / max(training_times) * 1000 + 100) for time in training_times]
        else:
            bubble_sizes = [100] * len(training_times)
        
        scatter = plt.scatter(param_counts, rmse_values, s=bubble_sizes, c=colors, alpha=0.7, edgecolors='black')
        
        # Add model labels with epoch info
        for i, (name, (model_type, results, _)) in enumerate(zip(model_names, organized_models)):
            offset_x, offset_y = 5, 5
            if results.get("is_mmoe_stage", False):
                epochs = results.get("training_history", {}).get("total_epochs", 0)
                label = f"{name}\n({epochs}ep)"
            else:
                label = name
            
            plt.annotate(label, (param_counts[i], rmse_values[i]), 
                        xytext=(offset_x, offset_y), textcoords='offset points', 
                        fontsize=9, ha='left', va='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        
        plt.xlabel("Model Parameters", fontsize=12)
        plt.ylabel("RMSE (Lower is Better)", fontsize=12)
        plt.title("MMoE Stage Analysis - Including All Models\n(Bubble size represents training time, MMoE stages show individual performance)", 
                fontsize=16, fontweight='bold')
        
        # Add legend for bubble sizes
        if max(training_times) > 0:
            max_time = max(training_times)
            legend_times = [max_time * 0.2, max_time * 0.6, max_time]
            legend_sizes = [time / max_time * 1000 + 100 for time in legend_times]
            legend_labels = [f"{time:.1f}s" for time in legend_times]
            legend_elements = [plt.scatter([], [], s=size, c='gray', alpha=0.6) for size in legend_sizes]
            plt.legend(legend_elements, legend_labels, title="Training Time", loc='upper right')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / "09_mmoe_stage_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
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
            
            # 🔧 确定轮数信息
            total_epochs = training_history.get("total_epochs", 0)
            if results.get("is_mmoe_stage", False):
                stage_range = results.get("stage_range", f"{total_epochs} epochs")
                epochs_info = f"{total_epochs} ({stage_range})"
            elif "mmoe" in model_type.lower():
                epochs_info = "90 (30×3)"
            else:
                epochs_info = str(total_epochs) if total_epochs > 0 else "30"
            
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
        fig, ax = plt.subplots(figsize=(20, max(8, len(organized_models))))
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
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # 🔧 Color rows by model type including MMoE stages
        mmoe_stage_colors_light = ["#FFE5E5", "#E0F2F1", "#E3F2FD"]  # Light versions
        mmoe_stage_count = 0
        
        for i, (model_type, results, _) in enumerate(organized_models):
            row_idx = i + 1
            
            if results.get("is_mmoe_stage", False):
                color = mmoe_stage_colors_light[mmoe_stage_count % len(mmoe_stage_colors_light)]
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                color = '#FFE5E5'  # Light red
            elif "mmoe" in model_type.lower():
                color = '#F3E5F5'  # Light purple
            else:
                color = '#E3F2FD'  # Light blue
            
            for j in range(len(headers)):
                table[(row_idx, j)].set_facecolor(color)
        
        # Highlight best values (excluding MMoE stages since they use same test metrics)
        non_mmoe_indices = [i for i, (_, results, _) in enumerate(organized_models) 
                           if not results.get("is_mmoe_stage", False)]
        
        if non_mmoe_indices:
            non_mmoe_rmse = [float(table_data[i][1]) for i in non_mmoe_indices]
            non_mmoe_mae = [float(table_data[i][2]) for i in non_mmoe_indices]
            non_mmoe_corr = [float(table_data[i][4]) for i in non_mmoe_indices]
            
            best_rmse_relative_idx = non_mmoe_rmse.index(min(non_mmoe_rmse))
            best_mae_relative_idx = non_mmoe_mae.index(min(non_mmoe_mae))
            best_corr_relative_idx = non_mmoe_corr.index(max(non_mmoe_corr))
            
            best_rmse_idx = non_mmoe_indices[best_rmse_relative_idx]
            best_mae_idx = non_mmoe_indices[best_mae_relative_idx]
            best_corr_idx = non_mmoe_indices[best_corr_relative_idx]
            
            # Highlight best RMSE
            table[(best_rmse_idx + 1, 1)].set_facecolor('#C8E6C9')
            table[(best_rmse_idx + 1, 1)].set_text_props(weight='bold')
            
            # Highlight best MAE
            table[(best_mae_idx + 1, 2)].set_facecolor('#C8E6C9')
            table[(best_mae_idx + 1, 2)].set_text_props(weight='bold')
            
            # Highlight best Correlation
            table[(best_corr_idx + 1, 4)].set_facecolor('#C8E6C9')
            table[(best_corr_idx + 1, 4)].set_text_props(weight='bold')
        
        plt.title("Complete Model Performance Summary (Including MMoE Stages)\n(Green highlights indicate best performance among distinct models)", 
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        save_path = self.output_dir / "10_summary_table.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Summary table saved to: {save_path}")
        
        return table_data, headers

    def plot_mmoe_detailed_training_curves(self):
        """Plot 11: MMoE Detailed Stage Training - 使用阶段信息而非固定边界"""
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

        history = mmoe_results["training_history"]

        # Check training history data
        if "train_losses" not in history or not history["train_losses"]:
            print("❌ MMoE training history data not found")
            return

        total_epochs = len(history["train_losses"])
        stage_info = history.get("stage_info", [])
        
        print(f"MMoE total training epochs: {total_epochs}")

        # 🔧 修改：优先使用阶段信息
        if stage_info and len(stage_info) > 0:
            print(f"Using stage info with {len(stage_info)} stages")
            
            stages = {}
            for i, stage in enumerate(stage_info):
                start_epoch = stage.get("start_epoch", 0)
                end_epoch = stage.get("end_epoch", start_epoch + stage.get("epochs", 10) - 1)
                stage_name = stage.get("stage_name", f"Stage{i+1}")
                
                # 生成显示名称
                if "temporal" in stage_name.lower():
                    display_name = "Stage 1: Temporal Modeling"
                elif "cf" in stage_name.lower():
                    display_name = "Stage 2: CF Modeling"
                elif "mmoe" in stage_name.lower() or "fusion" in stage_name.lower():
                    display_name = "Stage 3: MMoE Fusion"
                else:
                    display_name = f"Stage {i+1}: {stage_name}"
                
                # 转换为边界格式 (注意+1因为切片是左闭右开)
                stages[display_name] = (start_epoch, min(end_epoch + 1, total_epochs))
                print(f"  {display_name}: epochs {start_epoch+1}-{end_epoch+1} ({stage.get('epochs', 0)} epochs)")
            
        else:
            print("No stage info found, using fixed boundaries")
            # 回退到原来的固定边界逻辑
            if total_epochs < 10:
                print(f"⚠️  MMoE training too short for stage analysis: {total_epochs} epochs")
                return

            # 根据实际轮数动态分阶段
            if total_epochs >= 90:
                stages = {
                    "Stage 1: Temporal Modeling": (0, 30),
                    "Stage 2: CF Modeling": (30, 60),
                    "Stage 3: MMoE Fusion": (60, 90),
                }
            elif total_epochs >= 60:
                stages = {
                    "Stage 1: Temporal Modeling": (0, 30),
                    "Stage 2: CF Modeling": (30, 60),
                    "Stage 3: MMoE Fusion (Early Stop)": (60, total_epochs),
                }
            elif total_epochs >= 30:
                stage_2_end = min(total_epochs, 60)
                stages = {
                    "Stage 1: Temporal Modeling": (0, 30),
                    "Stage 2: CF Modeling (Early Stop)": (30, stage_2_end),
                }
            else:
                stages = {
                    "Stage 1: Temporal Modeling (Early Stop)": (0, total_epochs),
                }

        fig, axes = plt.subplots(len(stages), 2, figsize=(16, 6 * len(stages)))
        fig.suptitle(
            f"MMoE Training Analysis ({total_epochs} Epochs Total)",
            fontsize=16,
            fontweight="bold",
        )
        
        # 如果只有一个阶段，调整axes的形状
        if len(stages) == 1:
            axes = axes.reshape(1, -1)

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
                    color=colors[idx % len(colors)],
                    linewidth=2,
                    label="Training Loss",
                )

                # If validation loss exists, plot it too
                if "val_losses" in history and history["val_losses"] and end <= len(history["val_losses"]):
                    stage_val_losses = history["val_losses"][start:end]
                    ax.plot(
                        stage_epochs_range,
                        stage_val_losses,
                        color=colors[idx % len(colors)],
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
            ax = axes[idx, 1]

            if ("learning_rates" in history and history["learning_rates"] and 
                end > start and end <= len(history["learning_rates"])):

                stage_epochs_range = range(start + 1, end + 1)
                stage_learning_rates = history["learning_rates"][start:end]

                ax.plot(
                    stage_epochs_range,
                    stage_learning_rates,
                    color=colors[idx % len(colors)],
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
        plt.close()
        print(f"✅ MMoE detailed training curves saved to: {save_path} ({total_epochs} epochs)")

    def plot_training_curves_with_mmoe_comparison(self):
        """Plot 12: Training Curves with MMoE Comparison - 使用动态阶段信息"""
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
            training_history = results.get("training_history", {})
            total_epochs = len(training_history.get("train_losses", []))
            
            model_info = {
                "type": model_type,
                "results": results,
                "name": display_name,
                "is_mmoe": False,
                "epochs": total_epochs,
            }

            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_data = (model_type, results)
                model_info["is_mmoe"] = True
            else:
                other_models.append((model_type, results))

            all_models_info.append(model_info)

        print(f"Found {len(all_models_info)} models for comparison:")
        for model_info in all_models_info:
            # 🔧 修复：使用实际轮数而不是硬编码
            epochs = model_info["epochs"]
            model_type_label = f" (MMoE - {epochs} epochs)" if model_info["is_mmoe"] else f" ({epochs} epochs)"
            print(f"  - {model_info['name']}{model_type_label}")

        # 1. Training loss comparison - show all models
        ax = axes[0, 0]

        # Plot other models
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)
                total_epochs = len(history["train_losses"])
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["train_losses"],
                    label=f"{display_name} ({total_epochs}ep)",  # 🔧 使用实际轮数
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE with dynamic stage boundaries
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                epochs = range(1, len(history["train_losses"]) + 1)
                total_epochs = len(history["train_losses"])

                # MMoE with thick line and special color
                ax.plot(
                    epochs, history["train_losses"],
                    label=f"{mmoe_display_name} ({total_epochs}ep)",  # 🔧 使用实际轮数
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # 🔧 修复：使用阶段信息来添加分隔线
                stage_info = history.get("stage_info", [])
                if stage_info:
                    print(f"Using MMoE stage info with {len(stage_info)} stages")
                    
                    # 添加阶段分隔线和标签
                    stage_labels = []
                    stage_positions = []
                    
                    for i, stage in enumerate(stage_info):
                        start_epoch = stage.get("start_epoch", 0)
                        end_epoch = stage.get("end_epoch", start_epoch + stage.get("epochs", 10) - 1)
                        stage_name = stage.get("stage_name", f"Stage{i+1}")
                        
                        # 添加分隔线（除了第一个阶段的开始）
                        if i > 0 and start_epoch < total_epochs:
                            ax.axvline(x=start_epoch, color="#FF1744", linestyle=":", alpha=0.5)
                        
                        # 计算标签位置（阶段中点）
                        mid_point = (start_epoch + end_epoch + 1) // 2
                        if mid_point < total_epochs:
                            stage_positions.append(mid_point)
                            
                            # 生成简短的标签名称
                            if "temporal" in stage_name.lower():
                                label = "Temporal"
                            elif "cf" in stage_name.lower():
                                label = "CF"
                            elif "mmoe" in stage_name.lower() or "fusion" in stage_name.lower():
                                label = "MMoE"
                            else:
                                label = stage_name
                            
                            stage_labels.append(label)

                    # 添加阶段标签
                    for pos, label in zip(stage_positions, stage_labels):
                        if pos < total_epochs:
                            ax.text(
                                pos, max(history["train_losses"]) * 0.9, label,
                                ha="center", va="center",
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
                                fontsize=9, color="#FF1744",
                            )
                else:
                    print("No stage info found for MMoE, using fixed boundaries as fallback")
                    # 回退到固定边界（但根据实际轮数调整）
                    if total_epochs >= 90:
                        stage_boundaries = [30, 60]
                        stage_labels = ["Temporal", "CF", "MMoE"]
                        stage_positions = [15, 45, 75]
                    elif total_epochs >= 60:
                        stage_boundaries = [30]
                        stage_labels = ["Temporal", "CF", "MMoE"]
                        stage_positions = [15, 45, total_epochs - 15]
                    elif total_epochs >= 30:
                        stage_boundaries = []
                        stage_labels = ["Temporal", "CF"]
                        stage_positions = [15, total_epochs - 15]
                    else:
                        stage_boundaries = []
                        stage_labels = ["Temporal"]
                        stage_positions = [total_epochs // 2]
                    
                    for boundary in stage_boundaries:
                        if boundary < total_epochs:
                            ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)
                    
                    for pos, label in zip(stage_positions, stage_labels):
                        if pos < total_epochs:
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

        # 2. Validation loss comparison - show all models (类似修复)
        ax = axes[0, 1]

        # Plot other models' validation loss
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                total_epochs = len(history["val_losses"])
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, history["val_losses"],
                    label=f"{display_name} ({total_epochs}ep)",  # 🔧 使用实际轮数
                    color=color, linewidth=2, alpha=0.8,
                )

        # Plot MMoE validation loss
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            if "val_losses" in history and history["val_losses"]:
                epochs = range(1, len(history["val_losses"]) + 1)
                total_epochs = len(history["val_losses"])
                ax.plot(
                    epochs, history["val_losses"],
                    label=f"{mmoe_display_name} ({total_epochs}ep)",  # 🔧 使用实际轮数
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # 🔧 添加阶段分隔线（使用阶段信息）
                stage_info = history.get("stage_info", [])
                if stage_info:
                    for i, stage in enumerate(stage_info[1:], 1):  # 跳过第一个阶段
                        start_epoch = stage.get("start_epoch", 0)
                        if start_epoch < total_epochs:
                            ax.axvline(x=start_epoch, color="#FF1744", linestyle=":", alpha=0.5)
                else:
                    # 回退到固定边界
                    boundaries = [30, 60] if total_epochs >= 90 else [30] if total_epochs >= 60 else []
                    for boundary in boundaries:
                        if boundary < total_epochs:
                            ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

        ax.set_title("Validation Loss Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("MSE Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 3. Learning rate evolution comparison (类似修复)
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
                total_epochs = len(history["learning_rates"])
                ax.plot(
                    epochs, history["learning_rates"],
                    label=f"{mmoe_display_name}",
                    color="#FF1744", linewidth=3, alpha=0.9,
                )

                # 🔧 添加阶段分隔线
                stage_info = history.get("stage_info", [])
                if stage_info:
                    for i, stage in enumerate(stage_info[1:], 1):
                        start_epoch = stage.get("start_epoch", 0)
                        if start_epoch < total_epochs:
                            ax.axvline(x=start_epoch, color="#FF1744", linestyle=":", alpha=0.5)
                else:
                    boundaries = [30, 60] if total_epochs >= 90 else [30] if total_epochs >= 60 else []
                    for boundary in boundaries:
                        if boundary < total_epochs:
                            ax.axvline(x=boundary, color="#FF1744", linestyle=":", alpha=0.5)

        ax.set_title("Learning Rate Evolution (All Models)")
        ax.set_xlabel("Training Epochs")
        ax.set_ylabel("Learning Rate")
        ax.set_yscale("log")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        # 4. 🔧 修复：公平比较 - 使用MMoE的最后一个阶段
        ax = axes[1, 1]

        # Plot other models (原始轮数)
        other_models_epochs = []
        for idx, (model_type, results) in enumerate(other_models):
            display_name = self.get_display_name(model_type, results)
            history = results["training_history"]
            if "train_losses" in history and history["train_losses"]:
                total_epochs = len(history["train_losses"])
                other_models_epochs.append(total_epochs)
                
                # 为了公平比较，可以截取前30轮或者全部轮数
                comparison_epochs = min(30, total_epochs)
                epochs = range(1, comparison_epochs + 1)
                train_losses = history["train_losses"][:comparison_epochs]
                color = colors[idx % len(colors)]
                ax.plot(
                    epochs, train_losses,
                    label=f"{display_name}",
                    color=color, linewidth=2, alpha=0.8,
                )

        # 🔧 修复：使用MMoE的最后一个阶段进行比较
        if mmoe_data:
            _, mmoe_results = mmoe_data
            mmoe_display_name = self.get_display_name(mmoe_data[0], mmoe_results)
            history = mmoe_results["training_history"]
            stage_info = history.get("stage_info", [])
            
            if stage_info and len(stage_info) > 0:
                # 使用最后一个阶段的数据
                last_stage = stage_info[-1]
                start_epoch = last_stage.get("start_epoch", 0)
                end_epoch = last_stage.get("end_epoch", start_epoch + last_stage.get("epochs", 10) - 1)
                stage_name = last_stage.get("stage_name", "Final Stage")
                
                if "train_losses" in history and end_epoch < len(history["train_losses"]):
                    # 提取最后阶段的损失
                    last_stage_losses = history["train_losses"][start_epoch:end_epoch + 1]
                    epochs = range(1, len(last_stage_losses) + 1)
                    ax.plot(
                        epochs, last_stage_losses,
                        label=f"{mmoe_display_name} ({stage_name})",
                        color="#FF1744", linewidth=3, alpha=0.9,
                    )
                    
                    # 确定比较轮数
                    comparison_epochs = len(last_stage_losses)
            else:
                # 回退：如果没有阶段信息，使用最后30轮
                if "train_losses" in history and len(history["train_losses"]) >= 30:
                    total_epochs = len(history["train_losses"])
                    if total_epochs >= 60:
                        # 使用最后30轮
                        mmoe_last_stage = history["train_losses"][-30:]
                        epochs = range(1, len(mmoe_last_stage) + 1)
                        ax.plot(
                            epochs, mmoe_last_stage,
                            label=f"{mmoe_display_name} (Final 30)",
                            color="#FF1744", linewidth=3, alpha=0.9,
                        )
                        comparison_epochs = 30

        # 动态设置标题
        avg_other_epochs = int(np.mean(other_models_epochs)) if other_models_epochs else 30
        ax.set_title(f"Fair Comparison\n(Others: ~{avg_other_epochs} epochs vs MMoE: Final Stage)")
        ax.set_xlabel("Epoch (within comparison window)")
        ax.set_ylabel("Training Loss")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        save_path = self.output_dir / "12_training_comparison_all_models_with_mmoe.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Complete model training comparison saved to: {save_path}")

    def plot_mmoe_stage_performance_analysis(self):
        """Plot 13: MMoE Stage Performance Analysis - 修复阶段分析"""
        # Find MMoE model
        mmoe_results = None
        for model_type, results in self.results.items():
            if "mmoe" in model_type.lower() or "twostage" in model_type.lower():
                mmoe_results = results
                break

        if mmoe_results is None:
            print("❌ MMoE model results not found")
            return

        history = mmoe_results["training_history"]
        if "train_losses" not in history:
            print("❌ MMoE training history not found")
            return
        
        total_epochs = len(history["train_losses"])
        stage_info = history.get("stage_info", [])
        
        if total_epochs < 3:
            print(f"❌ MMoE training history insufficient for stage analysis: {total_epochs} epochs")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"MMoE Stage-by-Stage Performance Analysis ({total_epochs} Epochs)", fontsize=16, fontweight="bold")

        # 🔧 修改：优先使用阶段信息
        if stage_info and len(stage_info) > 0:
            print(f"Using stage info with {len(stage_info)} stages")
            
            stages = {}
            stage_endpoints = []  # 🔧 新增：记录阶段端点
            stage_names_for_plot = []  # 🔧 新增：记录阶段名称
            
            for i, stage in enumerate(stage_info):
                start_epoch = stage.get("start_epoch", 0)
                end_epoch = stage.get("end_epoch", start_epoch + stage.get("epochs", 10) - 1)
                stage_name = stage.get("stage_name", f"Stage{i+1}")
                
                # 生成显示名称
                if "temporal" in stage_name.lower():
                    display_name = "Stage 1: Temporal Modeling"
                    short_name = "Stage 1"
                elif "cf" in stage_name.lower():
                    display_name = "Stage 2: CF Modeling"
                    short_name = "Stage 2"
                elif "mmoe" in stage_name.lower() or "fusion" in stage_name.lower():
                    display_name = "Stage 3: MMoE Fusion"
                    short_name = "Stage 3"
                else:
                    display_name = f"Stage {i+1}: {stage_name}"
                    short_name = f"Stage {i+1}"
                
                stages[display_name] = (start_epoch, min(end_epoch + 1, total_epochs))
                stage_endpoints.append(end_epoch)  # 🔧 记录端点
                stage_names_for_plot.append(short_name)  # 🔧 记录短名称
                
            print(f"Stage endpoints: {stage_endpoints}")
            print(f"Stage names: {stage_names_for_plot}")
            
        else:
            # 回退到固定边界逻辑
            print("Using fixed boundaries")
            if total_epochs >= 90:
                stages = {
                    "Stage 1: Temporal Modeling": (0, 30),
                    "Stage 2: CF Modeling": (30, 60),
                    "Stage 3: MMoE Fusion": (60, 90),
                }
                stage_endpoints = [29, 59, 89]  # 🔧 修复：0-based索引
                stage_names_for_plot = ["Stage 1", "Stage 2", "Stage 3"]
            elif total_epochs >= 60:
                stages = {
                    "Stage 1: Temporal Modeling": (0, 30),
                    "Stage 2: CF Modeling": (30, 60),
                    "Stage 3: MMoE Fusion": (60, total_epochs),
                }
                stage_endpoints = [29, 59, total_epochs - 1]
                stage_names_for_plot = ["Stage 1", "Stage 2", "Stage 3"]
            elif total_epochs >= 30:
                stage_2_end = min(total_epochs, 60)
                stages = {
                    "Stage 1: Temporal Modeling": (0, 30),
                    "Stage 2: CF Modeling": (30, stage_2_end),
                }
                stage_endpoints = [29, stage_2_end - 1]
                stage_names_for_plot = ["Stage 1", "Stage 2"]
            else:
                stages = {
                    "Stage 1: Temporal Modeling": (0, total_epochs),
                }
                stage_endpoints = [total_epochs - 1]
                stage_names_for_plot = ["Stage 1"]

        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"][:len(stages)]

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

        if stage_avg_losses:
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
            if stage_losses and len(stage_losses) > 1:
                initial_loss = stage_losses[0]
                final_loss = stage_losses[-1]
                reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
                loss_reductions.append(reduction)
            else:
                loss_reductions.append(0)

        if loss_reductions and len(loss_reductions) == len(stage_names):
            bars = ax.bar(stage_names, loss_reductions, color=colors[:len(stage_names)])
            ax.set_title("Loss Reduction by Stage")
            ax.set_ylabel("Loss Reduction (%)")

            for bar, reduction in zip(bars, loss_reductions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + max(loss_reductions)*0.01,
                    f"{reduction:.1f}%", ha="center", va="bottom")

        # 3. Learning rate decay analysis
        ax = axes[0, 2]
        if "learning_rates" in history and history["learning_rates"] and len(history["learning_rates"]) >= total_epochs:
            lr_changes = []

            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                stage_lrs = history["learning_rates"][start:end]
                if stage_lrs and len(stage_lrs) > 1:
                    initial_lr = stage_lrs[0]
                    final_lr = stage_lrs[-1]
                    change = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
                    lr_changes.append(change)
                else:
                    lr_changes.append(0)

            if lr_changes and len(lr_changes) == len(stage_names):
                bars = ax.bar(stage_names, lr_changes, color=colors[:len(stage_names)])
                ax.set_title("Learning Rate Decay by Stage")
                ax.set_ylabel("LR Decay (%)")

                for bar, change in zip(bars, lr_changes):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2.0, height + max(lr_changes)*0.01,
                        f"{change:.1f}%", ha="center", va="bottom")
        else:
            ax.text(0.5, 0.5, "Learning Rate Data\nNot Available", 
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title("Learning Rate Decay by Stage")

        # 4. Training stability analysis (loss variance)
        ax = axes[1, 0]
        loss_stds = []

        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            stage_losses = history["train_losses"][start:end]
            if stage_losses:
                loss_std = np.std(stage_losses)
                loss_stds.append(loss_std)

        if loss_stds and len(loss_stds) == len(stage_names):
            bars = ax.bar(stage_names, loss_stds, color=colors[:len(stage_names)])
            ax.set_title("Training Stability by Stage")
            ax.set_ylabel("Loss Standard Deviation")

            for bar, std in zip(bars, loss_stds):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, height + height*0.01,
                    f"{std:.4f}", ha="center", va="bottom")

        # 5. 🔧 修复：阶段过渡损失变化 - 使用动态端点
        ax = axes[1, 1]
        
        # 使用动态确定的端点
        endpoint_losses = []
        actual_labels = []
        
        for i, endpoint in enumerate(stage_endpoints):
            if endpoint < len(history["train_losses"]):
                endpoint_losses.append(history["train_losses"][endpoint])
                actual_labels.append(stage_names_for_plot[i])
                print(f"Endpoint {i}: epoch {endpoint+1}, loss {history['train_losses'][endpoint]:.4f}")

        if endpoint_losses:
            x_pos = range(len(endpoint_losses))
            ax.plot(x_pos, endpoint_losses, "o-", linewidth=3, markersize=8, color="#FF1744")
            ax.set_title("Loss Evolution Across Stages")
            ax.set_ylabel("Training Loss at Stage End")
            ax.set_xlabel("Stage")
            ax.set_xticks(x_pos)
            ax.set_xticklabels(actual_labels)
            ax.grid(True, alpha=0.3)

            for i, loss in enumerate(endpoint_losses):
                ax.text(i, loss + max(endpoint_losses)*0.01, f"{loss:.4f}", ha="center", va="bottom")
                
            print(f"✅ Loss evolution plot: {len(endpoint_losses)} points")
        else:
            print("❌ No valid endpoints found for loss evolution plot")

        # 6. Final test performance
        ax = axes[1, 2]
        test_metrics = mmoe_results["test_metrics"]
        metrics = ["RMSE", "MAE", "MAPE", "Correlation"]
        values = [test_metrics[metric] for metric in metrics if metric in test_metrics]
        metrics = [metric for metric in metrics if metric in test_metrics]

        if values:
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
        plt.close()
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
        plt.close()
        print(f"✅ Performance comparison saved to: {save_path}")

    def plot_efficiency_analysis(self):
        """Plot 15: Model Efficiency Analysis - Single chart"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle("Model Efficiency Analysis - Multi-Dimensional Comparison", fontsize=16, fontweight='bold')
        
        organized_models = self.organize_models()
        
        # 提取数据
        model_names = []
        rmse_values = []
        param_counts = []
        training_times = []
        inference_times = []
        
        # 🔧 颜色映射：包含MMoE阶段
        colors = []
        mmoe_stage_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]
        mmoe_stage_count = 0
        
        for model_type, results, display_name in organized_models:
            test_metrics = results.get("test_metrics", {})
            model_params = results.get("model_params", {})
            training_history = results.get("training_history", {})
            
            model_names.append(display_name)
            rmse_values.append(test_metrics.get("RMSE", 0))
            param_counts.append(model_params.get("total_params", 0))
            training_times.append(training_history.get("total_training_time", 0))
            inference_times.append(test_metrics.get("Inference_Time", 0))
            
            # 颜色分配
            if results.get("is_mmoe_stage", False):
                colors.append(mmoe_stage_colors[mmoe_stage_count % len(mmoe_stage_colors)])
                mmoe_stage_count += 1
            elif "baseline" in model_type.lower():
                colors.append("#FF6B6B")  # Red
            elif "mmoe" in model_type.lower():
                colors.append("#9C27B0")  # Purple
            else:
                colors.append("#2196F3")  # Blue

        # 创建简短但唯一的标签
        def create_short_labels(names):
            """创建简短但唯一的模型标签"""
            short_labels = []
            for name in names:
                if "Baseline" in name:
                    short_labels.append("Baseline CF")
                elif "User Time-Aware" in name:
                    short_labels.append("User Time")
                elif "Independent Time" in name:
                    short_labels.append("Independent Time")
                elif "User-Movie Time" in name:
                    short_labels.append("User-Movie Time")
                elif "Two-Stage MMoE" in name:
                    short_labels.append("Two-Stage MMoE")
                elif "MMoE Stage 1" in name:
                    short_labels.append("MMoE S1")
                elif "MMoE Stage 2" in name:
                    short_labels.append("MMoE S2")
                elif "MMoE Stage 3" in name:
                    short_labels.append("MMoE S3")
                else:
                    # 备用：使用前两个单词
                    words = name.split()
                    if len(words) >= 2:
                        short_labels.append(f"{words[0]} {words[1]}")
                    else:
                        short_labels.append(name)
            return short_labels
        
        short_model_names = create_short_labels(model_names)

        # 1. 训练效率：训练时间 vs 性能提升
        ax = axes[0, 0]
        
        # 计算相对于基线的性能提升
        baseline_rmse = rmse_values[0] if len(rmse_values) > 0 else 1.0  # 假设第一个是基线
        performance_improvement = [(baseline_rmse - rmse) / baseline_rmse * 100 for rmse in rmse_values]
        
        scatter = ax.scatter(training_times, performance_improvement, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # 添加模型标签（使用简短名称）
        for i, short_name in enumerate(short_model_names):
            ax.annotate(short_name, (training_times[i], performance_improvement[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        ax.set_xlabel("Training Time (seconds)")
        ax.set_ylabel("Performance Improvement vs Baseline (%)")
        ax.set_title("Training Efficiency\n(Higher improvement with less time is better)")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline Performance')

        # 2. 参数效率：参数数量 vs 性能
        ax = axes[0, 1]
        
        # 计算参数效率（RMSE/参数数量，越小越好）
        param_efficiency = [rmse / (params / 1000) if params > 0 else 0 for rmse, params in zip(rmse_values, param_counts)]
        
        bars = ax.bar(range(len(model_names)), param_efficiency, color=colors, alpha=0.8)
        ax.set_xlabel("Models")
        ax.set_ylabel("RMSE per 1K Parameters (Lower is Better)")
        ax.set_title("Parameter Efficiency\n(RMSE normalized by parameter count)")
        ax.set_xticks(range(len(model_names)))
        # 🔧 修复：使用简短名称
        ax.set_xticklabels(short_model_names, rotation=45, ha='right', fontsize=9)
        
        # 添加数值标签
        for bar, value in zip(bars, param_efficiency):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.6f}', ha='center', va='bottom', fontsize=8)

        # 3. 推理效率：推理时间 vs 性能
        ax = axes[1, 0]
        
        scatter = ax.scatter(inference_times, rmse_values, c=colors, s=100, alpha=0.7, edgecolors='black')
        
        # 添加模型标签（使用简短名称）
        for i, short_name in enumerate(short_model_names):
            ax.annotate(short_name, (inference_times[i], rmse_values[i]), 
                    xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, ha='left', va='bottom',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        ax.set_xlabel("Inference Time (seconds)")
        ax.set_ylabel("RMSE")
        ax.set_title("Inference Efficiency\n(Lower RMSE with faster inference is better)")
        ax.grid(True, alpha=0.3)

        # 4. 综合效率评分
        ax = axes[1, 1]
        
        # 标准化各项指标（0-1范围）
        def normalize(values, reverse=False):
            if not values or max(values) == min(values):
                return [0.5] * len(values)
            normalized = [(v - min(values)) / (max(values) - min(values)) for v in values]
            return [(1 - n) for n in normalized] if reverse else normalized
        
        # 标准化指标（越小越好的指标需要reverse=True）
        norm_rmse = normalize(rmse_values, reverse=True)  # RMSE越小越好
        norm_training_time = normalize(training_times, reverse=True)  # 训练时间越少越好
        norm_inference_time = normalize(inference_times, reverse=True)  # 推理时间越少越好
        norm_param_efficiency = normalize(param_efficiency, reverse=True)  # 参数效率越小越好
        
        # 计算综合效率评分（平均值）
        efficiency_scores = []
        for i in range(len(model_names)):
            score = (norm_rmse[i] + norm_training_time[i] + norm_inference_time[i] + norm_param_efficiency[i]) / 4
            efficiency_scores.append(score)
        
        bars = ax.bar(range(len(model_names)), efficiency_scores, color=colors, alpha=0.8)
        ax.set_xlabel("Models")
        ax.set_ylabel("Efficiency Score (Higher is Better)")
        ax.set_title("Overall Efficiency Score\n(Weighted average of all efficiency metrics)")
        ax.set_xticks(range(len(model_names)))
        # 🔧 修复：使用简短名称
        ax.set_xticklabels(short_model_names, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, efficiency_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 添加颜色图例
        legend_elements = []
        legend_labels = []
        
        # 添加基线、时间感知、MMoE的图例
        if any("baseline" in model_type.lower() for model_type, _, _ in organized_models):
            legend_elements.append(plt.scatter([], [], c="#FF6B6B", s=100, alpha=0.7))
            legend_labels.append("Baseline")
        
        if any(not results.get("is_mmoe_stage", False) and "mmoe" not in model_type.lower() and "baseline" not in model_type.lower() 
            for model_type, results, _ in organized_models):
            legend_elements.append(plt.scatter([], [], c="#2196F3", s=100, alpha=0.7))
            legend_labels.append("Time-Aware")
        
        if any(results.get("is_mmoe_stage", False) for _, results, _ in organized_models):
            legend_elements.append(plt.scatter([], [], c="#4ECDC4", s=100, alpha=0.7))
            legend_labels.append("MMoE Stages")
        
        if any("mmoe" in model_type.lower() and not results.get("is_mmoe_stage", False) 
            for model_type, results, _ in organized_models):
            legend_elements.append(plt.scatter([], [], c="#9C27B0", s=100, alpha=0.7))
            legend_labels.append("MMoE Original")
        
        if legend_elements:
            ax.legend(legend_elements, legend_labels, loc='upper right', fontsize=9)

        plt.tight_layout()
        
        save_path = self.output_dir / "15_efficiency_analysis.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Model efficiency analysis saved to: {save_path}")

    def run_complete_analysis(self):
        """Run complete analysis workflow - Generate all individual charts"""
        print("🚀 Starting comprehensive model comparison analysis...")
        print(f"📁 Output directory: {self.output_dir}")
        
        if not self.results:
            print("❌ No model results found. Please run training scripts first.")
            return None
        
        # 🔧 显示MMoE阶段处理信息
        if self.split_mmoe_stages:
            print("🔄 MMoE stages will be split into independent models")
        else:
            print("📊 MMoE will be treated as a single 90-epoch model")
        
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