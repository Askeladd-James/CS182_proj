import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
from data_process import data_path

# 设置英文字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ModelComparison:
    def __init__(self, results_path=None):
        """Initialize model comparison class"""
        if results_path is None:
            results_path = data_path + 'results/all_models_summary.json'
        
        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = Path(data_path) / 'analysis_plots'
        self.output_dir.mkdir(exist_ok=True)
    
    def load_results(self):
        """Load all model results - 支持baseline和改进模型"""
        results = {}
        
        # 尝试加载主要汇总文件（包括baseline和改进版本）
        summary_files = [
            self.results_path,
            data_path + 'results/all_models_summary_with_baseline.json',
            data_path + 'results/all_models_summary_with_scheduler.json',
            data_path + 'results/all_models_summary.json'
        ]
        
        for file_path in summary_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                    print(f"Successfully loaded results file: {file_path}")
                    break
            except FileNotFoundError:
                continue
        
        if not results:
            print(f"Warning: No summary file found, trying to load individual result files...")
            # 尝试加载单独的结果文件（包括baseline）
            results_dir = Path(data_path) / 'results'
            if results_dir.exists():
                for result_file in results_dir.glob('results_*.json'):
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            single_result = json.load(f)
                            model_type = single_result.get('model_type', result_file.stem)
                            results[model_type] = single_result
                            print(f"Loaded individual result file: {result_file}")
                    except Exception as e:
                        print(f"Failed to load file {result_file}: {e}")
        
        return results
    
    def plot_training_curves(self):
        """Plot training curves comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Training Process Comparison', fontsize=16, fontweight='bold')
        
        # Define colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            
            # Check if necessary training history data exists
            if 'train_losses' not in history or not history['train_losses']:
                print(f"Warning: Model {results['model_name']} lacks training history data")
                continue
                
            epochs = range(1, len(history['train_losses']) + 1)
            color = colors[idx % len(colors)]
            
            # Training loss
            axes[0, 0].plot(epochs, history['train_losses'], 
                           label=f"{results['model_name']} (Train)", 
                           color=color, linestyle='-', linewidth=2)
            
            # Validation loss (if exists)
            if 'val_losses' in history and history['val_losses']:
                val_epochs = range(1, len(history['val_losses']) + 1)
                axes[0, 0].plot(val_epochs, history['val_losses'], 
                               label=f"{results['model_name']} (Val)", 
                               color=color, linestyle='--', linewidth=2)
        
        axes[0, 0].set_title('Loss Function Evolution')
        axes[0, 0].set_xlabel('Training Epochs')
        axes[0, 0].set_ylabel('MSE Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            color = colors[idx % len(colors)]
            
            # Use RMSE history if available; otherwise calculate from losses
            if 'train_rmse' in history and history['train_rmse']:
                train_rmse = history['train_rmse']
                epochs = range(1, len(train_rmse) + 1)
            elif 'train_losses' in history and history['train_losses']:
                train_rmse = [np.sqrt(loss) for loss in history['train_losses']]
                epochs = range(1, len(train_rmse) + 1)
            else:
                continue
                
            axes[0, 1].plot(epochs, train_rmse, 
                           label=f"{results['model_name']} (Train)", 
                           color=color, linestyle='-', linewidth=2)
            
            # Validation RMSE
            if 'val_rmse' in history and history['val_rmse']:
                val_rmse = history['val_rmse']
                val_epochs = range(1, len(val_rmse) + 1)
            elif 'val_losses' in history and history['val_losses']:
                val_rmse = [np.sqrt(loss) for loss in history['val_losses']]
                val_epochs = range(1, len(val_rmse) + 1)
            else:
                val_rmse = None
                
            if val_rmse:
                axes[0, 1].plot(val_epochs, val_rmse, 
                               label=f"{results['model_name']} (Val)", 
                               color=color, linestyle='--', linewidth=2)
        
        axes[0, 1].set_title('RMSE Evolution')
        axes[0, 1].set_xlabel('Training Epochs')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Learning rate evolution
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            color = colors[idx % len(colors)]
            
            if 'learning_rates' in history and history['learning_rates']:
                epochs = range(1, len(history['learning_rates']) + 1)
                axes[1, 0].plot(epochs, history['learning_rates'], 
                               label=results['model_name'], 
                               color=color, linewidth=2)
        
        axes[1, 0].set_title('Learning Rate Evolution')
        axes[1, 0].set_xlabel('Training Epochs')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Training time per epoch
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            color = colors[idx % len(colors)]
            
            if 'epoch_times' in history and history['epoch_times']:
                epochs = range(1, len(history['epoch_times']) + 1)
                axes[1, 1].plot(epochs, history['epoch_times'], 
                               label=results['model_name'], 
                               color=color, linewidth=2, alpha=0.7)
                axes[1, 1].axhline(y=np.mean(history['epoch_times']), 
                                  color=color, linestyle=':', 
                                  label=f"{results['model_name']} Avg")
        
        axes[1, 1].set_title('Training Time per Epoch')
        axes[1, 1].set_xlabel('Training Epochs')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_mmoe_stage_analysis(self):
        """Plot special stage analysis chart for MMOE model"""
        # Find MMOE model
        mmoe_results = None
        for model_type, results in self.results.items():
            if 'MMoE' in model_type or 'mmoe' in model_type.lower():
                mmoe_results = results
                break
        
        if mmoe_results is None:
            print("MMOE model results not found")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MMOE Model Three-Stage Training Analysis (with LR Scheduling)', fontsize=16, fontweight='bold')
        
        history = mmoe_results['training_history']
        
        # Assume we can split stages based on training history length
        total_epochs = len(history['train_losses'])
        stage1_end = total_epochs // 3
        stage2_end = 2 * total_epochs // 3
        
        stages = {
            'Temporal Modeling': (0, stage1_end),
            'CF Modeling': (stage1_end, stage2_end),
            'MMoE Fusion': (stage2_end, total_epochs)
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Training loss by stage
        ax = axes[0, 0]
        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            epochs = range(start + 1, end + 1)
            if end > start and end <= len(history['train_losses']):
                ax.plot(epochs, history['train_losses'][start:end], 
                    label=stage_name, color=colors[idx], linewidth=2)
        
        ax.set_title('Training Loss by Stage')
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Validation loss by stage
        ax = axes[0, 1]
        if 'val_losses' in history and history['val_losses']:
            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                epochs = range(start + 1, end + 1)
                if end > start and end <= len(history['val_losses']):
                    ax.plot(epochs, history['val_losses'][start:end], 
                        label=stage_name, color=colors[idx], linewidth=2)
        
        ax.set_title('Validation Loss by Stage')
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('MSE Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate evolution by stage
        ax = axes[0, 2]
        if 'learning_rates' in history and history['learning_rates']:
            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                epochs = range(start + 1, end + 1)
                if end > start and end <= len(history['learning_rates']):
                    ax.plot(epochs, history['learning_rates'][start:end], 
                        label=stage_name, color=colors[idx], linewidth=2)
        
        ax.set_title('Learning Rate Evolution by Stage')
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Performance comparison by stage
        ax = axes[1, 0]
        stage_performance = []
        stage_names = []
        
        for stage_name, (start, end) in stages.items():
            if end > start and end <= len(history['train_losses']):
                avg_loss = np.mean(history['train_losses'][start:end])
                stage_performance.append(avg_loss)
                stage_names.append(stage_name)
        
        bars = ax.bar(stage_names, stage_performance, color=colors[:len(stage_names)])
        ax.set_title('Average Training Loss by Stage')
        ax.set_ylabel('MSE Loss')
        
        for bar, perf in zip(bars, stage_performance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{perf:.4f}', ha='center', va='bottom')
        
        # Learning rate decay effect
        ax = axes[1, 1]
        if 'learning_rates' in history and history['learning_rates']:
            stage_lr_changes = []
            stage_names_lr = []
            
            for stage_name, (start, end) in stages.items():
                if end > start and end <= len(history['learning_rates']):
                    initial_lr = history['learning_rates'][start]
                    final_lr = history['learning_rates'][end-1]
                    lr_reduction = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
                    stage_lr_changes.append(lr_reduction)
                    stage_names_lr.append(stage_name)
            
            bars = ax.bar(stage_names_lr, stage_lr_changes, color=colors[:len(stage_names_lr)])
            ax.set_title('Learning Rate Decay by Stage')
            ax.set_ylabel('Decay Percentage (%)')
            
            for bar, change in zip(bars, stage_lr_changes):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{change:.1f}%', ha='center', va='bottom')
        
        # Final test performance
        ax = axes[1, 2]
        test_metrics = mmoe_results['test_metrics']
        metrics = ['RMSE', 'MAE', 'MAPE', 'Correlation']
        values = [test_metrics[metric] for metric in metrics if metric in test_metrics]
        metrics = [metric for metric in metrics if metric in test_metrics]
        
        bars = ax.bar(metrics, values, color='#d62728')
        ax.set_title('MMOE Final Test Performance')
        ax.set_ylabel('Metric Value')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mmoe_stage_analysis_with_scheduler.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self):
        """Plot performance metrics comparison - 包括baseline"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Metrics Comparison (Including Baseline)', fontsize=16, fontweight='bold')
        
        # Prepare data
        model_names = [results['model_name'] for results in self.results.values()]
        
        # 确保baseline排在第一位（如果存在）
        sorted_items = []
        baseline_item = None
        
        for model_type, results in self.results.items():
            if 'baseline' in model_type.lower() or 'cfmodel' in model_type.lower():
                baseline_item = (model_type, results)
            else:
                sorted_items.append((model_type, results))
        
        if baseline_item:
            sorted_items.insert(0, baseline_item)
        
        # 重新组织数据
        model_names = [results['model_name'] for _, results in sorted_items]
        
        # Main performance metrics
        metrics = ['RMSE', 'MAE', 'MAPE', 'Correlation']
        metric_values = {metric: [] for metric in metrics}
        
        for _, results in sorted_items:
            test_metrics = results['test_metrics']
            for metric in metrics:
                metric_values[metric].append(test_metrics[metric])
        
        # Plot main metrics
        positions = np.arange(len(model_names))
        
        # 为baseline使用特殊颜色
        colors = sns.color_palette("husl", len(model_names))
        if baseline_item:
            colors[0] = '#FF6B6B'  # 红色突出baseline
        
        for i, metric in enumerate(metrics[:4]):
            if i < 2:  # RMSE, MAE
                ax = axes[0, i]
                bars = ax.bar(positions, metric_values[metric], color=colors)
                ax.set_title(f'{metric} Comparison (Lower is Better)')
                ax.set_ylabel(metric)
                ax.set_xticks(positions)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                
                # Add value labels
                for bar, value in zip(bars, metric_values[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # Correlation coefficient
        ax = axes[0, 2]
        bars = ax.bar(positions, metric_values['Correlation'], color=colors)
        ax.set_title('Prediction Correlation Comparison (Higher is Better)')
        ax.set_ylabel('Correlation')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, metric_values['Correlation']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom')
        
        # Model complexity comparison
        ax = axes[1, 0]
        param_counts = [results['model_params']['total_params'] for _, results in sorted_items]
        bars = ax.bar(positions, param_counts, color=colors)
        ax.set_title('Model Parameter Count Comparison')
        ax.set_ylabel('Parameter Count')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,}', ha='center', va='bottom')
        
        # Training time comparison
        ax = axes[1, 1]
        training_times = [results['training_history']['total_training_time'] for _, results in sorted_items]
        bars = ax.bar(positions, training_times, color=colors)
        ax.set_title('Total Training Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}s', ha='center', va='bottom')
        
        # Inference time comparison
        ax = axes[1, 2]
        inference_times = [results['test_metrics']['Inference_Time'] for _, results in sorted_items]
        bars = ax.bar(positions, inference_times, color=colors)
        ax.set_title('Inference Time Comparison')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison_with_baseline.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_analysis(self):
        """Plot prediction analysis charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Prediction Results Analysis', fontsize=16, fontweight='bold')
        
        # Plot prediction vs actual scatter plots for each model
        for idx, (model_type, results) in enumerate(self.results.items()):
            if idx >= 3:  # Show at most 3 models
                break
                
            test_metrics = results['test_metrics']
            predictions = np.array(test_metrics['predictions'])
            actuals = np.array(test_metrics['actuals'])
            
            # Prediction vs actual scatter plot
            ax = axes[0, idx]
            ax.scatter(actuals, predictions, alpha=0.5, s=1)
            ax.plot([1, 5], [1, 5], 'r--', linewidth=2)  # Perfect prediction line
            ax.set_xlabel('Actual Rating')
            ax.set_ylabel('Predicted Rating')
            ax.set_title(f'{results["model_name"]}\nPredicted vs Actual')
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
            
            # Add statistics
            correlation = test_metrics['Correlation']
            rmse = test_metrics['RMSE']
            ax.text(0.05, 0.95, f'Correlation: {correlation:.3f}\nRMSE: {rmse:.3f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Residual distribution
            ax = axes[1, idx]
            residuals = predictions - actuals
            ax.hist(residuals, bins=50, alpha=0.7, density=True)
            ax.set_xlabel('Residuals (Predicted - Actual)')
            ax.set_ylabel('Density')
            ax.set_title(f'{results["model_name"]}\nResidual Distribution')
            ax.axvline(x=0, color='red', linestyle='--')
            
            # Add residual statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax.text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rating_distribution_analysis(self):
        """Plot rating distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rating Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Collect all model prediction data
        all_predictions = {}
        actual_ratings = None
        
        for model_type, results in self.results.items():
            test_metrics = results['test_metrics']
            predictions = np.array(test_metrics['predictions'])
            actuals = np.array(test_metrics['actuals'])
            
            all_predictions[results['model_name']] = predictions
            if actual_ratings is None:
                actual_ratings = actuals
        
        # Actual rating distribution
        ax = axes[0, 0]
        ax.hist(actual_ratings, bins=np.arange(0.5, 6.5, 1), alpha=0.7, 
               label='Actual Ratings', color='gray', edgecolor='black')
        ax.set_xlabel('Rating')
        ax.set_ylabel('Frequency')
        ax.set_title('Actual Rating Distribution')
        ax.set_xticks(range(1, 6))
        ax.legend()
        
        # Predicted rating distribution comparison
        ax = axes[0, 1]
        for model_name, predictions in all_predictions.items():
            ax.hist(predictions, bins=50, alpha=0.5, label=model_name, density=True)
        ax.hist(actual_ratings, bins=50, alpha=0.7, label='Actual Ratings', 
               density=True, color='black', linestyle='--', histtype='step', linewidth=2)
        ax.set_xlabel('Rating')
        ax.set_ylabel('Density')
        ax.set_title('Predicted Rating Distribution Comparison')
        ax.legend()
        
        # MAE comparison by rating level
        ax = axes[1, 0]
        rating_levels = [1, 2, 3, 4, 5]
        width = 0.25
        x = np.arange(len(rating_levels))
        
        for idx, (model_type, results) in enumerate(self.results.items()):
            test_metrics = results['test_metrics']
            rating_maes = []
            for rating in rating_levels:
                mae_key = f'rating_{rating}_mae'
                rating_maes.append(test_metrics.get(mae_key, 0))
            
            ax.bar(x + idx * width, rating_maes, width, 
                  label=results['model_name'], alpha=0.8)
        
        ax.set_xlabel('Rating Level')
        ax.set_ylabel('MAE')
        ax.set_title('Prediction Error by Rating Level')
        ax.set_xticks(x + width)
        ax.set_xticklabels(rating_levels)
        ax.legend()
        
        # Prediction range comparison
        ax = axes[1, 1]
        model_names = []
        pred_mins = []
        pred_maxs = []
        pred_means = []
        pred_stds = []
        
        for model_type, results in self.results.items():
            test_metrics = results['test_metrics']
            predictions = np.array(test_metrics['predictions'])
            
            model_names.append(results['model_name'])
            pred_mins.append(np.min(predictions))
            pred_maxs.append(np.max(predictions))
            pred_means.append(np.mean(predictions))
            pred_stds.append(np.std(predictions))
        
        x_pos = np.arange(len(model_names))
        ax.errorbar(x_pos, pred_means, yerr=pred_stds, fmt='o', capsize=5, capthick=2)
        
        for i, (mean, std, min_val, max_val) in enumerate(zip(pred_means, pred_stds, pred_mins, pred_maxs)):
            ax.scatter(i, min_val, color='red', marker='v', s=50, label='Min' if i == 0 else '')
            ax.scatter(i, max_val, color='red', marker='^', s=50, label='Max' if i == 0 else '')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Predicted Rating')
        ax.set_title('Prediction Range Comparison (Mean±Std)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rating_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_learning_rate_analysis(self):
        """Analyze learning rate scheduler effects"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Learning Rate Scheduler Effect Analysis', fontsize=16, fontweight='bold')
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        # Learning rate evolution curves
        ax = axes[0, 0]
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            color = colors[idx % len(colors)]
            
            if 'learning_rates' in history and history['learning_rates']:
                epochs = range(1, len(history['learning_rates']) + 1)
                ax.plot(epochs, history['learning_rates'], 
                    label=results['model_name'], 
                    color=color, linewidth=2)
        
        ax.set_title('Learning Rate Evolution Curves')
        ax.set_xlabel('Training Epochs')
        ax.set_ylabel('Learning Rate')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate decay statistics
        ax = axes[0, 1]
        model_names = []
        lr_reductions = []
        final_lrs = []
        
        for model_type, results in self.results.items():
            history = results['training_history']
            if 'learning_rates' in history and history['learning_rates']:
                initial_lr = history['learning_rates'][0]
                final_lr = history['learning_rates'][-1]
                reduction = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
                
                model_names.append(results['model_name'])
                lr_reductions.append(reduction)
                final_lrs.append(final_lr)
        
        if model_names:
            bars = ax.bar(model_names, lr_reductions, color=colors[:len(model_names)])
            ax.set_title('Total Learning Rate Decay')
            ax.set_ylabel('Decay Percentage (%)')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, reduction in zip(bars, lr_reductions):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{reduction:.1f}%', ha='center', va='bottom')
        
        # Final learning rate comparison
        ax = axes[1, 0]
        if model_names:
            bars = ax.bar(model_names, final_lrs, color=colors[:len(model_names)])
            ax.set_title('Final Learning Rate Comparison')
            ax.set_ylabel('Final Learning Rate')
            ax.set_yscale('log')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, final_lr in zip(bars, final_lrs):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{final_lr:.1e}', ha='center', va='bottom', rotation=0)
        
        # Learning rate vs performance relationship
        ax = axes[1, 1]
        if model_names:
            rmse_values = [self.results[list(self.results.keys())[i]]['test_metrics']['RMSE'] 
                        for i in range(len(model_names))]
            
            scatter = ax.scatter(final_lrs, rmse_values, 
                            c=range(len(model_names)), 
                            cmap='viridis', s=100, alpha=0.7)
            
            # Add model name labels
            for i, name in enumerate(model_names):
                ax.annotate(name, (final_lrs[i], rmse_values[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
            
            ax.set_title('Final Learning Rate vs Test RMSE')
            ax.set_xlabel('Final Learning Rate')
            ax.set_ylabel('Test RMSE')
            ax.set_xscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'learning_rate_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_table(self):
        """Generate enhanced model performance summary table with learning rate info"""
        summary_data = []
        
        for model_type, results in self.results.items():
            test_metrics = results['test_metrics']
            training_history = results['training_history']
            model_params = results['model_params']
            
            # Calculate learning rate change information
            lr_info = "N/A"
            if 'learning_rates' in training_history and training_history['learning_rates']:
                initial_lr = training_history['learning_rates'][0]
                final_lr = training_history['learning_rates'][-1]
                reduction = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
                lr_info = f"{initial_lr:.1e}→{final_lr:.1e}(-{reduction:.1f}%)"
            
            summary_data.append({
                'Model Name': results['model_name'],
                'Model Type': model_type,
                'RMSE': f"{test_metrics['RMSE']:.4f}",
                'MAE': f"{test_metrics['MAE']:.4f}",
                'MAPE (%)': f"{test_metrics['MAPE']:.2f}",
                'Correlation': f"{test_metrics['Correlation']:.4f}",
                'Parameter Count': f"{model_params['total_params']:,}",
                'Training Epochs': training_history['total_epochs'],
                'Training Time (s)': f"{training_history['total_training_time']:.1f}",
                'Inference Time (s)': f"{test_metrics['Inference_Time']:.2f}",
                'Learning Rate Change': lr_info,
                'Best Epoch': training_history.get('best_epoch', 0) + 1
            })
        
        df = pd.DataFrame(summary_data)
        
        # Save as CSV
        csv_path = self.output_dir / 'enhanced_model_comparison_summary.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # Print table
        print("\n" + "="*120)
        print("Enhanced Model Performance Summary Table (with Learning Rate Scheduling Info)")
        print("="*120)
        print(df.to_string(index=False))
        print("="*120)
        
        return df
    
    def run_complete_analysis(self):
        """Run complete analysis workflow - enhanced version"""
        print("Starting enhanced model comparison analysis...")
        
        if not self.results:
            print("No model result data found. Please run training scripts first.")
            return
        
        print(f"Found results for {len(self.results)} models")
        
        # Check if learning rate information is included
        has_scheduler_info = any(
            'learning_rates' in results['training_history'] and results['training_history']['learning_rates']
            for results in self.results.values()
        )
        
        if has_scheduler_info:
            print("Learning rate scheduler information detected, generating enhanced analysis...")
        
        # Generate all charts
        print("1. Plotting training curves comparison...")
        self.plot_training_curves()
        
        print("2. Plotting performance metrics comparison...")
        self.plot_performance_comparison()
        
        print("3. Plotting prediction results analysis...")
        self.plot_prediction_analysis()
        
        print("4. Plotting rating distribution analysis...")
        self.plot_rating_distribution_analysis()
        
        if has_scheduler_info:
            print("5. Plotting learning rate scheduling analysis...")
            self.plot_learning_rate_analysis()
        
        print("6. Plotting MMOE stage analysis...")
        self.plot_mmoe_stage_analysis()
        
        print("7. Generating enhanced summary table...")
        summary_df = self.generate_summary_table()
        
        print(f"\nAll analysis charts saved to: {self.output_dir}")
        print("Enhanced analysis completed!")
        
        return summary_df

def main():
    """Main function"""
    # Create model comparison analyzer
    analyzer = ModelComparison()
    
    # Run complete analysis
    summary_df = analyzer.run_complete_analysis()
    
    return analyzer, summary_df

if __name__ == "__main__":
    analyzer, summary = main()