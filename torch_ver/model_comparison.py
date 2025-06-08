import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm
from data_process import data_path

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class ModelComparison:
    def __init__(self, results_path=None):
        """初始化模型比较类"""
        if results_path is None:
            results_path = data_path + 'results/all_models_summary.json'
        
        self.results_path = results_path
        self.results = self.load_results()
        self.output_dir = Path(data_path) / 'analysis_plots'
        self.output_dir.mkdir(exist_ok=True)
    
    def load_results(self):
        """加载所有模型的结果"""
        try:
            with open(self.results_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"结果文件不存在: {self.results_path}")
            return {}
    
    def plot_training_curves(self):
        """绘制训练曲线对比"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('模型训练过程对比', fontsize=16, fontweight='bold')
        
        # 定义颜色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            
            # 检查是否有必要的训练历史数据
            if 'train_losses' not in history or not history['train_losses']:
                print(f"警告：模型 {results['model_name']} 缺少训练历史数据")
                continue
                
            epochs = range(1, len(history['train_losses']) + 1)
            color = colors[idx % len(colors)]
            
            # 训练损失
            axes[0, 0].plot(epochs, history['train_losses'], 
                           label=f"{results['model_name']} (训练)", 
                           color=color, linestyle='-', linewidth=2)
            
            # 验证损失（如果存在）
            if 'val_losses' in history and history['val_losses']:
                val_epochs = range(1, len(history['val_losses']) + 1)
                axes[0, 0].plot(val_epochs, history['val_losses'], 
                               label=f"{results['model_name']} (验证)", 
                               color=color, linestyle='--', linewidth=2)
        
        axes[0, 0].set_title('损失函数变化')
        axes[0, 0].set_xlabel('训练轮数')
        axes[0, 0].set_ylabel('MSE损失')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE对比
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            color = colors[idx % len(colors)]
            
            # 如果有RMSE历史，使用它；否则从损失计算
            if 'train_rmse' in history and history['train_rmse']:
                train_rmse = history['train_rmse']
                epochs = range(1, len(train_rmse) + 1)
            elif 'train_losses' in history and history['train_losses']:
                train_rmse = [np.sqrt(loss) for loss in history['train_losses']]
                epochs = range(1, len(train_rmse) + 1)
            else:
                continue
                
            axes[0, 1].plot(epochs, train_rmse, 
                           label=f"{results['model_name']} (训练)", 
                           color=color, linestyle='-', linewidth=2)
            
            # 验证RMSE
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
                               label=f"{results['model_name']} (验证)", 
                               color=color, linestyle='--', linewidth=2)
        
        axes[0, 1].set_title('RMSE变化')
        axes[0, 1].set_xlabel('训练轮数')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率变化
        for idx, (model_type, results) in enumerate(self.results.items()):
            history = results['training_history']
            color = colors[idx % len(colors)]
            
            if 'learning_rates' in history and history['learning_rates']:
                epochs = range(1, len(history['learning_rates']) + 1)
                axes[1, 0].plot(epochs, history['learning_rates'], 
                               label=results['model_name'], 
                               color=color, linewidth=2)
        
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_xlabel('训练轮数')
        axes[1, 0].set_ylabel('学习率')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 每轮训练时间
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
                                  label=f"{results['model_name']} 平均")
        
        axes[1, 1].set_title('每轮训练时间')
        axes[1, 1].set_xlabel('训练轮数')
        axes[1, 1].set_ylabel('时间 (秒)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_mmoe_stage_analysis(self):
        """为MMOE模型绘制特殊的阶段分析图"""
        # 查找MMOE模型
        mmoe_results = None
        for model_type, results in self.results.items():
            if 'MMoE' in model_type or 'mmoe' in model_type.lower():
                mmoe_results = results
                break
        
        if mmoe_results is None:
            print("未找到MMOE模型结果")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MMOE模型三阶段训练分析', fontsize=16, fontweight='bold')
        
        history = mmoe_results['training_history']
        
        # 假设我们可以根据训练历史的长度来分割阶段
        total_epochs = len(history['train_losses'])
        stage1_end = total_epochs // 3
        stage2_end = 2 * total_epochs // 3
        
        stages = {
            '时序建模': (0, stage1_end),
            'CF建模': (stage1_end, stage2_end),
            'MMoE融合': (stage2_end, total_epochs)
        }
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # 训练损失分阶段
        ax = axes[0, 0]
        for idx, (stage_name, (start, end)) in enumerate(stages.items()):
            epochs = range(start + 1, end + 1)
            if end > start and end <= len(history['train_losses']):
                ax.plot(epochs, history['train_losses'][start:end], 
                       label=stage_name, color=colors[idx], linewidth=2)
        
        ax.set_title('各阶段训练损失')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('MSE损失')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 验证损失分阶段
        ax = axes[0, 1]
        if 'val_losses' in history and history['val_losses']:
            for idx, (stage_name, (start, end)) in enumerate(stages.items()):
                epochs = range(start + 1, end + 1)
                if end > start and end <= len(history['val_losses']):
                    ax.plot(epochs, history['val_losses'][start:end], 
                           label=stage_name, color=colors[idx], linewidth=2)
        
        ax.set_title('各阶段验证损失')
        ax.set_xlabel('训练轮数')
        ax.set_ylabel('MSE损失')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 各阶段性能对比
        ax = axes[1, 0]
        stage_performance = []
        stage_names = []
        
        for stage_name, (start, end) in stages.items():
            if end > start and end <= len(history['train_losses']):
                avg_loss = np.mean(history['train_losses'][start:end])
                stage_performance.append(avg_loss)
                stage_names.append(stage_name)
        
        bars = ax.bar(stage_names, stage_performance, color=colors[:len(stage_names)])
        ax.set_title('各阶段平均训练损失')
        ax.set_ylabel('MSE损失')
        
        for bar, perf in zip(bars, stage_performance):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{perf:.4f}', ha='center', va='bottom')
        
        # 最终测试性能
        ax = axes[1, 1]
        test_metrics = mmoe_results['test_metrics']
        metrics = ['RMSE', 'MAE', 'MAPE', 'Correlation']
        values = [test_metrics[metric] for metric in metrics if metric in test_metrics]
        metrics = [metric for metric in metrics if metric in test_metrics]
        
        bars = ax.bar(metrics, values, color='#d62728')
        ax.set_title('MMOE最终测试性能')
        ax.set_ylabel('指标值')
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mmoe_stage_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self):
        """绘制性能指标对比"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('模型性能指标对比', fontsize=16, fontweight='bold')
        
        # 准备数据
        model_names = [results['model_name'] for results in self.results.values()]
        
        # 主要性能指标
        metrics = ['RMSE', 'MAE', 'MAPE', 'Correlation']
        metric_values = {metric: [] for metric in metrics}
        
        for results in self.results.values():
            test_metrics = results['test_metrics']
            for metric in metrics:
                metric_values[metric].append(test_metrics[metric])
        
        # 绘制主要指标
        positions = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics[:4]):
            if i < 2:  # RMSE, MAE
                ax = axes[0, i]
                bars = ax.bar(positions, metric_values[metric], 
                             color=sns.color_palette("husl", len(model_names)))
                ax.set_title(f'{metric} 对比')
                ax.set_ylabel(metric)
                ax.set_xticks(positions)
                ax.set_xticklabels(model_names, rotation=45, ha='right')
                
                # 添加数值标签
                for bar, value in zip(bars, metric_values[metric]):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        # 相关系数
        ax = axes[0, 2]
        bars = ax.bar(positions, metric_values['Correlation'], 
                     color=sns.color_palette("husl", len(model_names)))
        ax.set_title('预测相关性对比')
        ax.set_ylabel('Correlation')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylim(0, 1)
        
        for bar, value in zip(bars, metric_values['Correlation']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom')
        
        # 模型复杂度对比
        ax = axes[1, 0]
        param_counts = [results['model_params']['total_params'] for results in self.results.values()]
        bars = ax.bar(positions, param_counts, 
                     color=sns.color_palette("husl", len(model_names)))
        ax.set_title('模型参数数量对比')
        ax.set_ylabel('参数数量')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        for bar, value in zip(bars, param_counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:,}', ha='center', va='bottom')
        
        # 训练时间对比
        ax = axes[1, 1]
        training_times = [results['training_history']['total_training_time'] for results in self.results.values()]
        bars = ax.bar(positions, training_times, 
                     color=sns.color_palette("husl", len(model_names)))
        ax.set_title('总训练时间对比')
        ax.set_ylabel('时间 (秒)')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        for bar, value in zip(bars, training_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.1f}s', ha='center', va='bottom')
        
        # 推理时间对比
        ax = axes[1, 2]
        inference_times = [results['test_metrics']['Inference_Time'] for results in self.results.values()]
        bars = ax.bar(positions, inference_times, 
                     color=sns.color_palette("husl", len(model_names)))
        ax.set_title('推理时间对比')
        ax.set_ylabel('时间 (秒)')
        ax.set_xticks(positions)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        
        for bar, value in zip(bars, inference_times):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{value:.2f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_analysis(self):
        """绘制预测分析图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('预测结果分析', fontsize=16, fontweight='bold')
        
        # 为每个模型绘制预测vs实际散点图
        for idx, (model_type, results) in enumerate(self.results.items()):
            if idx >= 3:  # 最多显示3个模型
                break
                
            test_metrics = results['test_metrics']
            predictions = np.array(test_metrics['predictions'])
            actuals = np.array(test_metrics['actuals'])
            
            # 预测vs实际散点图
            ax = axes[0, idx]
            ax.scatter(actuals, predictions, alpha=0.5, s=1)
            ax.plot([1, 5], [1, 5], 'r--', linewidth=2)  # 完美预测线
            ax.set_xlabel('实际评分')
            ax.set_ylabel('预测评分')
            ax.set_title(f'{results["model_name"]}\n预测vs实际')
            ax.set_xlim(0.5, 5.5)
            ax.set_ylim(0.5, 5.5)
            
            # 添加统计信息
            correlation = test_metrics['Correlation']
            rmse = test_metrics['RMSE']
            ax.text(0.05, 0.95, f'相关性: {correlation:.3f}\nRMSE: {rmse:.3f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 残差分布
            ax = axes[1, idx]
            residuals = predictions - actuals
            ax.hist(residuals, bins=50, alpha=0.7, density=True)
            ax.set_xlabel('残差 (预测 - 实际)')
            ax.set_ylabel('密度')
            ax.set_title(f'{results["model_name"]}\n残差分布')
            ax.axvline(x=0, color='red', linestyle='--')
            
            # 添加残差统计
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            ax.text(0.05, 0.95, f'均值: {mean_residual:.3f}\n标准差: {std_residual:.3f}', 
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'prediction_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_rating_distribution_analysis(self):
        """绘制评分分布分析"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('评分分布分析', fontsize=16, fontweight='bold')
        
        # 收集所有模型的预测数据
        all_predictions = {}
        actual_ratings = None
        
        for model_type, results in self.results.items():
            test_metrics = results['test_metrics']
            predictions = np.array(test_metrics['predictions'])
            actuals = np.array(test_metrics['actuals'])
            
            all_predictions[results['model_name']] = predictions
            if actual_ratings is None:
                actual_ratings = actuals
        
        # 实际评分分布
        ax = axes[0, 0]
        ax.hist(actual_ratings, bins=np.arange(0.5, 6.5, 1), alpha=0.7, 
               label='实际评分', color='gray', edgecolor='black')
        ax.set_xlabel('评分')
        ax.set_ylabel('频次')
        ax.set_title('实际评分分布')
        ax.set_xticks(range(1, 6))
        ax.legend()
        
        # 各模型预测评分分布对比
        ax = axes[0, 1]
        for model_name, predictions in all_predictions.items():
            ax.hist(predictions, bins=50, alpha=0.5, label=model_name, density=True)
        ax.hist(actual_ratings, bins=50, alpha=0.7, label='实际评分', 
               density=True, color='black', linestyle='--', histtype='step', linewidth=2)
        ax.set_xlabel('评分')
        ax.set_ylabel('密度')
        ax.set_title('预测评分分布对比')
        ax.legend()
        
        # 各评分等级的MAE对比
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
        
        ax.set_xlabel('评分等级')
        ax.set_ylabel('MAE')
        ax.set_title('各评分等级的预测误差')
        ax.set_xticks(x + width)
        ax.set_xticklabels(rating_levels)
        ax.legend()
        
        # 预测范围对比
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
            ax.scatter(i, min_val, color='red', marker='v', s=50, label='最小值' if i == 0 else '')
            ax.scatter(i, max_val, color='red', marker='^', s=50, label='最大值' if i == 0 else '')
        
        ax.set_xlabel('模型')
        ax.set_ylabel('预测评分')
        ax.set_title('预测评分范围对比 (均值±标准差)')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'rating_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_table(self):
        """生成模型性能汇总表"""
        summary_data = []
        
        for model_type, results in self.results.items():
            test_metrics = results['test_metrics']
            training_history = results['training_history']
            model_params = results['model_params']
            
            summary_data.append({
                '模型名称': results['model_name'],
                '模型类型': model_type,
                'RMSE': f"{test_metrics['RMSE']:.4f}",
                'MAE': f"{test_metrics['MAE']:.4f}",
                'MAPE (%)': f"{test_metrics['MAPE']:.2f}",
                '相关系数': f"{test_metrics['Correlation']:.4f}",
                '参数数量': f"{model_params['total_params']:,}",
                '训练轮数': training_history['total_epochs'],
                '训练时间 (s)': f"{training_history['total_training_time']:.1f}",
                '推理时间 (s)': f"{test_metrics['Inference_Time']:.2f}",
                '最佳轮数': training_history['best_epoch'] + 1
            })
        
        df = pd.DataFrame(summary_data)
        
        # 保存为CSV
        csv_path = self.output_dir / 'model_comparison_summary.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        # 打印表格
        print("\n" + "="*100)
        print("模型性能汇总表")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        return df
    
    def run_complete_analysis(self):
        """运行完整的分析流程"""
        print("开始模型比较分析...")
        
        if not self.results:
            print("没有找到模型结果数据，请先运行训练脚本。")
            return
        
        print(f"找到 {len(self.results)} 个模型的结果")
        
        # 生成所有图表
        print("1. 绘制训练曲线对比...")
        self.plot_training_curves()
        
        print("2. 绘制性能指标对比...")
        self.plot_performance_comparison()
        
        print("3. 绘制预测结果分析...")
        self.plot_prediction_analysis()
        
        print("4. 绘制评分分布分析...")
        self.plot_rating_distribution_analysis()
        
        print("5. 绘制MMOE阶段分析...")
        self.plot_mmoe_stage_analysis()
        
        print("6. 生成汇总表...")
        summary_df = self.generate_summary_table()
        
        print(f"\n所有分析图表已保存至: {self.output_dir}")
        print("分析完成！")
        
        return summary_df

def main():
    """主函数"""
    # 创建模型比较分析器
    analyzer = ModelComparison()
    
    # 运行完整分析
    summary_df = analyzer.run_complete_analysis()
    
    return analyzer, summary_df

if __name__ == "__main__":
    analyzer, summary = main()