"""
完整的模型训练、评估和比较流程（包括baseline）
"""
import logging
import time
from pathlib import Path
from data_process import data_path

def setup_logging():
    """设置日志 - 每次运行时清空日志文件"""
    log_file = Path(data_path) / 'full_comparison_log.txt'
    
    # 每次运行时删除旧的日志文件
    if log_file.exists():
        log_file.unlink()
        print(f"🗑️  已清空旧日志文件: {log_file}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),  # 使用 'w' 模式确保覆盖
            logging.StreamHandler()
        ],
        force=True  # 强制重新配置logging
    )
    
    # 记录日志开始
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("🆕 新的完整模型比较流程开始")
    logger.info("=" * 80)
    
    return logger

def run_all_model_training():
    """运行所有模型训练（包括baseline和时间感知模型）"""
    logger = logging.getLogger(__name__)
    logger.info("开始训练所有模型...")
    
    try:
        # 运行所有模型训练（包括baseline、时间感知模型、MMOE）
        import train_comparison
        results = train_comparison.main()
        logger.info("✅ 所有模型训练完成!")
        return True, results
    except Exception as e:
        logger.error(f"❌ 模型训练过程中出错: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return False, None

def run_analysis():
    """运行模型分析"""
    logger = logging.getLogger(__name__)
    logger.info("开始模型分析阶段...")
    
    try:
        # 运行分析脚本
        import model_comparison
        analyzer, summary = model_comparison.main()
        logger.info("✅ 模型分析完成!")
        return analyzer, summary
    except Exception as e:
        logger.error(f"❌ 分析过程中出错: {str(e)}")
        import traceback
        logger.error(f"详细错误信息: {traceback.format_exc()}")
        return None, None

def check_data_availability():
    """检查数据文件是否存在"""
    logger = logging.getLogger(__name__)
    
    required_files = [
        'ratings.csv',
        'users.csv', 
        'movies.csv'
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = Path(data_path) / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        logger.error(f"缺少必要的数据文件: {missing_files}")
        logger.error(f"请确保以下文件存在于 {data_path} 目录中:")
        for file_name in missing_files:
            logger.error(f"  - {file_name}")
        return False
    
    logger.info("✅ 所有必要的数据文件都存在")
    return True

def print_progress_summary(training_success, analysis_success):
    """打印进度总结"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 执行进度总结")
    logger.info("=" * 80)
    
    steps = [
        ("所有模型训练", training_success),
        ("模型分析和比较", analysis_success)
    ]
    
    success_count = sum(1 for _, success in steps if success)
    total_steps = len(steps)
    
    for step_name, success in steps:
        status = "✅ 成功" if success else "❌ 失败"
        logger.info(f"{step_name}: {status}")
    
    logger.info(f"\n总体进度: {success_count}/{total_steps} 步骤完成")
    
    if success_count == total_steps:
        logger.info("🎉 所有步骤都成功完成！")
    elif success_count > 0:
        logger.info("⚠️  部分步骤完成，请检查失败的步骤")
    else:
        logger.info("❌ 所有步骤都失败了，请检查配置和数据")
    
    return success_count == total_steps

def main():
    """主流程 - 训练所有模型并进行分析"""
    start_time = time.time()
    logger = setup_logging()
    
    logger.info("🚀 开始完整的模型比较流程...")
    logger.info(f"📁 工作目录: {data_path}")
    
    # 检查数据可用性
    if not check_data_availability():
        logger.error("❌ 数据检查失败，终止流程")
        return None, None
    
    # 创建必要的目录
    directories = ['model', 'results', 'analysis_plots']
    for dir_name in directories:
        dir_path = Path(data_path) / dir_name
        dir_path.mkdir(exist_ok=True)
        logger.info(f"📂 确保目录存在: {dir_path}")
    
    # 阶段1：训练所有模型（Baseline + 时间感知模型 + MMOE）
    logger.info("\n" + "=" * 80)
    logger.info("🔄 阶段1：训练所有模型")
    logger.info("=" * 80)
    logger.info("包括：Baseline、时间感知模型（UserTime、IndependentTime、UMTime）、MMOE")
    
    training_start = time.time()
    training_success, training_results = run_all_model_training()
    training_time = time.time() - training_start
    
    if training_success:
        logger.info(f"✅ 所有模型训练完成，耗时: {training_time:.2f}秒")
        
        # 显示训练结果概览
        if training_results:
            logger.info("\n📊 训练结果概览:")
            for model_type, results in training_results.items():
                test_metrics = results.get('test_metrics', {})
                rmse = test_metrics.get('RMSE', 'N/A')
                mae = test_metrics.get('MAE', 'N/A')
                logger.info(f"  {results.get('model_name', model_type)}:")
                logger.info(f"    RMSE: {rmse}")
                logger.info(f"    MAE: {mae}")
    else:
        logger.error("❌ 模型训练失败")
    
    # 阶段2：模型分析和比较（只有在训练成功时才执行）
    analysis_success = False
    if training_success:
        logger.info("\n" + "=" * 80)
        logger.info("🔄 阶段2：模型分析和比较")
        logger.info("=" * 80)
        
        analysis_start = time.time()
        analyzer, summary = run_analysis()
        analysis_time = time.time() - analysis_start
        
        if analyzer is not None:
            analysis_success = True
            logger.info(f"✅ 模型分析完成，耗时: {analysis_time:.2f}秒")
        else:
            logger.error("❌ 模型分析失败")
    else:
        logger.error("❌ 训练失败，跳过分析阶段")
        analyzer, summary = None, None
    
    # 总结
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("🏁 完整流程执行完成！")
    logger.info("=" * 80)
    
    # 打印详细的总结
    all_success = print_progress_summary(training_success, analysis_success)
    
    logger.info(f"\n⏱️  总执行时间: {total_time:.2f}秒")
    
    if training_success:
        logger.info(f"   - 模型训练: {training_time:.2f}秒")
    if analysis_success:
        logger.info(f"   - 模型分析: {analysis_time:.2f}秒")
    
    # 输出文件位置信息
    if all_success:
        logger.info(f"\n📋 生成的文件:")
        logger.info(f"📊 结果文件: {data_path}results/")
        logger.info(f"🤖 模型文件: {data_path}model/")
        logger.info(f"📈 分析图表: {data_path}analysis_plots/")
        logger.info(f"📝 日志文件: {data_path}full_comparison_log.txt")
        
        # 检查汇总文件
        summary_files = [
            'all_models_summary_with_baseline.json',
            'all_models_summary_with_scheduler.json'
        ]
        
        for summary_file in summary_files:
            summary_path = Path(data_path) / 'results' / summary_file
            if summary_path.exists():
                logger.info(f"📄 模型汇总: {summary_path}")
                break
    
    return analyzer, summary

if __name__ == "__main__":
    analyzer, summary = main()
    
    # 如果运行成功，提供简单的交互提示
    if analyzer is not None:
        print("\n" + "="*60)
        print("🎉 所有模型训练和分析完成！")
        print("="*60)
        print("主要输出文件:")
        print(f"• 模型性能汇总: {data_path}results/all_models_summary_with_baseline.json")
        print(f"• 性能对比图表: {data_path}analysis_plots/")
        print(f"• 训练日志: {data_path}full_comparison_log.txt")
        print("="*60)
    else:
        print("\n❌ 流程执行失败，请检查日志文件获取详细信息")