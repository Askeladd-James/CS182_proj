"""
完整的模型训练、评估和比较流程（包括baseline）
"""
import subprocess
import sys
import logging
import time
from pathlib import Path
from data_process import data_path

def setup_logging():
    """设置日志"""
    log_file = Path(data_path) / 'full_comparison_log.txt'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_baseline_training():
    """运行baseline模型训练"""
    logger = logging.getLogger(__name__)
    logger.info("开始训练Baseline模型...")
    
    try:
        # 添加origin目录到路径
        origin_path = Path(__file__).parent.parent / 'origin'
        if str(origin_path) not in sys.path:
            sys.path.insert(0, str(origin_path))
        
        # 导入并运行baseline训练
        import train as baseline_train
        
        # 运行baseline训练
        model, test_data, results = baseline_train.main()
        logger.info("✅ Baseline模型训练完成!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Baseline训练过程中出错: {str(e)}")
        logger.error(f"错误详情: {type(e).__name__}")
        return False
    finally:
        # 清理路径
        if str(origin_path) in sys.path:
            sys.path.remove(str(origin_path))

def run_time_aware_training():
    """运行时间感知模型训练"""
    logger = logging.getLogger(__name__)
    logger.info("开始训练时间感知模型...")
    
    try:
        # 运行时间感知模型训练
        import train_comparison
        results = train_comparison.main()
        logger.info("✅ 时间感知模型训练完成!")
        return True
    except Exception as e:
        logger.error(f"❌ 时间感知模型训练过程中出错: {str(e)}")
        return False

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

def print_progress_summary(baseline_success, time_aware_success, analysis_success):
    """打印进度总结"""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "=" * 80)
    logger.info("📊 执行进度总结")
    logger.info("=" * 80)
    
    steps = [
        ("Baseline模型训练", baseline_success),
        ("时间感知模型训练", time_aware_success), 
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
    """主流程 - 包含baseline和时间感知模型的完整训练流程"""
    start_time = time.time()
    logger = setup_logging()
    
    logger.info("🚀 开始完整的模型比较流程（包括Baseline）...")
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
    
    # 初始化结果追踪
    baseline_success = False
    time_aware_success = False
    analysis_success = False
    
    # 阶段1：训练Baseline模型
    logger.info("\n" + "=" * 80)
    logger.info("🔄 阶段1：训练Baseline模型")
    logger.info("=" * 80)
    
    baseline_start = time.time()
    baseline_success = run_baseline_training()
    baseline_time = time.time() - baseline_start
    
    if baseline_success:
        logger.info(f"✅ Baseline训练完成，耗时: {baseline_time:.2f}秒")
    else:
        logger.warning("⚠️  Baseline训练失败，但继续执行其他模型")
    
    # 阶段2：训练时间感知模型（包括MMOE）
    logger.info("\n" + "=" * 80)
    logger.info("🔄 阶段2：训练时间感知模型（包括MMOE）")
    logger.info("=" * 80)
    
    time_aware_start = time.time()
    time_aware_success = run_time_aware_training()
    time_aware_time = time.time() - time_aware_start
    
    if time_aware_success:
        logger.info(f"✅ 时间感知模型训练完成，耗时: {time_aware_time:.2f}秒")
    else:
        logger.error("❌ 时间感知模型训练失败")
    
    # 阶段3：模型分析和比较（只有在至少有一个模型训练成功时才执行）
    if baseline_success or time_aware_success:
        logger.info("\n" + "=" * 80)
        logger.info("🔄 阶段3：模型分析和比较")
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
        logger.error("❌ 没有成功训练的模型，跳过分析阶段")
        analyzer, summary = None, None
    
    # 总结
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info("🏁 完整流程执行完成！")
    logger.info("=" * 80)
    
    # 打印详细的总结
    all_success = print_progress_summary(baseline_success, time_aware_success, analysis_success)
    
    logger.info(f"\n⏱️  总执行时间: {total_time:.2f}秒")
    
    if baseline_success:
        logger.info(f"   - Baseline训练: {baseline_time:.2f}秒")
    if time_aware_success:
        logger.info(f"   - 时间感知模型训练: {time_aware_time:.2f}秒")
    if analysis_success:
        logger.info(f"   - 模型分析: {analysis_time:.2f}秒")
    
    # 输出文件位置信息
    if all_success:
        logger.info(f"\n📋 生成的文件:")
        logger.info(f"📊 结果文件: {data_path}results/")
        logger.info(f"🤖 模型文件: {data_path}model/")
        logger.info(f"📈 分析图表: {data_path}analysis_plots/")
        logger.info(f"📝 日志文件: {data_path}full_comparison_log.txt")
        
        # 如果有具体的汇总文件，也记录一下
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