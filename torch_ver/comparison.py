"""
完整的模型训练、评估和比较流程（支持分离模式）
"""
import logging
import time
import argparse
from pathlib import Path
from data_process import data_path

def setup_logging(mode="full"):
    """设置日志 - 根据模式选择日志文件名"""
    if mode == "train":
        log_file = Path(data_path) / 'training_only_log.txt'
    elif mode == "analysis":
        log_file = Path(data_path) / 'analysis_only_log.txt'
    else:
        log_file = Path(data_path) / 'full_comparison_log.txt'
    
    # 每次运行时删除旧的日志文件
    if log_file.exists():
        log_file.unlink()
        print(f"🗑️  已清空旧日志文件: {log_file}")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8', mode='w'),
            logging.StreamHandler()
        ],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"🆕 {mode.upper()} 模式开始")
    logger.info("=" * 80)
    
    return logger

def check_training_results_exist():
    """检查是否已有训练结果"""
    results_dir = Path(data_path) / "results"
    model_dir = Path(data_path) / "model"
    
    if not results_dir.exists() or not model_dir.exists():
        return False, "结果或模型目录不存在"
    
    # 检查关键结果文件
    required_files = [
        "results_UserTimeModel_with_scheduler.json",
        "results_IndependentTimeModel_with_scheduler.json", 
        "results_UMTimeModel_with_scheduler.json",
        "results_TwoStageMMoE_with_scheduler.json",
        "results_baseline_with_scheduler.json"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = results_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
    
    if missing_files:
        return False, f"缺少结果文件: {missing_files}"
    
    # 检查汇总文件
    summary_files = [
        "all_models_summary_with_baseline_new.json",
        "all_models_summary_with_baseline.json"
    ]
    
    has_summary = any((results_dir / f).exists() for f in summary_files)
    if not has_summary:
        return False, "缺少汇总文件"
    
    return True, "所有训练结果文件都存在"

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
        status = "✅ 成功" if success else "❌ 失败" if success is False else "⏭️ 跳过"
        logger.info(f"{step_name}: {status}")
    
    logger.info(f"\n总体进度: {success_count}/{total_steps} 步骤完成")
    
    if success_count == total_steps:
        logger.info("🎉 所有步骤都成功完成！")
    elif success_count > 0:
        logger.info("⚠️  部分步骤完成，请检查失败的步骤")
    else:
        logger.info("❌ 所有步骤都失败了，请检查配置和数据")
    
    return success_count == total_steps

def main(mode="full", force_retrain=False):
    """
    主流程 - 支持不同运行模式
    
    Args:
        mode: "full" | "train" | "analysis" 
        force_retrain: 是否强制重新训练
    """
    start_time = time.time()
    logger = setup_logging(mode)
    
    logger.info(f"🚀 开始{mode.upper()}模式流程...")
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
    
    training_success = None
    analysis_success = None
    training_results = None
    analyzer = None
    summary = None
    
    # 🎯 分离逻辑：根据模式决定执行哪些步骤
    if mode in ["full", "train"]:
        # 检查是否需要训练
        should_train = force_retrain
        
        if not should_train:
            has_results, result_msg = check_training_results_exist()
            if not has_results:
                logger.info(f"📋 检测结果: {result_msg}")
                logger.info("🔄 需要进行模型训练")
                should_train = True
            else:
                logger.info(f"📋 检测结果: {result_msg}")
                if mode == "train":
                    logger.info("⚠️  结果已存在，但您选择了仅训练模式")
                    should_train = True
                else:
                    logger.info("✅ 发现已有训练结果，跳过训练阶段")
        
        if should_train:
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
                
                # 如果是仅训练模式且失败，直接返回
                if mode == "train":
                    return None, None
        else:
            training_success = True  # 跳过但标记为成功
            logger.info("⏭️  跳过训练阶段（使用已有结果）")
    
    if mode in ["full", "analysis"]:
        # 分析阶段
        if mode == "analysis":
            # 仅分析模式：再次检查训练结果
            has_results, result_msg = check_training_results_exist()
            if not has_results:
                logger.error(f"❌ 无法进行分析: {result_msg}")
                logger.error("💡 建议先运行训练模式: python comparison.py --mode train")
                return None, None
            else:
                logger.info(f"✅ 发现训练结果: {result_msg}")
        
        # 只有在训练成功或跳过训练时才进行分析
        if training_success is not False:
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
                analysis_success = False
                logger.error("❌ 模型分析失败")
        else:
            logger.error("❌ 训练失败，跳过分析阶段")
            analysis_success = False
    
    # 总结
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 80)
    logger.info(f"🏁 {mode.upper()}模式执行完成！")
    logger.info("=" * 80)
    
    # 打印详细的总结
    all_success = print_progress_summary(training_success, analysis_success)
    
    logger.info(f"\n⏱️  总执行时间: {total_time:.2f}秒")
    
    if training_success and mode in ["full", "train"]:
        logger.info(f"   - 模型训练: {training_time:.2f}秒")
    if analysis_success and mode in ["full", "analysis"]:
        logger.info(f"   - 模型分析: {analysis_time:.2f}秒")
    
    # 输出文件位置信息
    if all_success or (mode == "analysis" and analysis_success):
        logger.info(f"\n📋 生成的文件:")
        logger.info(f"📊 结果文件: {data_path}results/")
        logger.info(f"🤖 模型文件: {data_path}model/")
        
        if analysis_success:
            logger.info(f"📈 分析图表: {data_path}analysis_plots/")
        
        if mode == "train":
            logger.info(f"📝 训练日志: {data_path}training_only_log.txt")
        elif mode == "analysis":
            logger.info(f"📝 分析日志: {data_path}analysis_only_log.txt")
        else:
            logger.info(f"📝 完整日志: {data_path}full_comparison_log.txt")
        
        # 检查汇总文件
        if mode in ["full", "train"]:
            summary_files = [
                'all_models_summary_with_baseline_new.json',
                'all_models_summary_with_baseline.json'
            ]
            
            for summary_file in summary_files:
                summary_path = Path(data_path) / 'results' / summary_file
                if summary_path.exists():
                    logger.info(f"📄 模型汇总: {summary_path}")
                    break
    
    return analyzer, summary

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型训练和分析工具")
    parser.add_argument(
        "--mode", 
        choices=["full", "train", "analysis"], 
        default="full",
        help="运行模式: full(完整流程), train(仅训练), analysis(仅分析)"
    )
    parser.add_argument(
        "--force-retrain", 
        action="store_true",
        help="强制重新训练（即使已有结果）"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # 支持命令行参数
    args = parse_arguments()
    
    print("=" * 60)
    print("🎯 模型训练和分析工具")
    print("=" * 60)
    print(f"运行模式: {args.mode.upper()}")
    if args.force_retrain:
        print("🔄 强制重新训练模式")
    print("=" * 60)
    
    analyzer, summary = main(mode=args.mode, force_retrain=args.force_retrain)
    
    # 根据模式提供不同的提示
    if analyzer is not None or (args.mode == "train" and summary is not None):
        print("\n" + "="*60)
        
        if args.mode == "train":
            print("🎉 模型训练完成！")
            print("💡 运行分析: python comparison.py --mode analysis")
        elif args.mode == "analysis":
            print("🎉 模型分析完成！")
            print("📈 查看生成的图表文件")
        else:
            print("🎉 完整流程完成！")
        
        print("="*60)
        print("主要输出文件:")
        
        if args.mode in ["full", "train"]:
            print(f"• 模型汇总: {data_path}results/all_models_summary_with_baseline_new.json")
        
        if args.mode in ["full", "analysis"]:
            print(f"• 分析图表: {data_path}analysis_plots/")
        
        print(f"• 日志文件: {data_path}{args.mode}_log.txt" if args.mode != "full" else f"{data_path}full_comparison_log.txt")
        print("="*60)
    else:
        print("\n❌ 流程执行失败，请检查日志文件获取详细信息")