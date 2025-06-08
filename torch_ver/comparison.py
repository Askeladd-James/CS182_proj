"""
完整的模型训练、评估和比较流程
"""
import subprocess
import sys
import logging
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

def run_training():
    """运行模型训练"""
    logger = logging.getLogger(__name__)
    logger.info("开始模型训练阶段...")
    
    try:
        # 运行训练脚本
        import train_comparison
        results = train_comparison.main()
        logger.info("模型训练完成!")
        return True
    except Exception as e:
        logger.error(f"训练过程中出错: {str(e)}")
        return False

def run_analysis():
    """运行模型分析"""
    logger = logging.getLogger(__name__)
    logger.info("开始模型分析阶段...")
    
    try:
        # 运行分析脚本
        import model_comparison
        analyzer, summary = model_comparison.main()
        logger.info("模型分析完成!")
        return analyzer, summary
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")
        return None, None

def main():
    """主流程"""
    logger = setup_logging()
    logger.info("开始完整的模型比较流程...")
    
    # 创建必要的目录
    Path(data_path + 'model').mkdir(exist_ok=True)
    Path(data_path + 'results').mkdir(exist_ok=True)
    Path(data_path + 'analysis_plots').mkdir(exist_ok=True)
    
    # 阶段1：训练所有模型
    logger.info("=" * 60)
    logger.info("阶段1：训练所有模型")
    logger.info("=" * 60)
    
    training_success = run_training()
    
    if not training_success:
        logger.error("训练阶段失败，终止流程")
        return
    
    # 阶段2：分析和比较
    logger.info("=" * 60)
    logger.info("阶段2：模型分析和比较")
    logger.info("=" * 60)
    
    analyzer, summary = run_analysis()
    
    if analyzer is None:
        logger.error("分析阶段失败")
        return
    
    # 完成
    logger.info("=" * 60)
    logger.info("完整流程执行完成！")
    logger.info("=" * 60)
    logger.info(f"结果保存在: {data_path}")
    logger.info("- 模型文件: model/")
    logger.info("- 结果数据: results/")
    logger.info("- 分析图表: analysis_plots/")
    
    return analyzer, summary

if __name__ == "__main__":
    analyzer, summary = main()