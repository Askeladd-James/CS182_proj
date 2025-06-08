"""
独立的模型分析脚本 - 可在已有训练结果基础上生成图表
"""
import argparse
from pathlib import Path
from data_process import data_path
from train_comparison import rebuild_summary_standalone
import json

def check_and_rebuild_summary():
    """检查并重建汇总文件"""
    summary_path = Path(data_path).parent / "results" / "all_models_summary_with_baseline_new.json"
    
    print("📄 检查汇总文件...")
    
    if not summary_path.exists():
        print("❌ 汇总文件不存在，正在重建...")
        summary_data = rebuild_summary_standalone()
        if not summary_data:
            print("❌ 重建汇总文件失败")
            return None
    else:
        print(f"✅ 发现汇总文件: {summary_path.name}")
        
        # 读取并验证汇总文件
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
            
            if not summary_data:
                print("⚠️ 汇总文件为空，正在重建...")
                summary_data = rebuild_summary_standalone()
            else:
                print(f"📊 汇总文件包含 {len(summary_data)} 个模型")
                
        except Exception as e:
            print(f"❌ 读取汇总文件失败: {e}")
            print("🔄 正在重建汇总文件...")
            summary_data = rebuild_summary_standalone()
    
    return summary_data

def validate_summary_data(summary_data):
    """验证汇总数据的完整性"""
    if not summary_data:
        return False, "汇总数据为空"
    
    # 检查必要的模型类型
    expected_models = [
        "CFModel_Baseline",
        "UserTime", 
        "IndependentTime",
        "UMTime",
        "TwoStage_MMoE"
    ]
    
    missing_models = []
    for model_type in expected_models:
        if model_type not in summary_data:
            missing_models.append(model_type)
    
    if missing_models:
        return False, f"缺少模型: {missing_models}"
    
    # 检查每个模型的必要数据
    for model_type, data in summary_data.items():
        if not isinstance(data, dict):
            return False, f"模型 {model_type} 数据格式错误"
        
        if "test_metrics" not in data:
            return False, f"模型 {model_type} 缺少测试指标"
        
        if "training_history" not in data:
            return False, f"模型 {model_type} 缺少训练历史"
    
    return True, "数据验证通过"

def main():
    """独立分析主函数"""
    print("🔍 独立模型分析工具")
    print("=" * 60)
    
    # 检查训练结果目录是否存在
    results_dir = Path(data_path) / "results"
    if not results_dir.exists():
        print("❌ 结果目录不存在，请先运行模型训练")
        print("💡 运行: python train_comparison.py")
        return None
    
    # 🔧 新增：检查并处理汇总文件
    summary_data = check_and_rebuild_summary()
    if not summary_data:
        print("❌ 无法获取有效的汇总数据")
        return None
    
    # 🔧 新增：验证汇总数据
    is_valid, message = validate_summary_data(summary_data)
    if not is_valid:
        print(f"❌ 汇总数据验证失败: {message}")
        print("🔄 尝试重建汇总文件...")
        summary_data = rebuild_summary_standalone()
        
        if not summary_data:
            print("❌ 重建后仍无法获取有效数据")
            return None
        
        # 再次验证
        is_valid, message = validate_summary_data(summary_data)
        if not is_valid:
            print(f"❌ 重建后验证仍失败: {message}")
            return None
    
    print(f"✅ 数据验证通过: {message}")
    
    # 🔧 新增：显示可用模型信息
    print("\n📋 可用模型:")
    for model_type, data in summary_data.items():
        model_name = data.get("model_name", model_type)
        test_rmse = data.get("test_metrics", {}).get("RMSE", "N/A")
        print(f"  - {model_name}: RMSE = {test_rmse}")
    
    print("\n🚀 开始生成分析图表...")
    
    try:
        # 🔧 修改：使用汇总文件路径运行分析
        import model_comparison
        
        # 传递汇总文件路径给分析器
        summary_path = Path(data_path) / "results" / "all_models_summary_with_baseline_new.json"
        analyzer = model_comparison.ModelComparison(results_path=str(summary_path))
        
        # 运行完整分析
        analyzer.run_complete_analysis()
        
        print("✅ 分析完成！")
        print("\n📈 生成的图表:")
        
        # 列出生成的图表文件
        plots_dir = analyzer.output_dir
        if plots_dir.exists():
            plot_files = list(plots_dir.glob("*.png"))
            plot_files.sort()
            
            for i, plot_file in enumerate(plot_files, 1):
                print(f"  {i:2d}. {plot_file.name}")
            
            print(f"\n📁 图表保存位置: {plots_dir}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ 分析过程出错: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None

def rebuild_summary_only():
    """仅重建汇总文件的独立函数"""
    print("🔄 重建汇总文件模式")
    print("=" * 60)
    
    try:
        summary_data = rebuild_summary_standalone()
        
        if summary_data:
            print(f"✅ 汇总文件重建成功! 包含 {len(summary_data)} 个模型")
            
            # 验证重建的数据
            is_valid, message = validate_summary_data(summary_data)
            if is_valid:
                print(f"✅ 数据验证通过: {message}")
            else:
                print(f"⚠️ 数据验证警告: {message}")
            
            return summary_data
        else:
            print("❌ 汇总文件重建失败")
            return None
            
    except Exception as e:
        print(f"❌ 重建过程中出错: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None

if __name__ == "__main__":
    # 🔧 新增：支持命令行参数
    parser = argparse.ArgumentParser(description="模型分析工具")
    parser.add_argument(
        "--mode", 
        choices=["analyze", "rebuild"], 
        default="analyze",
        help="运行模式: analyze=生成分析图表, rebuild=重建汇总文件"
    )
    
    args = parser.parse_args()
    
    if args.mode == "rebuild":
        # 仅重建汇总文件
        summary_data = rebuild_summary_only()
        
        if summary_data:
            print("\n" + "="*60)
            print("🎉 汇总文件重建完成！")
            print("💡 现在可以运行: python run_analysis.py --mode analyze")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 汇总文件重建失败")
            print("="*60)
    
    else:
        # 运行完整分析
        analyzer = main()
        
        if analyzer:
            print("\n" + "="*60)
            print("🎉 独立分析完成！")
            print("📊 所有对比图表已生成")
            print("="*60)
        else:
            print("\n" + "="*60)
            print("❌ 分析失败")
            print("💡 建议运行: python run_analysis.py --mode rebuild")
            print("="*60)