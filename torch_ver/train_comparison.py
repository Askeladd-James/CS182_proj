import torch
import torch.nn as nn
import torch.optim as optim
from data_process import (
    load_data,
    create_time_aware_split,
    save_split_data,
    check_split_data_exists,
    load_existing_split_data,
    MovieLensDataset,
    data_path,
)
from model import (
    IndependentTimeModel,
    UserTimeModel,
    UMTimeModel,
    TwoStageMMoEModel,
    CFModel,
)
from torch.utils.data import DataLoader
import logging
import math
import pandas as pd
import json
import time
import numpy as np
from pathlib import Path

# ============= 模型训练选择配置 =============

MODELS_TO_TRAIN = {
    # 🔧 取消注释来训练对应的模型
    # "baseline": True,
    # "usertime": True,
    # "independent": True,
    # "umtime": True,
    "mmoe": True,
}

FORCE_RETRAIN_ALL = False

# 单独控制每个模型的强制重训练
FORCE_RETRAIN = {
    # "baseline": True,
    # "usertime": True,
    # "independent": False,
    # "umtime": True,
    "mmoe": True,
}
# ===============================================


def get_model_file_mapping():
    """获取模型类型与文件名的映射关系 - 修复模型名称"""
    return {
        "baseline": {
            "result_file": "results_baseline_with_scheduler.json",
            "model_file": "model_checkpoint_baseline_with_scheduler.pt",
            "model_type": "CFModel_Baseline",
            "display_name": "Baseline Collaborative Filtering",  # 添加显示名称
        },
        "usertime": {
            "result_file": "results_UserTimeModel_with_scheduler.json",
            "model_file": "model_checkpoint_UserTimeModel_with_scheduler.pt",
            "model_type": "UserTime",
            "display_name": "User Time-Aware Model",
        },
        "independent": {
            "result_file": "results_IndependentTimeModel_with_scheduler.json",
            "model_file": "model_checkpoint_IndependentTimeModel_with_scheduler.pt",
            "model_type": "IndependentTime",
            "display_name": "Independent Time Feature Model",  # 修复：应该是这个名称
        },
        "umtime": {
            "result_file": "results_UMTimeModel_with_scheduler.json",
            "model_file": "model_checkpoint_UMTimeModel_with_scheduler.pt",
            "model_type": "UMTime",
            "display_name": "User-Movie Time-Aware Model",
        },
        "mmoe": {
            "result_file": "results_TwoStageMMoE_with_scheduler.json",
            "model_file": "model_checkpoint_TwoStageMMoE_with_scheduler.pt",
            "model_type": "TwoStage_MMoE",
            "display_name": "Two-Stage MMoE Model",
        },
    }


def should_train_model(model_key):
    """判断是否应该训练某个模型"""
    # 检查是否在训练列表中
    if not MODELS_TO_TRAIN.get(model_key, False):
        return False, f"❌ {model_key} 未被选中训练"

    # 检查是否强制重训练
    if FORCE_RETRAIN_ALL or FORCE_RETRAIN.get(model_key, False):
        return True, f"🔄 {model_key} 强制重新训练"

    # 检查结果文件是否存在
    file_mapping = get_model_file_mapping()
    results_path = Path(data_path) / "results" / file_mapping[model_key]["result_file"]

    if results_path.exists():
        return False, f"⏭️ {model_key} 结果已存在，跳过训练"
    else:
        return True, f"📝 {model_key} 结果不存在，需要训练"


def clear_specific_model_files(model_key):
    """清除特定模型的结果和模型文件"""
    file_mapping = get_model_file_mapping()
    model_info = file_mapping[model_key]

    results_dir = Path(data_path) / "results"
    model_dir = Path(data_path) / "model"

    # 清除结果文件
    result_path = results_dir / model_info["result_file"]
    if result_path.exists():
        try:
            result_path.unlink()
            print(f"🗑️ 删除旧结果文件: {result_path.name}")
        except Exception as e:
            print(f"❌ 删除结果文件失败 {result_path}: {e}")

    # 清除模型文件
    model_path = model_dir / model_info["model_file"]
    if model_path.exists():
        try:
            model_path.unlink()
            print(f"🗑️ 删除旧模型文件: {model_path.name}")
        except Exception as e:
            print(f"❌ 删除模型文件失败 {model_path}: {e}")


def prepare_training_environment():
    """准备训练环境，只清理需要训练的模型"""
    results_dir = Path(data_path) / "results"
    model_dir = Path(data_path) / "model"

    # 确保目录存在
    results_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("🔍 检查训练计划...")
    print("=" * 60)

    models_to_train_list = []
    models_to_skip_list = []

    for model_key in ["baseline", "usertime", "independent", "umtime", "mmoe"]:
        should_train, reason = should_train_model(model_key)

        if should_train:
            models_to_train_list.append(model_key)
            print(f"✅ {reason}")
            # 清理要训练的模型的旧文件
            clear_specific_model_files(model_key)
        else:
            models_to_skip_list.append(model_key)
            print(f"{reason}")

    print("=" * 60)
    print(f"📋 训练计划汇总:")
    print(f"   🎯 将要训练: {len(models_to_train_list)} 个模型")
    for model in models_to_train_list:
        print(f"      - {model}")

    print(f"   ⏭️ 将要跳过: {len(models_to_skip_list)} 个模型")
    for model in models_to_skip_list:
        print(f"      - {model}")

    print("=" * 60)

    return models_to_train_list


def convert_numpy_types(obj):
    """递归转换 NumPy 类型为 Python 原生类型，用于 JSON 序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


def save_results_safely(results, results_path, model_name):
    """Safely save results JSON, ensure directory exists and overwrite old files"""
    try:
        # Ensure directory exists
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)

        # 🔧 修复：转换 NumPy 类型为 Python 原生类型
        json_compatible_results = convert_numpy_types(results)

        # Write new results
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(json_compatible_results, f, indent=2, ensure_ascii=False)

        print(f"💾 保存 {model_name} 结果到: {Path(results_path).name}")

    except Exception as e:
        logging.error(f"Failed to save result file {results_path}: {e}")
        raise


def prepare_config_for_json(config):
    """Prepare config for JSON serialization (convert device to string and handle NumPy types)"""
    json_config = config.copy()
    if "DEVICE" in json_config:
        json_config["DEVICE"] = str(json_config["DEVICE"])

    # 🔧 新增：转换配置中的 NumPy 类型
    json_config = convert_numpy_types(json_config)

    return json_config


def save_model_safely(checkpoint, model_path, model_name):
    """Safely save model checkpoint, ensure directory exists and overwrite old files"""
    try:
        # Ensure directory exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        # Save new model
        torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
        print(f"💾 保存 {model_name} 模型到: {Path(model_path).name}")

    except Exception as e:
        logging.error(f"Failed to save model file {model_path}: {e}")
        raise


def train_model_with_metrics(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=30,
    patience=10,
    display_name=None,  # 新增参数
):
    """Train model and record detailed training metrics"""
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Record training metrics
    training_history = {
        "train_losses": [],
        "val_losses": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rates": [],
        "epoch_times": [],
        "best_epoch": 0,
        "total_epochs": 0,
    }

    # 🔧 修复：使用传入的显示名称
    model_display_name = (
        display_name
        if display_name
        else (model.name if hasattr(model, "name") else "Model")
    )
    logging.info(f"开始训练模型: {model_display_name}")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        for user, movie, rating, daytime, weekend, year in train_loader:
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)
            daytime = daytime.to(device)
            weekend = weekend.to(device)
            year = year.to(device)

            optimizer.zero_grad()
            prediction = model(user, movie, daytime, weekend, year)

            # Calculate loss (MSE + L2 regularization)
            mse_loss = criterion(prediction, rating)

            # Add regularization loss if model has the method
            if hasattr(model, "get_regularization_loss"):
                reg_loss = model.get_regularization_loss()
                total_loss = mse_loss + reg_loss
            else:
                total_loss = mse_loss

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += mse_loss.item()  # Only record MSE for comparison
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        train_rmse = math.sqrt(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for user, movie, rating, daytime, weekend, year in val_loader:
                user = user.to(device)
                movie = movie.to(device)
                rating = rating.to(device)
                daytime = daytime.to(device)
                weekend = weekend.to(device)
                year = year.to(device)

                prediction = model(user, movie, daytime, weekend, year)
                loss = criterion(prediction, rating)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_rmse = math.sqrt(avg_val_loss)

        # Record learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)

        # Record metrics
        training_history["train_losses"].append(avg_train_loss)
        training_history["val_losses"].append(avg_val_loss)
        training_history["train_rmse"].append(train_rmse)
        training_history["val_rmse"].append(val_rmse)
        training_history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start
        training_history["epoch_times"].append(epoch_time)

        # Print training info every 5 epochs or at the end
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            logging.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            logging.info(
                f"Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}"
            )
            logging.info(f"Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            logging.info(f"Learning rate: {current_lr:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            training_history["best_epoch"] = epoch
            if (epoch + 1) % 5 == 0 or epoch < 3:
                logging.info(f"🎯 新的最佳验证损失: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"⏰ 早停于第 {epoch+1} 轮")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

    training_history["total_epochs"] = epoch + 1
    total_time = time.time() - start_time
    training_history["total_training_time"] = total_time

    # Final statistics
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f"✅ 训练完成! 总时间: {total_time:.2f}s")
    logging.info(f"📊 最佳验证损失: {best_val_loss:.4f}")
    logging.info(f"📊 最佳验证RMSE: {best_rmse:.4f}")

    return best_model_state, training_history


def evaluate_model_detailed(model, test_data, device):
    """Detailed model performance evaluation"""
    model.eval()
    predictions = []
    actuals = []

    start_time = time.time()

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row["user_emb_id"]]).to(device)
            movie_id = torch.LongTensor([row["movie_emb_id"]]).to(device)
            daytime = torch.LongTensor([row["daytime"]]).to(device)
            weekend = torch.LongTensor([row["is_weekend"]]).to(device)
            year = torch.LongTensor([row["year"]]).to(device)

            pred = model(user_id, movie_id, daytime, weekend, year)
            predictions.append(pred.cpu().item())
            actuals.append(row["rating"])

    inference_time = time.time() - start_time

    # Calculate various evaluation metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Basic metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Other metrics
    mape = (
        np.mean(np.abs((actuals - predictions) / actuals)) * 100
    )  # Mean Absolute Percentage Error

    # Rating distribution statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)

    # Correlation coefficient
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Accuracy by rating range
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f"rating_{rating}_mae"] = rating_mae

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
        "Inference_Time": inference_time,
        "Predictions_Mean": pred_mean,
        "Predictions_Std": pred_std,
        "Actuals_Mean": actual_mean,
        "Actuals_Std": actual_std,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        **rating_accuracy,
    }


def train_and_evaluate_model(
    model_class,
    model_name,
    max_userid,
    max_movieid,
    train_loader,
    val_loader,
    test_data,
    device,
    config,
):
    """训练和评估传统模型 - 修复模型名称问题"""
    logging.info(f"\n{'='*60}")
    logging.info(f"🚀 开始训练和评估: {model_name}")
    logging.info(f"{'='*60}")

    # Create model
    model = model_class(
        max_userid + 1,
        max_movieid + 1,
        config["K_FACTORS"],
        config["TIME_FACTORS"],
        config["REG_STRENGTH"],
    ).to(device)

    # 🔧 修复：使用传入的正确模型名称，而不是依赖模型内部属性
    display_name = model_name
    model_type_name = model.name if hasattr(model, "name") else model_class.__name__

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(
        f"📊 {display_name} 参数统计: 总参数={total_params:,}, 可训练={trainable_params:,}"
    )
    logging.info(f"📊 模型类型: {model_type_name}")

    # Setup optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-6
    )

    # Use ReduceLROnPlateau scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=5, min_lr=1e-6
    )

    # Train model
    best_model_state, training_history = train_model_with_metrics(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config["NUM_EPOCHS"],
        patience=10,
        display_name=display_name,
    )

    # Evaluate model
    logging.info(f"📈 开始评估: {display_name}")
    test_metrics = evaluate_model_detailed(model, test_data, device)

    # 🔧 修复：确保结果中使用正确的模型名称
    results = {
        "model_name": display_name,  # 使用传入的正确名称
        "model_type": model_type_name,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "k_factors": config["K_FACTORS"],
            "time_factors": config["TIME_FACTORS"],
            "reg_strength": config["REG_STRENGTH"],
        },
        "training_config": prepare_config_for_json(config),
    }

    # Convert NumPy types
    results = convert_numpy_types(results)

    # Save model checkpoint
    checkpoint = {
        "max_userid": max_userid,
        "max_movieid": max_movieid,
        "k_factors": config["K_FACTORS"],
        "time_factors": config["TIME_FACTORS"],
        "reg_strength": config["REG_STRENGTH"],
        "best_model_state": model.state_dict(),
        "model_type": model_type_name,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "has_scheduler": True,
    }

    model_path = (
        data_path + f"model/model_checkpoint_{model_type_name}_with_scheduler.pt"
    )
    save_model_safely(checkpoint, model_path, display_name)

    # Save results JSON
    results_path = data_path + f"results/results_{model_type_name}_with_scheduler.json"
    save_results_safely(results, results_path, display_name)

    logging.info(f"✅ 模型 {display_name} 训练和评估完成!")
    logging.info(f"📊 测试 RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"📊 测试 MAE: {test_metrics['MAE']:.4f}")

    return results


def train_and_evaluate_mmoe_model(
    model_name, max_userid, max_movieid, train_data, val_data, test_data, device, config
):
    """修复的MMOE模型训练和评估函数"""
    logging.info(f"\n{'='*60}")
    logging.info(f"🚀 开始训练和评估 MMOE: {model_name}")
    logging.info(f"{'='*60}")

    # Create MMOE model
    model = TwoStageMMoEModel(
        max_userid + 1,
        max_movieid + 1,
        config["K_FACTORS"],
        config["TIME_FACTORS"],
        config["REG_STRENGTH"],
        config["NUM_EXPERTS"],
    ).to(device)

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(
        f"📊 MMOE 模型参数: 总参数={total_params:,}, 可训练={trainable_params:,}"
    )

    criterion = nn.MSELoss()
    start_time = time.time()

    # Use MMOE training function
    from MMOE_train import train_mmoe

    try:
        model, training_history = train_mmoe(
            model,
            train_data,
            val_data,
            device,
            batch_size=config["BATCH_SIZE"],
            num_epochs_per_stage=config.get("NUM_EPOCHS_PER_STAGE", [30, 30, 30]),
            learning_rates=config.get("LEARNING_RATES", [0.0005, 0.001, 0.0005]),
        )

        # 🔧 修复：现在training_history已经是统一格式，直接使用即可
        total_training_time = training_history.get('total_training_time', time.time() - start_time)
        
        # 确保所有必要字段存在
        if 'total_training_time' not in training_history:
            training_history['total_training_time'] = total_training_time
            
        # 打印阶段信息（如果存在）
        if 'stage_info' in training_history:
            logging.info("📊 各阶段训练详情:")
            for stage in training_history['stage_info']:
                logging.info(f"  {stage['stage_name']}: {stage['epochs']} epochs, "
                           f"{stage['training_time']:.1f}s, 最佳损失: {stage['best_loss']:.4f}")

    except Exception as e:
        logging.error(f"MMOE训练过程中出错: {str(e)}")
        import traceback
        logging.error(f"详细错误信息: {traceback.format_exc()}")

        # 使用默认历史记录
        training_history = {
            'train_losses': [],
            'val_losses': [],
            'train_rmse': [],
            'val_rmse': [],
            'learning_rates': [],
            'epoch_times': [],
            'best_epoch': 0,
            'total_epochs': 0,
            'total_training_time': time.time() - start_time
        }
        raise

    # Evaluate MMOE model
    model.set_training_stage(4)  # Unfreeze all parameters for evaluation
    logging.info(f"📈 开始MMOE模型评估: {model_name}")
    test_metrics = evaluate_mmoe_model(model, test_data, device)

    # 🔧 修复：简化结果组织，直接使用统一的training_history
    results = {
        "model_name": model_name,
        "model_type": model.name if hasattr(model, "name") else "TwoStageMMoE",
        "training_history": training_history,  # 现在是统一格式
        "test_metrics": test_metrics,
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "k_factors": config["K_FACTORS"],
            "time_factors": config["TIME_FACTORS"],
            "reg_strength": config["REG_STRENGTH"],
            "num_experts": config["NUM_EXPERTS"],
        },
        "training_config": prepare_config_for_json(config),
    }

    # Convert NumPy types
    results = convert_numpy_types(results)

    # Save model checkpoint
    model_type_name = model.name if hasattr(model, "name") else "TwoStageMMoE"
    checkpoint = {
        "max_userid": max_userid,
        "max_movieid": max_movieid,
        "k_factors": config["K_FACTORS"],
        "time_factors": config["TIME_FACTORS"],
        "reg_strength": config["REG_STRENGTH"],
        "num_experts": config["NUM_EXPERTS"],
        "best_model_state": model.state_dict(),
        "model_type": model_type_name,
        "training_history": training_history,
        "test_metrics": test_metrics,
        "has_scheduler": True,
    }

    model_path = (
        data_path + f"model/model_checkpoint_{model_type_name}_with_scheduler.pt"
    )
    save_model_safely(checkpoint, model_path, model_name)

    # Save results JSON
    results_path = data_path + f"results/results_{model_type_name}_with_scheduler.json"
    save_results_safely(results, results_path, model_name)

    logging.info(f"✅ MMOE模型 {model_name} 训练和评估完成!")
    logging.info(f"📊 测试 RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"📊 测试 MAE: {test_metrics['MAE']:.4f}")

    # 🔧 修复：从统一历史中提取信息
    if training_history["train_losses"]:
        final_train_loss = training_history["train_losses"][-1]
        logging.info(f"📊 最终训练损失: {final_train_loss:.4f}")

    if training_history["val_losses"]:
        best_val_loss = min(training_history["val_losses"])
        logging.info(f"📊 最佳验证损失: {best_val_loss:.4f}")

    logging.info(f"📊 总训练轮数: {training_history['total_epochs']}")
    logging.info(f"⏱️ 训练时间: {training_history['total_training_time']:.2f} 秒")

    return results


def evaluate_mmoe_model(model, test_data, device):
    """Specialized MMOE model evaluation function - fix dimension errors"""
    model.eval()
    predictions = []
    actuals = []

    start_time = time.time()

    # Prepare user history statistical features (simplified version for evaluation)
    from MMOE_train import prepare_user_history_stats

    user_history_stats = prepare_user_history_stats(test_data)

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row["user_emb_id"]]).to(device)
            movie_id = torch.LongTensor([row["movie_emb_id"]]).to(device)
            daytime = torch.LongTensor([row["daytime"]]).to(device)
            weekend = torch.LongTensor([row["is_weekend"]]).to(device)
            year = torch.LongTensor([row["year"]]).to(device)

            # Get user history features
            user_emb_id = row["user_emb_id"]
            if user_emb_id in user_history_stats:
                history_features = (
                    torch.FloatTensor(user_history_stats[user_emb_id])
                    .unsqueeze(0)
                    .to(device)
                )
            else:
                history_features = (
                    torch.FloatTensor([3.0, 1.0, 1.0, 3.0, 0.0]).unsqueeze(0).to(device)
                )

            # Use stage 4 for complete evaluation
            model.set_training_stage(4)
            final_pred = model(
                user_id,
                movie_id,
                daytime,
                weekend,
                year,
                user_history_features=history_features,
            )

            # Ensure prediction is scalar
            if final_pred.dim() > 0:
                final_pred = final_pred.item()
            else:
                final_pred = final_pred.item()

            predictions.append(final_pred)
            actuals.append(row["rating"])

    inference_time = time.time() - start_time

    # Calculate various evaluation metrics (same as original function)
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Basic metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Other metrics
    mape = (
        np.mean(np.abs((actuals - predictions) / actuals)) * 100
    )  # Mean Absolute Percentage Error

    # Rating distribution statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)

    # Correlation coefficient
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Accuracy by rating range
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f"rating_{rating}_mae"] = rating_mae

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
        "Inference_Time": inference_time,
        "Predictions_Mean": pred_mean,
        "Predictions_Std": pred_std,
        "Actuals_Mean": actual_mean,
        "Actuals_Std": actual_std,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        **rating_accuracy,
    }


def train_and_evaluate_baseline_model(
    model_name, max_userid, max_movieid, train_data, val_data, test_data, device, config
):
    """Baseline model training and evaluation - safe save version"""
    logging.info(f"\n{'='*60}")
    logging.info(f"🚀 开始训练和评估 Baseline: {model_name}")
    logging.info(f"{'='*60}")

    # Import baseline model
    from model import CFModel

    # Create Baseline model
    model = CFModel(
        max_userid + 1,
        max_movieid + 1,
        100,  # baseline uses 100 factors
        0.0001,  # baseline uses lower regularization
    ).to(device)

    # Set model name attribute
    if not hasattr(model, "name"):
        model.name = "CFModel_Baseline"

    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(
        f"📊 Baseline 模型参数: 总参数={total_params:,}, 可训练={trainable_params:,}"
    )

    # Create baseline-specific data loaders
    baseline_train_dataset = BaselineDataset(
        train_data["user_emb_id"].values,
        train_data["movie_emb_id"].values,
        train_data["rating"].values,
    )

    baseline_val_dataset = BaselineDataset(
        val_data["user_emb_id"].values,
        val_data["movie_emb_id"].values,
        val_data["rating"].values,
    )

    baseline_train_loader = DataLoader(
        baseline_train_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        num_workers=0,
    )
    baseline_val_loader = DataLoader(
        baseline_val_dataset,
        batch_size=config["BATCH_SIZE"],
        shuffle=False,
        num_workers=0,
    )

    # Setup optimizer and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=config["LEARNING_RATE"], weight_decay=1e-6
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.6, patience=5, min_lr=1e-6
    )

    # Train model
    best_model_state, training_history = train_baseline_model_with_metrics(
        model,
        baseline_train_loader,
        baseline_val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        config["NUM_EPOCHS"],
        patience=10,
    )

    # Evaluate model
    logging.info(f"📈 开始Baseline模型评估: {model_name}")
    test_metrics = evaluate_baseline_model_detailed(model, test_data, device)

    # Organize results
    results = {
        "model_name": model_name,
        "model_type": "CFModel_Baseline",
        "training_history": training_history,
        "test_metrics": test_metrics,
        "model_params": {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "k_factors": 100,
            "reg_strength": 0.0001,
        },
        "training_config": prepare_config_for_json(config),
    }

    # 🔧 修复：在保存前转换 NumPy 类型
    results = convert_numpy_types(results)

    # Save model checkpoint - safe save
    checkpoint = {
        "max_userid": max_userid,
        "max_movieid": max_movieid,
        "k_factors": 100,
        "reg_strength": 0.0001,
        "best_model_state": model.state_dict(),
        "model_type": "CFModel_Baseline",
        "training_history": training_history,
        "test_metrics": test_metrics,
        "has_scheduler": True,
    }

    model_path = data_path + f"model/model_checkpoint_baseline_with_scheduler.pt"
    save_model_safely(checkpoint, model_path, model_name)

    # Save results JSON - safe save
    results_path = data_path + f"results/results_baseline_with_scheduler.json"
    save_results_safely(results, results_path, model_name)

    logging.info(f"✅ Baseline模型 {model_name} 训练和评估完成!")
    logging.info(f"📊 测试 RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"📊 测试 MAE: {test_metrics['MAE']:.4f}")

    return results


class BaselineDataset(torch.utils.data.Dataset):
    """Baseline model-specific dataset (no time features)"""

    def __init__(self, user_ids, movie_ids, ratings):
        self.user_ids = user_ids
        self.movie_ids = movie_ids
        self.ratings = ratings

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (
            torch.LongTensor([self.user_ids[idx]]),
            torch.LongTensor([self.movie_ids[idx]]),
            torch.FloatTensor([self.ratings[idx]]),
        )


def train_baseline_model_with_metrics(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    num_epochs=30,
    patience=10,
):
    """Specialized training function for Baseline model"""
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # Record training metrics
    training_history = {
        "train_losses": [],
        "val_losses": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rates": [],
        "epoch_times": [],
        "best_epoch": 0,
        "total_epochs": 0,
    }

    logging.info(f"开始Baseline模型训练")
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # Training phase
        model.train()
        train_loss = 0
        num_batches = 0

        for user, movie, rating in train_loader:
            user = user.squeeze().to(device)
            movie = movie.squeeze().to(device)
            rating = rating.squeeze().to(device)

            optimizer.zero_grad()
            prediction = model(user, movie)  # Baseline model only needs user and movie

            # Calculate loss (MSE + L2 regularization)
            mse_loss = criterion(prediction, rating)

            # Add regularization loss if model has the method
            if hasattr(model, "get_regularization_loss"):
                reg_loss = model.get_regularization_loss()
                total_loss = mse_loss + reg_loss
            else:
                total_loss = mse_loss

            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += mse_loss.item()  # Only record MSE for comparison
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        train_rmse = math.sqrt(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for user, movie, rating in val_loader:
                user = user.squeeze().to(device)
                movie = movie.squeeze().to(device)
                rating = rating.squeeze().to(device)

                prediction = model(
                    user, movie
                )  # Baseline model only needs user and movie
                loss = criterion(prediction, rating)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        val_rmse = math.sqrt(avg_val_loss)

        # Record learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Learning rate scheduling
        if scheduler:
            scheduler.step(avg_val_loss)

        # Record metrics
        training_history["train_losses"].append(avg_train_loss)
        training_history["val_losses"].append(avg_val_loss)
        training_history["train_rmse"].append(train_rmse)
        training_history["val_rmse"].append(val_rmse)
        training_history["learning_rates"].append(current_lr)

        epoch_time = time.time() - epoch_start
        training_history["epoch_times"].append(epoch_time)

        # Print training info every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == num_epochs - 1:
            logging.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)")
            logging.info(
                f"Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}"
            )
            logging.info(f"Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}")
            logging.info(f"Learning rate: {current_lr:.6f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            training_history["best_epoch"] = epoch
            if (epoch + 1) % 5 == 0 or epoch < 3:
                logging.info(f"🎯 新的最佳验证损失: {avg_val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logging.info(f"⏰ 早停于第 {epoch+1} 轮")
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break

    training_history["total_epochs"] = epoch + 1
    total_time = time.time() - start_time
    training_history["total_training_time"] = total_time

    # Final statistics
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f"✅ Baseline训练完成! 总时间: {total_time:.2f}s")
    logging.info(f"📊 最佳验证损失: {best_val_loss:.4f}")
    logging.info(f"📊 最佳验证RMSE: {best_rmse:.4f}")

    return best_model_state, training_history


def evaluate_baseline_model_detailed(model, test_data, device):
    """Specialized detailed evaluation function for Baseline model"""
    model.eval()
    predictions = []
    actuals = []

    start_time = time.time()

    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row["user_emb_id"]]).to(device)
            movie_id = torch.LongTensor([row["movie_emb_id"]]).to(device)

            pred = model(user_id, movie_id)  # Baseline model only needs user and movie
            predictions.append(pred.cpu().item())
            actuals.append(row["rating"])

    inference_time = time.time() - start_time

    # Calculate various evaluation metrics (same as other models)
    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Basic metrics
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))

    # Other metrics
    mape = (
        np.mean(np.abs((actuals - predictions) / actuals)) * 100
    )  # Mean Absolute Percentage Error

    # Rating distribution statistics
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)

    # Correlation coefficient
    correlation = np.corrcoef(predictions, actuals)[0, 1]

    # Accuracy by rating range
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f"rating_{rating}_mae"] = rating_mae

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Correlation": correlation,
        "Inference_Time": inference_time,
        "Predictions_Mean": pred_mean,
        "Predictions_Std": pred_std,
        "Actuals_Mean": actual_mean,
        "Actuals_Std": actual_std,
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        **rating_accuracy,
    }


def update_summary_file(all_results):
    """更新汇总文件（只包含新训练的模型）"""
    summary_path = data_path + "results/all_models_summary_with_baseline_new.json"

    # 如果汇总文件存在，读取现有结果
    existing_results = {}
    if Path(summary_path).exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            print(f"📄 加载现有汇总文件: {len(existing_results)} 个模型")
        except Exception as e:
            print(f"⚠️ 读取现有汇总文件失败: {e}")
            existing_results = {}

    # 合并结果（新训练的会覆盖旧的）
    final_results = existing_results.copy()
    final_results.update(all_results)

    try:
        # Ensure directory exists
        Path(summary_path).parent.mkdir(parents=True, exist_ok=True)

        # 🔧 修复：转换 NumPy 类型为 Python 原生类型
        json_compatible_results = convert_numpy_types(final_results)

        # Save updated summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(json_compatible_results, f, indent=2, ensure_ascii=False)

        print(f"💾 更新汇总文件: {summary_path}")
        print(f"📊 汇总包含 {len(final_results)} 个模型结果")

    except Exception as e:
        logging.error(f"Failed to save summary file: {e}")


def rebuild_summary_standalone():
    """🔧 独立运行的汇总文件重建函数"""
    print("=" * 60)
    print("🔄 重建模型结果汇总文件")
    print("=" * 60)

    results_dir = Path(data_path) / "results"
    summary_path = results_dir / "all_models_summary_with_baseline_new.json"

    # 检查 results 目录
    if not results_dir.exists():
        print(f"❌ Results 目录不存在: {results_dir}")
        return {}

    # 找到所有模型结果文件
    json_files = [
        f for f in results_dir.glob("*.json") if "summary" not in f.name.lower()
    ]

    if not json_files:
        print("❌ 未找到任何模型结果文件")
        return {}

    print(f"📁 找到 {len(json_files)} 个模型结果文件:")

    # 读取并重建汇总
    all_results = {}

    for json_file in json_files:
        try:
            print(f"📖 读取: {json_file.name}...")

            with open(json_file, "r", encoding="utf-8") as f:
                result_data = json.load(f)

            # 提取模型信息
            model_type = result_data.get("model_type", json_file.stem)
            model_name = result_data.get("model_name", "Unknown Model")
            test_rmse = result_data.get("test_metrics", {}).get("RMSE", "N/A")

            all_results[model_type] = result_data

            print(f"   ✅ {model_name} (RMSE: {test_rmse})")

        except Exception as e:
            print(f"   ❌ 读取失败: {e}")

    if not all_results:
        print("❌ 没有成功读取任何模型结果")
        return {}

    # 清空并重写汇总文件
    print(f"\n🗑️ 清空原有汇总文件...")
    if summary_path.exists():
        summary_path.unlink()
        print(f"   删除: {summary_path.name}")

    print(f"💾 重新写入汇总文件...")
    try:
        # 🔧 修复：转换 NumPy 类型
        json_compatible_results = convert_numpy_types(all_results)

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(json_compatible_results, f, indent=2, ensure_ascii=False)

        print(f"✅ 汇总文件重建成功!")
        print(f"📊 包含 {len(all_results)} 个模型结果")
        print(f"📁 文件位置: {summary_path}")

        # 显示汇总信息
        print(f"\n📋 模型汇总:")
        for model_type, result in all_results.items():
            model_name = result.get("model_name", model_type)
            test_rmse = result.get("test_metrics", {}).get("RMSE", "N/A")
            print(f"   - {model_name}: RMSE = {test_rmse}")

    except Exception as e:
        print(f"❌ 写入汇总文件失败: {e}")
        return {}

    print("=" * 60)
    return all_results


def main():
    """Main function: train and evaluate selected models"""

    # 新增：在开始前重建汇总文件
    print("🔄 更新模型结果汇总文件...")
    try:
        rebuild_summary_standalone()
        print("✅ 汇总文件更新完成")
    except Exception as e:
        print(f"⚠️ 汇总文件更新失败: {e}")
        print("📝 将继续执行训练流程...")

    print("\n" + "=" * 60)

    # 准备训练环境
    models_to_train_list = prepare_training_environment()

    if not models_to_train_list:
        print("🎉 没有需要训练的模型，退出。")
        print("💡 如需训练，请修改 MODELS_TO_TRAIN 配置")
        return {}

    # Setup logging - 清理旧日志文件
    log_file = data_path + "training_comparison_log.txt"
    log_path = Path(log_file)
    if log_path.exists():
        log_path.unlink()
        print(f"🗑️ 清理旧日志文件: {log_file}")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8", mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )

    # Record start information
    logging.info("=" * 80)
    logging.info("🆕 选择性模型训练流程开始")
    logging.info(f"📋 将要训练的模型: {models_to_train_list}")
    logging.info("=" * 80)

    # Configuration parameters
    config = {
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "K_FACTORS": 100,
        "TIME_FACTORS": 20,
        "BATCH_SIZE": 256,
        "NUM_EPOCHS": 30,
        "LEARNING_RATE": 0.001,
        "REG_STRENGTH": 0.001,
        "NUM_EXPERTS": 4,
        "NUM_EPOCHS_PER_STAGE": [10, 10, 10],
        "LEARNING_RATES": [0.001, 0.001, 0.0005],  # Use optimized MMOE learning rates
    }

    logging.info(f'🖥️ 使用设备: {config["DEVICE"]}')
    logging.info(f"⚙️ 配置参数: {config}")
    logging.info(f"📈 学习率调度器: ReduceLROnPlateau (factor=0.6, patience=5)")

    # 准备数据
    split_path = data_path + "split_data"

    if check_split_data_exists(split_path):
        logging.info("📂 加载现有数据分割...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
    else:
        logging.info("🔄 创建新数据分割...")
        ratings, users, movies = load_data(
            data_path + "ratings.csv", data_path + "users.csv", data_path + "movies.csv"
        )
        train_data, val_data, test_data = create_time_aware_split(
            ratings, random_state=42
        )
        save_split_data(train_data, val_data, test_data, split_path)

    logging.info(
        f"📊 数据分割 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}"
    )

    # Create data loaders
    train_dataset = MovieLensDataset(
        train_data["user_emb_id"].values,
        train_data["movie_emb_id"].values,
        train_data["rating"].values,
        train_data["daytime"].values,
        train_data["is_weekend"].values,
        train_data["year"].values,
    )

    val_dataset = MovieLensDataset(
        val_data["user_emb_id"].values,
        val_data["movie_emb_id"].values,
        val_data["rating"].values,
        val_data["daytime"].values,
        val_data["is_weekend"].values,
        val_data["year"].values,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=0
    )

    # Get max IDs
    max_userid = train_data["user_emb_id"].max()
    max_movieid = train_data["movie_emb_id"].max()

    # Store all results (only newly trained models)
    all_results = {}
    training_success_count = 0

    # Train models based on selection
    if "baseline" in models_to_train_list:
        try:
            logging.info(f"\n🔄 训练 Baseline 模型...")
            baseline_results = train_and_evaluate_baseline_model(
                "Baseline Collaborative Filtering",
                max_userid,
                max_movieid,
                train_data,
                val_data,
                test_data,
                config["DEVICE"],
                config,
            )
            all_results[baseline_results["model_type"]] = baseline_results
            training_success_count += 1
        except Exception as e:
            logging.error(f"❌ Baseline模型训练失败: {str(e)}")
            import traceback

            logging.error(f"详细错误信息: {traceback.format_exc()}")

    # Define time-aware models mapping
    file_mapping = get_model_file_mapping()
    time_aware_models_map = {
        "usertime": (UserTimeModel, file_mapping["usertime"]["display_name"]),
        "independent": (
            IndependentTimeModel,
            file_mapping["independent"]["display_name"],
        ),  # 修复名称
        "umtime": (UMTimeModel, file_mapping["umtime"]["display_name"]),
    }

    # Train selected time-aware models
    for model_key in ["usertime", "independent", "umtime"]:
        if model_key in models_to_train_list:
            try:
                model_class, correct_model_name = time_aware_models_map[model_key]
                logging.info(f"🎯 开始训练 {model_key}: {correct_model_name}")

                results = train_and_evaluate_model(
                    model_class,
                    correct_model_name,  # 使用正确的名称
                    max_userid,
                    max_movieid,
                    train_loader,
                    val_loader,
                    test_data,
                    config["DEVICE"],
                    config,
                )

                file_info = file_mapping[model_key]
                all_results[file_info["model_type"]] = results
                training_success_count += 1
                logging.info(f"✅ {correct_model_name} 训练成功")

            except Exception as e:
                logging.error(f"❌ {model_key} 训练失败: {str(e)}")
                logging.error(f"详细错误: {traceback.format_exc()}")
                continue

    # Train MMOE model
    if "mmoe" in models_to_train_list:
        try:
            logging.info(f"\n🔄 训练 MMOE 模型...")
            mmoe_results = train_and_evaluate_mmoe_model(
                "Two-Stage MMoE Model (Optimized)",
                max_userid,
                max_movieid,
                train_data,
                val_data,
                test_data,
                config["DEVICE"],
                config,
            )
            all_results[mmoe_results["model_type"]] = mmoe_results
            training_success_count += 1
        except Exception as e:
            logging.error(f"❌ MMOE模型训练失败: {str(e)}")
            import traceback

            logging.error(f"详细错误信息: {traceback.format_exc()}")

    # 更新汇总文件
    if all_results:
        update_summary_file(all_results)

    # 输出训练总结
    logging.info(f"\n{'='*60}")
    logging.info("🎉 选择性模型训练完成!")
    logging.info(
        f"📊 成功训练: {training_success_count}/{len(models_to_train_list)} 个模型"
    )
    logging.info("📋 训练结果摘要:")

    for model_type, results in all_results.items():
        test_metrics = results["test_metrics"]
        training_history = results["training_history"]

        # 显示学习率变化信息
        lr_info = ""
        if "learning_rates" in training_history and training_history["learning_rates"]:
            initial_lr = training_history["learning_rates"][0]
            final_lr = training_history["learning_rates"][-1]
            lr_reduction = (initial_lr - final_lr) / initial_lr * 100
            lr_info = f", LR: {initial_lr:.6f}→{final_lr:.6f}(-{lr_reduction:.1f}%)"

        logging.info(f"  ✅ {results['model_name']}:")
        logging.info(f"    📊 RMSE: {test_metrics['RMSE']:.4f}")
        logging.info(f"    📊 MAE: {test_metrics['MAE']:.4f}")
        logging.info(f"    ⏱️ 训练时间: {training_history['total_training_time']:.2f}s")
        logging.info(
            f"    🔢 参数量: {results['model_params']['total_params']:,}{lr_info}"
        )

    logging.info("=" * 80)
    logging.info("🎯 选择性训练流程完成!")

    if all_results:
        logging.info(f"📁 结果文件保存在: {data_path}results/")
        logging.info(f"🤖 模型文件保存在: {data_path}model/")
        logging.info(f"📝 日志文件: {log_file}")

    logging.info("=" * 80)

    return all_results


if __name__ == "__main__":
    main()
