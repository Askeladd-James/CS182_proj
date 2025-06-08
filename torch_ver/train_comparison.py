import torch
import torch.nn as nn
import torch.optim as optim
from data_process import load_data, create_time_aware_split, save_split_data, check_split_data_exists, load_existing_split_data, MovieLensDataset, data_path
from model import IndependentTimeModel, UserTimeModel, UMTimeModel
from MMOE import TwoStageMMoEModel
from torch.utils.data import DataLoader
import logging
import math
import pandas as pd
import json
import time
import numpy as np
from pathlib import Path

def train_model_with_metrics(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, patience=10):
    """训练模型，记录详细的训练指标"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 记录训练指标
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_rmse': [],
        'val_rmse': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_epoch': 0,
        'total_epochs': 0
    }
    
    logging.info(f"开始训练模型: {model.name}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
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
            
            # 计算损失（MSE + L2正则化）
            mse_loss = criterion(prediction, rating)
            
            # 如果模型有正则化方法，添加正则化损失
            if hasattr(model, 'get_regularization_loss'):
                reg_loss = model.get_regularization_loss()
                total_loss = mse_loss + reg_loss
            else:
                total_loss = mse_loss
            
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += mse_loss.item()  # 只记录MSE用于比较
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_rmse = math.sqrt(avg_train_loss)
        
        # 验证阶段
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
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 学习率调度
        if scheduler:
            scheduler.step(avg_val_loss)
        
        # 记录指标
        training_history['train_losses'].append(avg_train_loss)
        training_history['val_losses'].append(avg_val_loss)
        training_history['train_rmse'].append(train_rmse)
        training_history['val_rmse'].append(val_rmse)
        training_history['learning_rates'].append(current_lr)
        
        epoch_time = time.time() - epoch_start
        training_history['epoch_times'].append(epoch_time)
        
        # 打印训练信息
        logging.info(f'Epoch {epoch+1}/{num_epochs} ({epoch_time:.2f}s)')
        logging.info(f'Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}')
        logging.info(f'Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        logging.info(f'Learning rate: {current_lr:.6f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            training_history['best_epoch'] = epoch
            logging.info(f'New best validation loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch+1}')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    training_history['total_epochs'] = epoch + 1
    total_time = time.time() - start_time
    training_history['total_training_time'] = total_time
    
    # 最终统计
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f'训练完成！总时间: {total_time:.2f}s')
    logging.info(f'Best validation loss: {best_val_loss:.4f}')
    logging.info(f'Best validation RMSE: {best_rmse:.4f}')
    
    return best_model_state, training_history

def evaluate_model_detailed(model, test_data, device):
    """详细评估模型性能"""
    model.eval()
    predictions = []
    actuals = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            movie_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            daytime = torch.LongTensor([row['daytime']]).to(device)
            weekend = torch.LongTensor([row['is_weekend']]).to(device)
            year = torch.LongTensor([row['year']]).to(device)
            
            pred = model(user_id, movie_id, daytime, weekend, year)
            predictions.append(pred.cpu().item())
            actuals.append(row['rating'])
    
    inference_time = time.time() - start_time
    
    # 计算各种评估指标
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 基本指标
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # 其他指标
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # 平均绝对百分比误差
    
    # 评分分布统计
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)
    
    # 相关系数
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    # 按评分区间的准确度
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f'rating_{rating}_mae'] = rating_mae
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Correlation': correlation,
        'Inference_Time': inference_time,
        'Predictions_Mean': pred_mean,
        'Predictions_Std': pred_std,
        'Actuals_Mean': actual_mean,
        'Actuals_Std': actual_std,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist(),
        **rating_accuracy
    }

def train_and_evaluate_model(model_class, model_name, max_userid, max_movieid, train_loader, val_loader, test_data, device, config):
    """训练并评估单个模型"""
    logging.info(f"\n{'='*60}")
    logging.info(f"开始训练和评估模型: {model_name}")
    logging.info(f"{'='*60}")
    
    # 创建模型
    model = model_class(
        max_userid + 1, 
        max_movieid + 1, 
        config['K_FACTORS'], 
        config['TIME_FACTORS'], 
        config['REG_STRENGTH']
    ).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
    
    # 设置优化器和调度器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 训练模型
    best_model_state, training_history = train_model_with_metrics(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        device, config['NUM_EPOCHS'], patience=10
    )
    
    # 评估模型
    logging.info(f"开始评估模型: {model_name}")
    test_metrics = evaluate_model_detailed(model, test_data, device)
    
    # 保存模型和结果
    results = {
        'model_name': model_name,
        'model_type': model.name,
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_params': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'k_factors': config['K_FACTORS'],
            'time_factors': config['TIME_FACTORS'],
            'reg_strength': config['REG_STRENGTH']
        },
        'training_config': config
    }
    
    # 保存模型检查点
    checkpoint = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': config['K_FACTORS'],
        'time_factors': config['TIME_FACTORS'],
        'reg_strength': config['REG_STRENGTH'],
        'best_model_state': best_model_state,
        'model_type': model.name,
        'training_history': training_history,
        'test_metrics': test_metrics
    }
    
    model_path = data_path + f'model/model_checkpoint_{model.name}.pt'
    torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    logging.info(f'模型保存至: {model_path}')
    
    # 保存结果JSON
    results_path = data_path + f'results/results_{model.name}.json'
    Path(results_path).parent.mkdir(exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"模型 {model_name} 训练和评估完成!")
    logging.info(f"测试 RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"测试 MAE: {test_metrics['MAE']:.4f}")
    
    return results


def train_and_evaluate_mmoe_model(model_name, max_userid, max_movieid, train_data, val_data, test_data, device, config):
    """专门为MMOE模型的训练和评估函数"""
    logging.info(f"\n{'='*60}")
    logging.info(f"开始训练和评估MMOE模型: {model_name}")
    logging.info(f"{'='*60}")
    
    # 创建MMOE模型
    model = TwoStageMMoEModel(
        max_userid + 1, 
        max_movieid + 1, 
        config['K_FACTORS'], 
        config['TIME_FACTORS'], 
        config['REG_STRENGTH'], 
        config['NUM_EXPERTS']
    ).to(device)
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
    
    # 训练历史记录
    all_training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_rmse': [],
        'val_rmse': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_epoch': 0,
        'total_epochs': 0
    }
    
    criterion = nn.MSELoss()
    start_time = time.time()
    
    # 使用MMOE的训练函数
    from MMOE_train import train_optimized_mmoe
    
    try:
        model, mmoe_history = train_optimized_mmoe(
            model, train_data, val_data, device,
            batch_size=config['BATCH_SIZE'],
            num_epochs_per_stage=[5, 8, 5],  # 适应比较框架的训练轮数
            learning_rates=[0.001, 0.001, 0.0005]
        )
        
        # 整合训练历史
        total_training_time = time.time() - start_time
        
        # 合并所有阶段的历史
        for stage_name, stage_history in mmoe_history.items():
            all_training_history['train_losses'].extend(stage_history['train_losses'])
            all_training_history['val_losses'].extend(stage_history['val_losses'])
            # 计算RMSE
            stage_train_rmse = [math.sqrt(loss) for loss in stage_history['train_losses']]
            stage_val_rmse = [math.sqrt(loss) for loss in stage_history['val_losses']]
            all_training_history['train_rmse'].extend(stage_train_rmse)
            all_training_history['val_rmse'].extend(stage_val_rmse)
            all_training_history['epoch_times'].extend(stage_history.get('epoch_times', [1.0] * stage_history['total_epochs']))
        
        # 设置其他必要字段
        all_training_history['total_training_time'] = total_training_time
        all_training_history['total_epochs'] = sum(stage_history['total_epochs'] for stage_history in mmoe_history.values())
        all_training_history['best_epoch'] = len(all_training_history['train_losses']) - 1
        
        # 填充学习率（简化处理）
        total_epochs = all_training_history['total_epochs']
        all_training_history['learning_rates'] = [0.001] * total_epochs
        
    except Exception as e:
        logging.error(f"MMOE训练过程中出错: {str(e)}")
        raise
    
    # 评估模型 - 设置为最终阶段进行评估
    model.set_training_stage(4)  # 解冻所有参数用于评估
    logging.info(f"开始评估MMOE模型: {model_name}")
    test_metrics = evaluate_mmoe_model(model, test_data, device)
    
    # 整理结果
    results = {
        'model_name': model_name,
        'model_type': model.name,
        'training_history': all_training_history,
        'test_metrics': test_metrics,
        'model_params': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'k_factors': config['K_FACTORS'],
            'time_factors': config['TIME_FACTORS'],
            'reg_strength': config['REG_STRENGTH'],
            'num_experts': config['NUM_EXPERTS']
        },
        'training_config': config
    }
    
    # 保存模型检查点
    checkpoint = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': config['K_FACTORS'],
        'time_factors': config['TIME_FACTORS'],
        'reg_strength': config['REG_STRENGTH'],
        'num_experts': config['NUM_EXPERTS'],
        'best_model_state': model.state_dict(),
        'model_type': model.name,
        'training_history': all_training_history,
        'test_metrics': test_metrics
    }
    
    model_path = data_path + f'model/model_checkpoint_{model.name}.pt'
    torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    logging.info(f'模型保存至: {model_path}')
    
    # 保存结果JSON
    results_path = data_path + f'results/results_{model.name}.json'
    Path(results_path).parent.mkdir(exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"MMOE模型 {model_name} 训练和评估完成!")
    logging.info(f"测试 RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"测试 MAE: {test_metrics['MAE']:.4f}")
    
    return results

def evaluate_mmoe_model(model, test_data, device):
    """专门为MMOE模型的评估函数"""
    model.eval()
    predictions = []
    actuals = []
    
    start_time = time.time()
    
    # 准备用户历史统计特征（简化版本，用于评估）
    from MMOE_train import prepare_user_history_stats
    user_history_stats = prepare_user_history_stats(test_data)
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            movie_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            daytime = torch.LongTensor([row['daytime']]).to(device)
            weekend = torch.LongTensor([row['is_weekend']]).to(device)
            year = torch.LongTensor([row['year']]).to(device)
            
            # 获取用户历史特征
            user_emb_id = row['user_emb_id']
            if user_emb_id in user_history_stats:
                history_features = torch.FloatTensor(user_history_stats[user_emb_id]).unsqueeze(0).to(device)
            else:
                history_features = torch.FloatTensor([3.0, 1.0, 1.0, 3.0, 0.0]).unsqueeze(0).to(device)
            
            # 获取各阶段预测
            model.set_training_stage(1)
            temporal_pred = model(user_id, movie_id, daytime, weekend, year, history_features)
            
            model.set_training_stage(2)
            cf_pred = model(user_id, movie_id, daytime, weekend, year)
            
            model.set_training_stage(3)
            final_pred = model(user_id, movie_id, daytime, weekend, year, temporal_pred, cf_pred)
            
            predictions.append(final_pred.cpu().item())
            actuals.append(row['rating'])
    
    inference_time = time.time() - start_time
    
    # 计算各种评估指标（与原函数相同）
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # 基本指标
    mse = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - actuals))
    
    # 其他指标
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100  # 平均绝对百分比误差
    
    # 评分分布统计
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    actual_mean = np.mean(actuals)
    actual_std = np.std(actuals)
    
    # 相关系数
    correlation = np.corrcoef(predictions, actuals)[0, 1]
    
    # 按评分区间的准确度
    rating_accuracy = {}
    for rating in [1, 2, 3, 4, 5]:
        mask = actuals == rating
        if np.sum(mask) > 0:
            rating_predictions = predictions[mask]
            rating_mae = np.mean(np.abs(rating_predictions - rating))
            rating_accuracy[f'rating_{rating}_mae'] = rating_mae
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Correlation': correlation,
        'Inference_Time': inference_time,
        'Predictions_Mean': pred_mean,
        'Predictions_Std': pred_std,
        'Actuals_Mean': actual_mean,
        'Actuals_Std': actual_std,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist(),
        **rating_accuracy
    }

def main():
    """主函数：训练和评估所有模型"""
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(data_path + 'training_log.txt', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 配置参数
    config = {
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'K_FACTORS': 80,
        'TIME_FACTORS': 20,
        'BATCH_SIZE': 256,
        'LEARNING_RATE': 0.001,
        'NUM_EPOCHS': 30,
        'REG_STRENGTH': 0.001,
        'NUM_EXPERTS': 4  # 添加MMOE专家数量
    }
    
    logging.info(f'使用设备: {config["DEVICE"]}')
    logging.info(f'配置参数: {config}')
    
    # 准备数据
    split_path = data_path + 'split_data'
    
    if check_split_data_exists(split_path):
        logging.info("加载现有数据分割...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
    else:
        logging.info("创建新的数据分割...")
        ratings, users, movies = load_data(
            data_path + 'ratings.csv',
            data_path + 'users.csv',
            data_path + 'movies.csv'
        )
        train_data, val_data, test_data = create_time_aware_split(ratings, random_state=42)
        save_split_data(train_data, val_data, test_data, split_path)
    
    logging.info(f'数据分割 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}')
    
    # 创建数据加载器（用于非MMOE模型）
    train_dataset = MovieLensDataset(
        train_data['user_emb_id'].values, train_data['movie_emb_id'].values,
        train_data['rating'].values, train_data['daytime'].values,
        train_data['is_weekend'].values, train_data['year'].values
    )
    
    val_dataset = MovieLensDataset(
        val_data['user_emb_id'].values, val_data['movie_emb_id'].values,
        val_data['rating'].values, val_data['daytime'].values,
        val_data['is_weekend'].values, val_data['year'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    # 获取最大ID
    max_userid = train_data['user_emb_id'].max()
    max_movieid = train_data['movie_emb_id'].max()
    
    # 定义要训练的模型（包括MMOE）
    models_to_train = [
        (UserTimeModel, "基础协同过滤模型"),
        (IndependentTimeModel, "简化时间感知模型"),
        (UMTimeModel, "复杂时间感知模型"),
        # MMOE模型使用特殊处理
    ]
    
    # 存储所有结果
    all_results = {}
    
    # 训练传统模型
    for model_class, model_name in models_to_train:
        try:
            results = train_and_evaluate_model(
                model_class, model_name, max_userid, max_movieid,
                train_loader, val_loader, test_data, config['DEVICE'], config
            )
            all_results[results['model_type']] = results
        except Exception as e:
            logging.error(f"训练模型 {model_name} 时出错: {str(e)}")
            continue
    
    # 训练MMOE模型
    try:
        mmoe_results = train_and_evaluate_mmoe_model(
            "优化的两阶段MMoE模型", max_userid, max_movieid,
            train_data, val_data, test_data, config['DEVICE'], config
        )
        all_results[mmoe_results['model_type']] = mmoe_results
    except Exception as e:
        logging.error(f"训练MMOE模型时出错: {str(e)}")
    
    # 保存所有结果的汇总
    summary_path = data_path + 'results/all_models_summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"\n{'='*60}")
    logging.info("所有模型训练完成!")
    logging.info("结果汇总:")
    
    for model_type, results in all_results.items():
        test_metrics = results['test_metrics']
        logging.info(f"{results['model_name']}:")
        logging.info(f"  RMSE: {test_metrics['RMSE']:.4f}")
        logging.info(f"  MAE: {test_metrics['MAE']:.4f}")
        logging.info(f"  训练时间: {results['training_history']['total_training_time']:.2f}s")
        logging.info(f"  参数数量: {results['model_params']['total_params']:,}")
    
    return all_results

if __name__ == "__main__":
    main()