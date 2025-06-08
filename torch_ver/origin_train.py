import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from origin_model import CFModel
# 导入数据处理函数（不导入split_data，因为它不存在）
from data_process import (load_data, save_split_data, check_split_data_exists, 
                         load_existing_split_data, MovieLensDataset, data_path,
                         create_time_aware_split)

from torch.utils.data import DataLoader
import logging
from pathlib import Path
import math
import pandas as pd
import time
import numpy as np
import json

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, patience=10):
    """训练模型，包含早停机制和正则化 - 增加详细记录"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    # 详细训练历史记录
    training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_rmse': [],
        'val_rmse': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_epoch': 0,
        'total_epochs': 0,
        'total_training_time': 0
    }
    
    logging.info(f"开始训练Baseline模型")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # 训练阶段
        model.train()
        train_loss = 0
        num_batches = 0
        
        for user, movie, rating in train_loader:
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, movie)
            
            # 计算总损失（MSE + L2正则化）
            mse_loss = criterion(prediction, rating)
            reg_loss = model.get_regularization_loss()
            total_loss = mse_loss + reg_loss
            
            total_loss.backward()
            
            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += mse_loss.item()  # 只记录MSE损失用于比较
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_rmse = math.sqrt(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for user, movie, rating in val_loader:
                user, movie, rating = user.to(device), movie.to(device), rating.to(device)
                prediction = model(user, movie)
                loss = criterion(prediction, rating)  # 验证时只用MSE
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches
        val_rmse = math.sqrt(avg_val_loss)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        
        # 学习率调度
        old_lr = current_lr
        if scheduler:
            scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 记录训练指标
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
        logging.info(f'Learning rate: {new_lr:.6f}')
        
        # 如果学习率发生变化，记录日志
        if new_lr != old_lr:
            logging.info(f'Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}')
        
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
    
    # 记录总的训练信息
    training_history['total_epochs'] = epoch + 1
    total_training_time = time.time() - start_time
    training_history['total_training_time'] = total_training_time
    
    # 计算最终的最佳RMSE
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f'训练完成！总时间: {total_training_time:.2f}s')
    logging.info(f'Best validation loss: {best_val_loss:.4f}')
    logging.info(f'Best validation RMSE: {best_rmse:.4f}')
    
    return best_model_state, training_history

def evaluate_baseline_model(model, test_data, device):
    """详细评估baseline模型性能"""
    model.eval()
    predictions = []
    actuals = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            movie_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            
            pred = model(user_id, movie_id)
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

def main():
    # 设置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(data_path + 'baseline_training_log.txt', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 配置参数 - 更接近TensorFlow版本，与其他模型保持一致
    config = {
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'K_FACTORS': 100,  # baseline使用更多因子
        'BATCH_SIZE': 256,
        'LEARNING_RATE': 0.001,
        'NUM_EPOCHS': 30,
        'REG_STRENGTH': 0.0001,  # baseline使用较低正则化
        'PATIENCE': 10
    }
    
    logging.info(f'使用设备: {config["DEVICE"]}')
    logging.info(f'Baseline模型配置: {config}')
    logging.info(f'学习率调度器: ReduceLROnPlateau (factor=0.5, patience=5)')
    
    # 设置分割数据路径
    split_path = data_path + 'split_data'
    
    # 检查是否已有分割数据 - 使用现有数据
    if check_split_data_exists(split_path):
        logging.info("Found existing split data, loading...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
        
        if train_data is not None and val_data is not None and test_data is not None:
            logging.info(f'Loaded existing split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
        else:
            logging.error("Failed to load existing split data!")
            return None, None, None
    else:
        logging.error("No existing split data found! Please run data splitting first.")
        return None, None, None
    
    # 为baseline模型添加嵌入ID（如果不存在）
    for data in [train_data, val_data, test_data]:
        if 'user_emb_id' not in data.columns:
            data['user_emb_id'] = data['user_id'] - 1  # 转换为0-based索引
        if 'movie_emb_id' not in data.columns:
            data['movie_emb_id'] = data['movie_id'] - 1  # 转换为0-based索引
    
    # 创建Baseline专用的数据加载器（只需要user, movie, rating）
    class BaselineDataset(torch.utils.data.Dataset):
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
                torch.FloatTensor([self.ratings[idx]])
            )
    
    train_dataset = BaselineDataset(
        train_data['user_emb_id'].values,
        train_data['movie_emb_id'].values,
        train_data['rating'].values
    )
    
    val_dataset = BaselineDataset(
        val_data['user_emb_id'].values,
        val_data['movie_emb_id'].values,
        val_data['rating'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    # 获取用户和电影的最大ID
    max_userid = train_data['user_emb_id'].max()
    max_movieid = train_data['movie_emb_id'].max()
    
    logging.info(f'Max user ID: {max_userid}, Max movie ID: {max_movieid}')
    
    model = CFModel(max_userid + 1, max_movieid + 1, config['K_FACTORS'], config['REG_STRENGTH']).to(config['DEVICE'])
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Baseline模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
    
    criterion = nn.MSELoss()
    
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=config['LEARNING_RATE'], weight_decay=1e-6)
    
    # 学习率调度器 - 与其他模型保持一致
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=5, min_lr=1e-6
    )
    
    # 训练模型
    best_model_state, training_history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        config['DEVICE'], config['NUM_EPOCHS'], patience=config['PATIENCE']
    )
    
    # 评估模型
    logging.info("开始评估Baseline模型...")
    test_metrics = evaluate_baseline_model(model, test_data, config['DEVICE'])
    
    # 整理结果
    results = {
        'model_name': 'Baseline Collaborative Filtering',
        'model_type': 'CFModel',
        'training_history': training_history,
        'test_metrics': test_metrics,
        'model_params': {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'k_factors': config['K_FACTORS'],
            'reg_strength': config['REG_STRENGTH']
        },
        'training_config': config
    }
    
    # 保存模型和训练历史
    checkpoint = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': config['K_FACTORS'],
        'reg_strength': config['REG_STRENGTH'],
        'best_model_state': best_model_state,
        'model_type': 'CFModel',
        'data_split_path': split_path,
        'training_history': training_history,
        'test_metrics': test_metrics,
        'has_scheduler': True
    }
    
    model_path = data_path + 'model_checkpoint_baseline_with_scheduler.pt'
    torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    logging.info(f'模型保存至: {model_path}')
    
    # 保存结果JSON
    results_path = data_path + 'results/results_baseline_with_scheduler.json'
    Path(results_path).parent.mkdir(exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"Baseline模型训练和评估完成!")
    logging.info(f"测试 RMSE: {test_metrics['RMSE']:.4f}")
    logging.info(f"测试 MAE: {test_metrics['MAE']:.4f}")
    logging.info(f"相关系数: {test_metrics['Correlation']:.4f}")
    
    return model, test_data, results

def save_checkpoint(model, optimizer, config, path):
    """安全保存模型检查点"""
    try:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **config
        }
        torch.save(save_dict, path)
        logging.info(f'Model saved to {path}')
    except Exception as e:
        logging.error(f"保存模型失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()