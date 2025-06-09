import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from data_process import (load_data, create_time_aware_split, save_split_data, 
                         check_split_data_exists, load_existing_split_data, 
                         MovieLensDataset, data_path)
from MMOE import TwoStageMMoEModel
from torch.utils.data import DataLoader, Dataset
import logging
import pandas as pd
import numpy as np
import time
import math
from pathlib import Path

class OptimizedTemporalDataset(Dataset):
    """优化的时序训练数据集 - 使用统计特征而不是完整序列"""
    def __init__(self, data, user_history_stats):
        self.data = data.reset_index(drop=True)
        self.user_history_stats = user_history_stats
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        user_id = row['user_emb_id']
        
        # 获取用户历史统计特征
        if user_id in self.user_history_stats:
            history_features = torch.FloatTensor(self.user_history_stats[user_id])
        else:
            # 默认历史特征
            history_features = torch.FloatTensor([3.0, 1.0, 1.0, 3.0, 0.0])
        
        return (
            torch.LongTensor([row['user_emb_id']]),
            torch.LongTensor([row['movie_emb_id']]),
            torch.FloatTensor([row['rating']]),
            torch.LongTensor([row['daytime']]),
            torch.LongTensor([row['is_weekend']]),
            torch.LongTensor([row['year']]),
            history_features
        )

class FusionDataset(Dataset):
    """融合训练数据集"""
    def __init__(self, data, temporal_predictions, cf_predictions):
        self.data = data.reset_index(drop=True)
        self.temporal_predictions = temporal_predictions
        self.cf_predictions = cf_predictions
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return (
            torch.LongTensor([row['user_emb_id']]),
            torch.LongTensor([row['movie_emb_id']]),
            torch.FloatTensor([row['rating']]),
            torch.LongTensor([row['daytime']]),
            torch.LongTensor([row['is_weekend']]),
            torch.LongTensor([row['year']]),
            torch.FloatTensor([self.temporal_predictions[idx]]),
            torch.FloatTensor([self.cf_predictions[idx]])
        )

def prepare_user_history_stats(train_data):
    """改进的用户历史统计特征准备"""
    user_stats = {}
    
    # 按用户分组计算统计特征
    for user_id in train_data['user_emb_id'].unique():
        user_data = train_data[train_data['user_emb_id'] == user_id].copy()
        user_data = user_data.sort_values('timestamp')
        
        ratings = user_data['rating'].values
        
        if len(ratings) > 0:
            # 计算更丰富的统计特征
            avg_rating = np.mean(ratings)
            std_rating = np.std(ratings) if len(ratings) > 1 else 0.1  # 避免0方差
            num_ratings = min(len(ratings), 200)  # 限制最大值避免特征过大
            recent_rating = ratings[-1] if len(ratings) > 0 else avg_rating
            
            # 改进的评分趋势计算
            if len(ratings) >= 3:
                # 使用加权平均，更重视最近的评分
                weights = np.exp(np.linspace(-1, 0, len(ratings)))
                weighted_avg = np.average(ratings, weights=weights)
                trend = (weighted_avg - avg_rating) / max(std_rating, 0.1)
            else:
                trend = 0.0
            
            # 归一化特征
            user_stats[user_id] = [
                avg_rating,                    # 平均评分 [1-5]
                min(std_rating, 2.0),         # 标准差限制在合理范围
                np.log1p(num_ratings) / 5.0,  # 对数变换的评分数量
                recent_rating,                # 最近评分 [1-5]
                np.tanh(trend)                # tanh限制趋势在[-1,1]
            ]
        else:
            user_stats[user_id] = [3.0, 0.5, 0.1, 3.0, 0.0]
    
    return user_stats

def create_temporal_dataloader(data, user_history_stats, batch_size, shuffle=True):
    """创建时序数据加载器"""
    dataset = OptimizedTemporalDataset(data, user_history_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def create_standard_dataloader(data, batch_size, shuffle=True):
    """创建标准数据加载器"""
    dataset = MovieLensDataset(
        data['user_emb_id'].values,
        data['movie_emb_id'].values,
        data['rating'].values,
        data['daytime'].values,
        data['is_weekend'].values,
        data['year'].values
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_predictions_for_data(model, data, user_history_stats, device, stage):
    """为数据获取预测结果"""
    model.eval()
    model.set_training_stage(stage)
    
    predictions = []
    
    with torch.no_grad():
        for _, row in data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            item_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            daytime = torch.LongTensor([row['daytime']]).to(device)
            weekend = torch.LongTensor([row['is_weekend']]).to(device)
            year = torch.LongTensor([row['year']]).to(device)
            
            if stage == 1:  # 时序预测
                user_emb_id = row['user_emb_id']
                if user_emb_id in user_history_stats:
                    history_features = torch.FloatTensor(user_history_stats[user_emb_id]).unsqueeze(0).to(device)
                else:
                    history_features = torch.FloatTensor([3.0, 1.0, 1.0, 3.0, 0.0]).unsqueeze(0).to(device)
                
                pred = model(user_id, item_id, daytime, weekend, year, history_features)
            else:  # CF预测
                pred = model(user_id, item_id, daytime, weekend, year)
            
            predictions.append(pred.item())
    
    return predictions

def get_predictions_for_data_batch(model, data, user_history_stats, device, stage, batch_size=1024):
    """批量获取预测结果 - 优化版本"""
    model.eval()
    model.set_training_stage(stage)
    
    predictions = []
    
    # 创建数据加载器进行批处理
    if stage == 1:
        # 时序预测需要用户历史特征
        dataset = OptimizedTemporalDataset(data, user_history_stats)
    else:
        # CF预测使用标准数据集
        dataset = MovieLensDataset(
            data['user_emb_id'].values,
            data['movie_emb_id'].values,
            data['rating'].values,  # 这里的rating不会被使用，只是为了保持接口一致
            data['daytime'].values,
            data['is_weekend'].values,
            data['year'].values
        )
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in dataloader:
            if stage == 1:
                # 时序预测
                users, items, _, daytime, weekend, years, history_features = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                history_features = history_features.to(device)
                
                batch_preds = model(users, items, daytime, weekend, years, history_features)
            else:
                # CF预测
                users, items, _, daytime, weekend, years = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                
                batch_preds = model(users, items, daytime, weekend, years)
            
            # 处理单个值和批量值的情况
            if batch_preds.dim() == 0:
                predictions.append(batch_preds.item())
            else:
                predictions.extend(batch_preds.cpu().numpy().tolist())
    
    return predictions

def create_cached_fusion_dataloader(model, data, user_history_stats, batch_size, device, shuffle=True, cache_file=None):
    """创建带缓存的融合数据加载器"""
    
    if cache_file and Path(cache_file).exists():
        print(f"  从缓存加载预测结果: {cache_file}")
        cache_data = torch.load(cache_file)
        temporal_predictions = cache_data['temporal_predictions']
        cf_predictions = cache_data['cf_predictions']
    else:
        print("  生成并缓存预测结果...")
        temporal_predictions = get_predictions_for_data_batch(model, data, user_history_stats, device, stage=1, batch_size=2048)
        cf_predictions = get_predictions_for_data_batch(model, data, user_history_stats, device, stage=2, batch_size=2048)
        
        if cache_file:
            cache_data = {
                'temporal_predictions': temporal_predictions,
                'cf_predictions': cf_predictions
            }
            torch.save(cache_data, cache_file)
            print(f"  预测结果已缓存至: {cache_file}")
    
    dataset = FusionDataset(data, temporal_predictions, cf_predictions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_stage(model, train_loader, val_loader, criterion, optimizer, device, 
                         num_epochs, stage_name, patience=5, scheduler=None, stage_num=1):
    """优化的训练阶段函数"""
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # 根据不同阶段调整梯度裁剪
    max_grad_norms = {1: 0.5, 2: 1.0, 3: 1.5}  # 时序阶段用更小的梯度裁剪
    max_grad_norm = max_grad_norms.get(stage_num, 1.0)
    
    print(f"开始 {stage_name} 训练，最大梯度裁剪: {max_grad_norm}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        reg_loss_total = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            try:
                if model.training_stage == 1:
                    # 时序训练 - 需要7个元素（包括history_features）
                    users, items, ratings, daytime, weekend, years, history_features = batch
                    users = users.squeeze().to(device)
                    items = items.squeeze().to(device)
                    ratings = ratings.squeeze().to(device)
                    daytime = daytime.squeeze().to(device)
                    weekend = weekend.squeeze().to(device)
                    years = years.squeeze().to(device)
                    history_features = history_features.to(device)
                    
                    predictions = model(users, items, daytime, weekend, years, history_features)
                    targets = ratings
                    
                elif model.training_stage == 2:
                    # CF训练
                    users, items, ratings, daytime, weekend, years, *extra_features = batch
                    users = users.squeeze().to(device)
                    items = items.squeeze().to(device)
                    ratings = ratings.squeeze().to(device)
                    daytime = daytime.squeeze().to(device)
                    weekend = weekend.squeeze().to(device)
                    years = years.squeeze().to(device)
                    
                    predictions = model(users, items, daytime, weekend, years)
                    targets = ratings
                    
                elif model.training_stage == 3:
                    # MMoE训练
                    users, items, ratings, daytime, weekend, years, *extra_features = batch
                    users = users.squeeze().to(device)
                    items = items.squeeze().to(device)
                    ratings = ratings.squeeze().to(device)
                    daytime = daytime.squeeze().to(device)
                    weekend = weekend.squeeze().to(device)
                    years = years.squeeze().to(device)
                    
                    if len(extra_features) == 2:
                        # FusionDataset
                        temporal_preds, cf_preds = extra_features
                        temporal_preds = temporal_preds.squeeze().to(device)
                        cf_preds = cf_preds.squeeze().to(device)
                        predictions = model(users, items, daytime, weekend, years, temporal_preds, cf_preds)
                    else:
                        predictions = model(users, items, daytime, weekend, years)
                    
                    targets = ratings
                
                # 计算损失
                mse_loss = criterion(predictions, targets)
                reg_loss = model.get_regularization_loss()
                total_loss_batch = mse_loss + reg_loss
                
                # 反向传播
                total_loss_batch.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                
                optimizer.step()
                
                total_loss += mse_loss.item()
                reg_loss_total += reg_loss.item()
                num_batches += 1
                
                # 每100个batch打印一次进度（仅在第一个epoch）
                if epoch == 0 and batch_idx % 100 == 0:
                    print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {mse_loss.item():.4f}")
                    
            except Exception as e:
                print(f"训练批次错误: {e}")
                continue
        
        if num_batches == 0:
            print(f"警告: {stage_name} Epoch {epoch+1} 没有成功处理任何批次")
            continue
            
        avg_train_loss = total_loss / num_batches
        avg_reg_loss = reg_loss_total / num_batches
        train_losses.append(avg_train_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 验证阶段
        if val_loader is not None:
            val_loss = evaluate_stage(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Reg Loss: {avg_reg_loss:.4f}, LR: {current_lr:.6f}")
            
            # 更新学习率调度器
            old_lr = current_lr
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                elif isinstance(scheduler, CosineAnnealingWarmRestarts):
                    scheduler.step()
                else:
                    scheduler.step()
                
                # 检查学习率是否改变
                new_lr = optimizer.param_groups[0]['lr']
                if abs(new_lr - old_lr) > 1e-8:
                    print(f"  → 学习率调整: {old_lr:.6f} → {new_lr:.6f}")
            
            # 早停检查
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                print(f"  → 新的最佳验证损失: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"  → 验证损失未改善 ({patience_counter}/{patience})")
                
            # 早停
            if patience_counter >= patience:
                print(f"  → 早停：验证损失连续 {patience} 轮未改善")
                break
        else:
            print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Reg Loss: {avg_reg_loss:.4f}, LR: {current_lr:.6f}")
                    
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_model_state = model.state_dict().copy()
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  → 已恢复到最佳模型状态 (最佳损失: {best_loss:.4f})")
    
    # 返回训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses if val_loader is not None else [],
        'learning_rates': learning_rates,
        'best_loss': best_loss,
        'total_epochs': epoch + 1
    }
    
    return best_loss, training_history

def evaluate_stage(model, val_loader, criterion, device):
    """评估阶段性能 - 修复所有阶段的batch解包问题"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if model.training_stage == 1:
                # 时序验证 - 7个元素
                users, items, ratings, daytime, weekend, years, history_features = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                ratings = ratings.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                history_features = history_features.to(device)
                
                predictions = model(users, items, daytime, weekend, years, history_features)
                targets = ratings
                
            elif model.training_stage == 2:
                # CF验证 - 使用*extra_features处理可能的变长参数
                users, items, ratings, daytime, weekend, years, *extra_features = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                ratings = ratings.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                
                predictions = model(users, items, daytime, weekend, years)
                targets = ratings
                
            elif model.training_stage == 3:
                # MMoE验证 - 使用*extra_features处理可变数量的元素
                users, items, ratings, daytime, weekend, years, *extra_features = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                ratings = ratings.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                
                if len(extra_features) == 2:
                    # FusionDataset - 有temporal_preds和cf_preds
                    temporal_preds, cf_preds = extra_features
                    temporal_preds = temporal_preds.squeeze().to(device)
                    cf_preds = cf_preds.squeeze().to(device)
                    predictions = model(users, items, daytime, weekend, years, temporal_preds, cf_preds)
                else:
                    # 标准数据集 - 没有额外特征
                    predictions = model(users, items, daytime, weekend, years)
                
                targets = ratings
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_mmoe(model, train_data, val_data, device, batch_size=256, 
               num_epochs_per_stage=[40, 50, 40], learning_rates=[0.001, 0.002, 0.0005]):
    """修复的MMoE训练函数 - 正确记录训练时间"""
    
    print("准备用户历史统计特征...")
    user_history_stats = prepare_user_history_stats(train_data)
    val_user_history_stats = prepare_user_history_stats(val_data)
    
    criterion = nn.MSELoss()
    
    # 🔧 修复：创建统一的训练历史记录
    unified_training_history = {
        'train_losses': [],
        'val_losses': [],
        'train_rmse': [],
        'val_rmse': [],
        'learning_rates': [],
        'epoch_times': [],
        'best_epoch': 0,
        'total_epochs': 0,
        'total_training_time': 0.0,
        'stage_info': []  # 记录每个阶段的信息
    }
    
    # 🔧 记录总的开始时间
    total_start_time = time.time()
    
    # 阶段1：时序建模
    print("=" * 50)
    print("阶段1：时序建模")
    print("=" * 50)
    
    stage1_start = time.time()
    model.set_training_stage(1)
    temporal_loader = create_temporal_dataloader(train_data, user_history_stats, batch_size)
    temporal_val_loader = create_temporal_dataloader(val_data, val_user_history_stats, batch_size, shuffle=False)
    
    optimizer1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rates[0], 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer1, mode='min', factor=0.7, patience=5, min_lr=1e-7
    )
    
    best_temporal_loss, temporal_history = train_stage(
        model, temporal_loader, temporal_val_loader, criterion, optimizer1, 
        device, num_epochs_per_stage[0], "Temporal", patience=5, scheduler=scheduler1, stage_num=1
    )
    
    stage1_time = time.time() - stage1_start
    
    # 🔧 合并阶段1的历史到统一历史
    unified_training_history['train_losses'].extend(temporal_history['train_losses'])
    unified_training_history['val_losses'].extend(temporal_history['val_losses'])
    if 'learning_rates' in temporal_history:
        unified_training_history['learning_rates'].extend(temporal_history['learning_rates'])
    if 'epoch_times' in temporal_history:
        unified_training_history['epoch_times'].extend(temporal_history['epoch_times'])
    
    # 计算RMSE
    temporal_rmse = [math.sqrt(loss) for loss in temporal_history['train_losses']]
    temporal_val_rmse = [math.sqrt(loss) for loss in temporal_history['val_losses']]
    unified_training_history['train_rmse'].extend(temporal_rmse)
    unified_training_history['val_rmse'].extend(temporal_val_rmse)
    
    # 记录阶段信息
    unified_training_history['stage_info'].append({
        'stage_name': 'Temporal',
        'start_epoch': 0,
        'end_epoch': temporal_history['total_epochs'] - 1,
        'epochs': temporal_history['total_epochs'],
        'best_loss': best_temporal_loss,
        'training_time': stage1_time
    })
    
    print(f"阶段1完成 - 训练时间: {stage1_time:.2f}s, 最佳损失: {best_temporal_loss:.4f}")
    
    # 阶段2：CF建模
    print("=" * 50)
    print("阶段2：协同过滤建模")
    print("=" * 50)
    
    stage2_start = time.time()
    model.set_training_stage(2)
    cf_loader = create_standard_dataloader(train_data, batch_size)
    val_loader = create_standard_dataloader(val_data, batch_size, shuffle=False)
    
    optimizer2 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rates[1], 
        weight_decay=1e-5,
        betas=(0.9, 0.999)
    )
    
    scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer2, mode='min', factor=0.6, patience=5, min_lr=1e-7
    )
    
    best_cf_loss, cf_history = train_stage(
        model, cf_loader, val_loader, criterion, optimizer2, 
        device, num_epochs_per_stage[1], "CF", patience=5, scheduler=scheduler2, stage_num=2
    )
    
    stage2_time = time.time() - stage2_start
    
    # 🔧 合并阶段2的历史
    current_epoch_offset = len(unified_training_history['train_losses'])
    unified_training_history['train_losses'].extend(cf_history['train_losses'])
    unified_training_history['val_losses'].extend(cf_history['val_losses'])
    if 'learning_rates' in cf_history:
        unified_training_history['learning_rates'].extend(cf_history['learning_rates'])
    if 'epoch_times' in cf_history:
        unified_training_history['epoch_times'].extend(cf_history['epoch_times'])
    
    # 计算RMSE
    cf_rmse = [math.sqrt(loss) for loss in cf_history['train_losses']]
    cf_val_rmse = [math.sqrt(loss) for loss in cf_history['val_losses']]
    unified_training_history['train_rmse'].extend(cf_rmse)
    unified_training_history['val_rmse'].extend(cf_val_rmse)
    
    # 记录阶段信息
    unified_training_history['stage_info'].append({
        'stage_name': 'CF',
        'start_epoch': current_epoch_offset,
        'end_epoch': current_epoch_offset + cf_history['total_epochs'] - 1,
        'epochs': cf_history['total_epochs'],
        'best_loss': best_cf_loss,
        'training_time': stage2_time
    })
    
    print(f"阶段2完成 - 训练时间: {stage2_time:.2f}s, 最佳损失: {best_cf_loss:.4f}")
    
    # 阶段3：MMoE融合
    print("=" * 50)
    print("阶段3：MMoE融合")
    print("=" * 50)
    
    stage3_start = time.time()
    model.set_training_stage(3)
    
    # 清除旧缓存
    train_cache_file = data_path + 'cache_train_predictions.pt'
    val_cache_file = data_path + 'cache_val_predictions.pt'
    
    for cache_file in [train_cache_file, val_cache_file]:
        if Path(cache_file).exists():
            Path(cache_file).unlink()
            print(f"  清除缓存: {cache_file}")
    
    print("准备融合训练数据...")
    fusion_loader = create_cached_fusion_dataloader(
        model, train_data, user_history_stats, batch_size, device, 
        cache_file=train_cache_file
    )
    
    print("准备融合验证数据...")
    fusion_val_loader = create_cached_fusion_dataloader(
        model, val_data, val_user_history_stats, batch_size, device, 
        shuffle=False, cache_file=val_cache_file
    )
    
    optimizer3 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rates[2], 
        weight_decay=1e-6,
        betas=(0.9, 0.999)
    )
    
    scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer3, mode='min', factor=0.8, patience=5, min_lr=1e-8
    )
    
    best_mmoe_loss, mmoe_history = train_stage(
        model, fusion_loader, fusion_val_loader, criterion, optimizer3, 
        device, num_epochs_per_stage[2], "MMoE", patience=5, scheduler=scheduler3, stage_num=3
    )
    
    stage3_time = time.time() - stage3_start
    
    # 🔧 合并阶段3的历史
    current_epoch_offset = len(unified_training_history['train_losses'])
    unified_training_history['train_losses'].extend(mmoe_history['train_losses'])
    unified_training_history['val_losses'].extend(mmoe_history['val_losses'])
    if 'learning_rates' in mmoe_history:
        unified_training_history['learning_rates'].extend(mmoe_history['learning_rates'])
    if 'epoch_times' in mmoe_history:
        unified_training_history['epoch_times'].extend(mmoe_history['epoch_times'])
    
    # 计算RMSE
    mmoe_rmse = [math.sqrt(loss) for loss in mmoe_history['train_losses']]
    mmoe_val_rmse = [math.sqrt(loss) for loss in mmoe_history['val_losses']]
    unified_training_history['train_rmse'].extend(mmoe_rmse)
    unified_training_history['val_rmse'].extend(mmoe_val_rmse)
    
    # 记录阶段信息
    unified_training_history['stage_info'].append({
        'stage_name': 'MMoE',
        'start_epoch': current_epoch_offset,
        'end_epoch': current_epoch_offset + mmoe_history['total_epochs'] - 1,
        'epochs': mmoe_history['total_epochs'],
        'best_loss': best_mmoe_loss,
        'training_time': stage3_time
    })
    
    print(f"阶段3完成 - 训练时间: {stage3_time:.2f}s, 最佳损失: {best_mmoe_loss:.4f}")
    
    # 🔧 计算总的训练时间和统计信息
    total_training_time = time.time() - total_start_time
    unified_training_history['total_training_time'] = total_training_time
    unified_training_history['total_epochs'] = len(unified_training_history['train_losses'])
    
    # 找到最佳验证损失的epoch
    if unified_training_history['val_losses']:
        best_val_idx = np.argmin(unified_training_history['val_losses'])
        unified_training_history['best_epoch'] = best_val_idx
    
    # 详细的性能分析
    print("=" * 60)
    print("🎯 训练完成! 详细结果分析:")
    print("=" * 60)
    
    print(f"📊 各阶段性能:")
    print(f"  时序建模损失: {best_temporal_loss:.6f} ({temporal_history['total_epochs']} epochs, {stage1_time:.1f}s)")
    print(f"  CF建模损失:   {best_cf_loss:.6f} ({cf_history['total_epochs']} epochs, {stage2_time:.1f}s)")
    print(f"  MMoE融合损失: {best_mmoe_loss:.6f} ({mmoe_history['total_epochs']} epochs, {stage3_time:.1f}s)")
    print(f"  总训练轮数:   {unified_training_history['total_epochs']}")
    print(f"  总训练时间:   {total_training_time:.2f}s")
    
    # 改进提升分析
    single_model_best = min(best_temporal_loss, best_cf_loss)
    if best_mmoe_loss < single_model_best:
        improvement = ((single_model_best - best_mmoe_loss) / single_model_best * 100)
        print(f"✅ MMoE相对最佳单模型提升: {improvement:.2f}%")
    else:
        degradation = ((best_mmoe_loss - single_model_best) / single_model_best * 100)
        print(f"⚠️  MMoE相对最佳单模型变化: +{degradation:.2f}%")
    
    print("=" * 60)
    
    # 🔧 修复：返回统一的训练历史，而不是分阶段的
    return model, unified_training_history

def main():
    """主函数 - 保持原有结构，适配改进模型"""
    logging.basicConfig(level=logging.INFO)
    
    # 配置参数调整以适配改进模型
    config = {
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'K_FACTORS': 100,  # 保持与UMTimeModel一致
        'TIME_FACTORS': 40,
        'BATCH_SIZE': 256,
        'NUM_EPOCHS_PER_STAGE': [30, 30, 30],  # CF阶段轮数适配改进层
        'LEARNING_RATES': [0.0005, 0.001, 0.001],
        'REG_STRENGTH': 0.0005,
        'NUM_EXPERTS': 4
    }
    
    print(f'使用设备: {config["DEVICE"]}')
    print(f'MMOE模型配置: {config}')
    
    # 数据加载（保持不变）
    split_path = data_path + 'split_data'
    
    if check_split_data_exists(split_path):
        print("加载现有数据分割...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
    else:
        print("创建新的数据分割...")
        ratings, users, movies = load_data(
            data_path + 'ratings.csv',
            data_path + 'users.csv', 
            data_path + 'movies.csv'
        )
        train_data, val_data, test_data = create_time_aware_split(ratings, random_state=42)
        save_split_data(train_data, val_data, test_data, split_path)
    
    print(f'数据分割 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}')
    
    max_userid = train_data['user_emb_id'].max()
    max_movieid = train_data['movie_emb_id'].max()
    
    # 创建模型（现在使用改进的MMOE，但保持原有接口）
    model = TwoStageMMoEModel(
        max_userid + 1, max_movieid + 1,
        config['K_FACTORS'], config['TIME_FACTORS'],
        config['REG_STRENGTH'], config['NUM_EXPERTS']
    ).to(config['DEVICE'])
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MMOE模型参数数量: {total_params:,}")
    
    # 训练模型（保持原有接口）
    model, training_history = train_mmoe(
        model, train_data, val_data, config['DEVICE'],
        batch_size=config['BATCH_SIZE'],
        num_epochs_per_stage=config['NUM_EPOCHS_PER_STAGE'],
        learning_rates=config['LEARNING_RATES']
    )
    
    # 保存模型（保持原有格式，但包含改进信息）
    checkpoint = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': config['K_FACTORS'],
        'time_factors': config['TIME_FACTORS'],
        'reg_strength': config['REG_STRENGTH'],
        'num_experts': config['NUM_EXPERTS'],
        'best_model_state': model.state_dict(),
        'model_type': model.name,
        'data_split_path': split_path,
        'training_history': training_history,
        'config': config,
        'has_scheduler': True
    }
    
    model_path = data_path + f'model/model_checkpoint_{model.name}.pt'
    torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    print(f'模型保存至: {model_path}')
    
    return model, test_data, training_history

if __name__ == "__main__":
    main()