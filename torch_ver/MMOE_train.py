import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from data_process import (load_data, create_time_aware_split, save_split_data, 
                         check_split_data_exists, load_existing_split_data, 
                         MovieLensDataset, data_path)
from MMOE import TwoStageMMoEModel
from torch.utils.data import DataLoader, Dataset
import logging
import pandas as pd
import numpy as np
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
    """准备用户历史统计特征 - 替代复杂的序列处理"""
    user_stats = {}
    
    # 按用户分组计算统计特征
    for user_id in train_data['user_emb_id'].unique():
        user_data = train_data[train_data['user_emb_id'] == user_id].copy()
        user_data = user_data.sort_values('timestamp')
        
        ratings = user_data['rating'].values
        
        if len(ratings) > 0:
            # 计算统计特征
            avg_rating = np.mean(ratings)
            std_rating = np.std(ratings) if len(ratings) > 1 else 0.0
            num_ratings = len(ratings)
            recent_rating = ratings[-1] if len(ratings) > 0 else 3.0
            
            # 计算评分趋势（最近5个评分的斜率）
            if len(ratings) >= 5:
                recent_ratings = ratings[-5:]
                x = np.arange(len(recent_ratings))
                trend = np.polyfit(x, recent_ratings, 1)[0]  # 线性回归斜率
            else:
                trend = 0.0
            
            user_stats[user_id] = [avg_rating, std_rating, min(num_ratings, 100), recent_rating, trend]
        else:
            user_stats[user_id] = [3.0, 1.0, 1.0, 3.0, 0.0]
    
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

def train_stage(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, stage_name, patience=5, scheduler=None):
    """训练单个阶段 - 修复所有阶段的batch解包问题"""
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    learning_rates = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
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
                # CF训练 - 使用*extra_features处理可能的变长参数
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
                # MMoE训练 - 使用*extra_features处理可变数量的元素
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
            reg_loss = model.get_regularization_loss()
            total_loss_batch = loss + reg_loss
            
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 验证阶段
        if val_loader is not None:
            val_loss = evaluate_stage(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # 更新学习率调度器
            old_lr = current_lr
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
                
                # 检查学习率是否改变
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
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
            print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f}")
            
            # 对于没有验证集的情况，基于训练损失更新调度器
            old_lr = current_lr
            if scheduler is not None:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_train_loss)
                else:
                    scheduler.step()
                    
                # 检查学习率是否改变
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != old_lr:
                    print(f"  → 学习率调整: {old_lr:.6f} → {new_lr:.6f}")
                    
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_model_state = model.state_dict().copy()
    
    # 恢复最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  → 已恢复到最佳模型状态")
    
    # 返回训练历史用于分析
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
               num_epochs_per_stage=[8, 12, 8], learning_rates=[0.001, 0.001, 0.0005]):
    """训练MMoE模型 - 直接适配改进的CF层，不改变函数签名"""
    
    # 准备用户历史统计特征
    print("准备用户历史统计特征...")
    user_history_stats = prepare_user_history_stats(train_data)
    val_user_history_stats = prepare_user_history_stats(val_data)
    
    criterion = nn.MSELoss()
    all_training_history = {}
    
    # 阶段1：时序建模（适配改进模型，增加轮数）
    print("=" * 50)
    print("阶段1：时序建模")
    print("=" * 50)
    
    model.set_training_stage(1)
    temporal_loader = create_temporal_dataloader(train_data, user_history_stats, batch_size)
    temporal_val_loader = create_temporal_dataloader(val_data, val_user_history_stats, batch_size, shuffle=False)
    
    # 调整优化器参数以适配改进模型
    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rates[0], weight_decay=1e-5)
    scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.6, patience=4, min_lr=1e-6)
    
    best_temporal_loss, temporal_history = train_stage(
        model, temporal_loader, temporal_val_loader, criterion, optimizer1, 
        device, num_epochs_per_stage[0], "Temporal", patience=8, scheduler=scheduler1
    )
    all_training_history['temporal'] = temporal_history
    
    print(f"阶段1完成 - 最终学习率: {optimizer1.param_groups[0]['lr']:.6f}")
    
    # 阶段2：CF建模（适配UMTimeModel风格的CF层，增加轮数）
    print("=" * 50)
    print("阶段2：协同过滤建模")
    print("=" * 50)
    
    model.set_training_stage(2)
    cf_loader = create_standard_dataloader(train_data, batch_size)
    val_loader = create_standard_dataloader(val_data, batch_size, shuffle=False)
    
    # CF阶段使用更细致的学习率调度
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rates[1], weight_decay=1e-5)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.6, patience=5, min_lr=1e-6)
    
    best_cf_loss, cf_history = train_stage(
        model, cf_loader, val_loader, criterion, optimizer2, 
        device, num_epochs_per_stage[1], "CF", patience=10, scheduler=scheduler2
    )
    all_training_history['cf'] = cf_history
    
    print(f"阶段2完成 - 最终学习率: {optimizer2.param_groups[0]['lr']:.6f}")
    
    # 阶段3：MMoE融合（清除缓存以使用新CF预测）
    print("=" * 50)
    print("阶段3：MMoE融合")
    print("=" * 50)
    
    model.set_training_stage(3)
    
    # 清除旧缓存文件以使用新的CF预测
    train_cache_file = data_path + 'cache_train_predictions.pt'
    val_cache_file = data_path + 'cache_val_predictions.pt'
    
    for cache_file in [train_cache_file, val_cache_file]:
        if Path(cache_file).exists():
            Path(cache_file).unlink()
    
    print("准备融合训练数据...")
    fusion_loader = create_cached_fusion_dataloader(
        model, train_data, user_history_stats, batch_size * 2, device, 
        cache_file=train_cache_file
    )
    
    print("准备融合验证数据...")
    fusion_val_loader = create_cached_fusion_dataloader(
        model, val_data, val_user_history_stats, batch_size * 2, device, 
        shuffle=False, cache_file=val_cache_file
    )
    
    optimizer3 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rates[2], weight_decay=1e-5)
    scheduler3 = ReduceLROnPlateau(optimizer3, mode='min', factor=0.6, patience=5, min_lr=1e-7)
    
    best_mmoe_loss, mmoe_history = train_stage(
        model, fusion_loader, fusion_val_loader, criterion, optimizer3, 
        device, num_epochs_per_stage[2], "MMoE", patience=10, scheduler=scheduler3
    )
    all_training_history['mmoe'] = mmoe_history
    
    print(f"阶段3完成 - 最终学习率: {optimizer3.param_groups[0]['lr']:.6f}")
    
    # 性能分析（保持原有格式）
    print("=" * 50)
    print("训练完成! 结果分析:")
    print("=" * 50)
    print(f"时序损失: {best_temporal_loss:.4f} (轮数: {temporal_history['total_epochs']})")
    print(f"CF损失: {best_cf_loss:.4f} (轮数: {cf_history['total_epochs']})")
    print(f"MMoE融合损失: {best_mmoe_loss:.4f} (轮数: {mmoe_history['total_epochs']})")
    
    better_base = min(best_temporal_loss, best_cf_loss)
    if best_mmoe_loss < better_base:
        improvement = ((better_base - best_mmoe_loss) / better_base * 100)
        print(f"✅ MMoE相对最好单模型提升: {improvement:.2f}%")
    else:
        degradation = ((best_mmoe_loss - better_base) / better_base * 100)
        print(f"❌ MMoE相对最好单模型下降: {degradation:.2f}%")
    
    print("=" * 50)
    
    return model, all_training_history

def main():
    """主函数 - 保持原有结构，适配改进模型"""
    logging.basicConfig(level=logging.INFO)
    
    # 配置参数调整以适配改进模型
    config = {
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'K_FACTORS': 60,  # 保持与UMTimeModel一致
        'TIME_FACTORS': 20,
        'BATCH_SIZE': 256,
        'NUM_EPOCHS_PER_STAGE': [1, 1, 1],  # CF阶段轮数适配改进层
        'LEARNING_RATES': [0.001, 0.001, 0.001],
        'REG_STRENGTH': 0.001,
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