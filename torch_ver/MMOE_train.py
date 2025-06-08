import torch
import torch.nn as nn
import torch.optim as optim
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

def create_fusion_dataloader(model, data, user_history_stats, batch_size, device, shuffle=True):
    """创建融合数据加载器"""
    # 获取时序和CF预测
    temporal_predictions = get_predictions_for_data(model, data, user_history_stats, device, stage=1)
    cf_predictions = get_predictions_for_data(model, data, user_history_stats, device, stage=2)
    
    dataset = FusionDataset(data, temporal_predictions, cf_predictions)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def train_stage(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, stage_name, patience=5):
    """训练单个阶段 - 改进版本，包含早停机制"""
    best_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # 记录训练历史
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            if model.training_stage == 1:
                # 时序训练
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
                users, items, ratings, daytime, weekend, years = batch
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
                users, items, ratings, daytime, weekend, years, temporal_preds, cf_preds = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                ratings = ratings.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                temporal_preds = temporal_preds.squeeze().to(device)
                cf_preds = cf_preds.squeeze().to(device)
                
                predictions = model(users, items, daytime, weekend, years, temporal_preds, cf_preds)
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
        
        # 验证阶段
        if val_loader is not None:
            val_loss = evaluate_stage(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            
            print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
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
            print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
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
        'best_loss': best_loss,
        'total_epochs': epoch + 1
    }
    
    return best_loss, training_history

def evaluate_stage(model, val_loader, criterion, device):
    """评估阶段性能"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if model.training_stage == 1:
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
                users, items, ratings, daytime, weekend, years = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                ratings = ratings.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                
                predictions = model(users, items, daytime, weekend, years)
                targets = ratings
            elif model.training_stage == 3:
                users, items, ratings, daytime, weekend, years, temporal_preds, cf_preds = batch
                users = users.squeeze().to(device)
                items = items.squeeze().to(device)
                ratings = ratings.squeeze().to(device)
                daytime = daytime.squeeze().to(device)
                weekend = weekend.squeeze().to(device)
                years = years.squeeze().to(device)
                temporal_preds = temporal_preds.squeeze().to(device)
                cf_preds = cf_preds.squeeze().to(device)
                
                predictions = model(users, items, daytime, weekend, years, temporal_preds, cf_preds)
                targets = ratings
            
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

def train_optimized_mmoe(model, train_data, val_data, device, batch_size=256, 
                        num_epochs_per_stage=[5, 8, 5], learning_rates=[0.001, 0.001, 0.0005]):
    """训练优化的MMoE模型"""
    
    # 准备用户历史统计特征
    print("准备用户历史统计特征...")
    user_history_stats = prepare_user_history_stats(train_data)
    val_user_history_stats = prepare_user_history_stats(val_data)
    
    criterion = nn.MSELoss()
    
    # 存储所有阶段的训练历史
    all_training_history = {}
    
    # 阶段1：时序建模
    print("=" * 50)
    print("阶段1：时序建模（基于统计特征）")
    print("=" * 50)
    
    model.set_training_stage(1)
    temporal_loader = create_temporal_dataloader(train_data, user_history_stats, batch_size)
    temporal_val_loader = create_temporal_dataloader(val_data, val_user_history_stats, batch_size, shuffle=False)
    
    optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rates[0])
    
    best_temporal_loss, temporal_history = train_stage(
        model, temporal_loader, temporal_val_loader, criterion, optimizer1, 
        device, num_epochs_per_stage[0], "Temporal", patience=3
    )
    all_training_history['temporal'] = temporal_history
    
    # 阶段2：CF建模
    print("=" * 50)
    print("阶段2：协同过滤建模")
    print("=" * 50)
    
    model.set_training_stage(2)
    cf_loader = create_standard_dataloader(train_data, batch_size)
    val_loader = create_standard_dataloader(val_data, batch_size, shuffle=False)
    
    optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rates[1])
    
    best_cf_loss, cf_history = train_stage(
        model, cf_loader, val_loader, criterion, optimizer2, 
        device, num_epochs_per_stage[1], "CF", patience=5
    )
    all_training_history['cf'] = cf_history
    
    # 阶段3：MMoE融合
    print("=" * 50)
    print("阶段3：MMoE融合")
    print("=" * 50)
    
    model.set_training_stage(3)
    fusion_loader = create_fusion_dataloader(model, train_data, user_history_stats, batch_size, device)
    fusion_val_loader = create_fusion_dataloader(model, val_data, val_user_history_stats, batch_size, device, shuffle=False)
    
    optimizer3 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                  lr=learning_rates[2])
    
    best_mmoe_loss, mmoe_history = train_stage(
        model, fusion_loader, fusion_val_loader, criterion, optimizer3, 
        device, num_epochs_per_stage[2], "MMoE", patience=3
    )
    all_training_history['mmoe'] = mmoe_history
    
    print("=" * 50)
    print("训练完成!")
    print(f"最佳时序损失: {best_temporal_loss:.4f} (轮数: {temporal_history['total_epochs']})")
    print(f"最佳CF损失: {best_cf_loss:.4f} (轮数: {cf_history['total_epochs']})")
    print(f"最佳MMoE损失: {best_mmoe_loss:.4f} (轮数: {mmoe_history['total_epochs']})")
    print("=" * 50)
    
    return model, all_training_history

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO)
    
    # 配置
    config = {
        'DEVICE': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'K_FACTORS': 60,
        'TIME_FACTORS': 16,
        'BATCH_SIZE': 512,
        'NUM_EPOCHS_PER_STAGE': [20, 20, 10],  # 适当增加轮数，因为有早停
        'LEARNING_RATES': [0.002, 0.001, 0.0005],
        'REG_STRENGTH': 0.001,
        'NUM_EXPERTS': 3
    }
    
    print(f'使用设备: {config["DEVICE"]}')
    
    # 加载数据
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
    
    # 获取最大ID
    max_userid = train_data['user_emb_id'].max()
    max_movieid = train_data['movie_emb_id'].max()
    
    # 创建优化的模型
    model = TwoStageMMoEModel(
        max_userid + 1, max_movieid + 1,
        config['K_FACTORS'], config['TIME_FACTORS'],
        config['REG_STRENGTH'], config['NUM_EXPERTS']
    ).to(config['DEVICE'])
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    model, training_history = train_optimized_mmoe(
        model, train_data, val_data, config['DEVICE'],
        batch_size=config['BATCH_SIZE'],
        num_epochs_per_stage=config['NUM_EPOCHS_PER_STAGE'],
        learning_rates=config['LEARNING_RATES']
    )
    
    # 保存模型和训练历史
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
        'training_history': training_history,  # 添加训练历史
        'config': config
    }
    
    model_path = data_path + f'model/model_checkpoint_{model.name}.pt'
    torch.save(checkpoint, model_path, _use_new_zipfile_serialization=False)
    print(f'模型保存至: {model_path}')
    
    return model, test_data, training_history

if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from data_process import (load_data, create_time_aware_split, save_split_data, 
#                          check_split_data_exists, load_existing_split_data, 
#                          MovieLensDataset, data_path)
# from MMOE import TwoStageMMoEModel
# from torch.utils.data import DataLoader, Dataset
# import logging
# import math
# import pandas as pd
# from pathlib import Path
# import numpy as np

# class SequentialTrainingDataset(Dataset):
#     """时序训练数据集"""
#     def __init__(self, sequential_data, model):
#         self.sequential_data = sequential_data
#         self.model = model
    
#     def __len__(self):
#         return len(self.sequential_data)
    
#     def __getitem__(self, idx):
#         item = self.sequential_data[idx]
#         sequence = item['sequence']
#         target_rating = item['target_rating']
        
#         # 构建序列的原始ID特征
#         raw_features = []
#         for _, row in sequence.iterrows():
#             feature_vector = [
#                 int(row['user_emb_id']),
#                 int(row['daytime']), 
#                 int(row['is_weekend']),
#                 int(row['year']),
#                 float(row['rating'])
#             ]
#             raw_features.append(feature_vector)
        
#         # 转换为tensor，padding到固定长度
#         max_seq_len = 20
#         if len(raw_features) > max_seq_len:
#             raw_features = raw_features[-max_seq_len:]
#         else:
#             # 用零填充
#             while len(raw_features) < max_seq_len:
#                 raw_features.insert(0, [0, 0, 0, 0, 0.0])
        
#         # 转换为tensor
#         sequence_tensor = torch.LongTensor([[int(f[0]), int(f[1]), int(f[2]), int(f[3])] for f in raw_features])  # IDs
#         ratings_tensor = torch.FloatTensor([f[4] for f in raw_features])  # 评分
#         target_tensor = torch.FloatTensor([target_rating])
        
#         return sequence_tensor, ratings_tensor, target_tensor

# class FusionTrainingDataset(Dataset):
#     """融合训练数据集"""
#     def __init__(self, data, lstm_predictions, cf_predictions):
#         self.data = data.reset_index(drop=True)
#         self.lstm_predictions = lstm_predictions
#         self.cf_predictions = cf_predictions
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         return (
#             torch.LongTensor([row['user_emb_id']]),
#             torch.LongTensor([row['movie_emb_id']]),
#             torch.FloatTensor([row['rating']]),
#             torch.LongTensor([row['daytime']]),
#             torch.LongTensor([row['is_weekend']]),
#             torch.LongTensor([row['year']]),
#             torch.FloatTensor([self.lstm_predictions[idx]]),
#             torch.FloatTensor([self.cf_predictions[idx]])
#         )

# def sequential_collate_fn(batch):
#     """时序数据的collate函数"""
#     sequences, ratings, targets = zip(*batch)
#     sequences = torch.stack(sequences)  # [batch_size, seq_len, 4]
#     ratings = torch.stack(ratings)      # [batch_size, seq_len]
#     targets = torch.stack(targets).squeeze()  # [batch_size]
#     return sequences, ratings, targets


# def prepare_sequential_training_data(train_data):
#     """准备时序训练数据"""
#     sequential_data = []
    
#     # 按用户分组，每个用户的数据按时间排序
#     for user_id in train_data['user_emb_id'].unique():
#         user_data = train_data[train_data['user_emb_id'] == user_id].copy()
#         user_data = user_data.sort_values('timestamp')
        
#         # 为每个用户创建序列
#         if len(user_data) > 1:  # 至少需要2个评分来创建序列
#             for i in range(1, len(user_data)):
#                 # 使用前i个评分预测第i+1个评分
#                 sequence = user_data.iloc[:i+1]
#                 sequential_data.append({
#                     'user_id': user_id,
#                     'sequence': sequence,
#                     'target_rating': sequence.iloc[-1]['rating']
#                 })
    
#     return sequential_data

# def create_sequential_dataloader(sequential_data, batch_size, model):
#     """创建时序数据加载器"""
#     dataset = SequentialTrainingDataset(sequential_data, model)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=sequential_collate_fn)

# def create_random_dataloader(data, batch_size, shuffle=True):
#     """创建随机顺序的数据加载器"""
#     dataset = MovieLensDataset(
#         data['user_emb_id'].values,
#         data['movie_emb_id'].values,
#         data['rating'].values,
#         data['daytime'].values,
#         data['is_weekend'].values,
#         data['year'].values
#     )
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# def get_lstm_predictions_for_data(model, data, sequential_data, device):
#     """为数据获取LSTM预测"""
#     model.eval()
#     model.set_training_stage(1)
    
#     predictions = []
    
#     # 创建用户序列映射
#     user_sequences = {}
#     for item in sequential_data:
#         user_id = item['user_id']
#         if user_id not in user_sequences:
#             user_sequences[user_id] = []
#         user_sequences[user_id].append(item)
    
#     with torch.no_grad():
#         for _, row in data.iterrows():
#             user_id = row['user_emb_id']
            
#             if user_id in user_sequences and user_sequences[user_id]:
#                 # 使用最新的序列
#                 latest_sequence = user_sequences[user_id][-1]
#                 sequence = latest_sequence['sequence']
                
#                 # 构建特征序列
#                 features = []
#                 for _, seq_row in sequence.iterrows():
#                     feature_vector = [
#                         seq_row['user_emb_id'],
#                         seq_row['daytime'],
#                         seq_row['is_weekend'],
#                         seq_row['year'],
#                         seq_row['rating']
#                     ]
#                     features.append(feature_vector)
                
#                 # 填充到固定长度
#                 max_seq_len = 20
#                 if len(features) > max_seq_len:
#                     features = features[-max_seq_len:]
#                 else:
#                     while len(features) < max_seq_len:
#                         features.insert(0, [0, 0, 0, 0, 0])
                
#                 sequence_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
#                 pred = model(sequence_tensor)
#                 predictions.append(pred.item())
#             else:
#                 # 如果没有序列数据，使用默认值
#                 predictions.append(3.0)  # 默认评分
    
#     return predictions

# def get_cf_predictions_for_data(model, data, device):
#     """为数据获取CF预测"""
#     model.eval()
#     model.set_training_stage(2)
    
#     predictions = []
    
#     with torch.no_grad():
#         for _, row in data.iterrows():
#             user_tensor = torch.LongTensor([row['user_emb_id']]).to(device)
#             item_tensor = torch.LongTensor([row['movie_emb_id']]).to(device)
#             daytime_tensor = torch.LongTensor([row['daytime']]).to(device)
#             weekend_tensor = torch.LongTensor([row['is_weekend']]).to(device)
#             year_tensor = torch.LongTensor([row['year']]).to(device)
            
#             pred = model(user_tensor, item_tensor, daytime_tensor, weekend_tensor, year_tensor)
#             predictions.append(pred.item())
    
#     return predictions

# def create_fusion_dataloader(model, data, sequential_data, batch_size, device, shuffle=True):
#     """创建融合训练的数据加载器"""
#     # 预计算LSTM和CF的预测结果
#     lstm_predictions = get_lstm_predictions_for_data(model, data, sequential_data, device)
#     cf_predictions = get_cf_predictions_for_data(model, data, device)
    
#     dataset = FusionTrainingDataset(data, lstm_predictions, cf_predictions)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# def evaluate_stage(model, val_loader, criterion, device):
#     """评估某个阶段的性能"""
#     model.eval()
#     total_loss = 0
#     num_batches = 0
    
#     with torch.no_grad():
#         for batch in val_loader:
#             if model.training_stage == 1:
#                 # LSTM评估
#                 sequences, ratings, targets = batch
#                 sequences = sequences.to(device)
#                 ratings = ratings.to(device)
#                 targets = targets.to(device)
#                 predictions = model(sequences, ratings)  # 修改这里
#             elif model.training_stage == 2:
#                 # CF评估
#                 users, items, ratings, daytime, weekend, years = batch
#                 users = users.to(device)
#                 items = items.to(device)
#                 ratings = ratings.to(device)
#                 daytime = daytime.to(device)
#                 weekend = weekend.to(device)
#                 years = years.to(device)
#                 predictions = model(users, items, daytime, weekend, years)
#                 targets = ratings
#             elif model.training_stage == 3:
#                 # MMoE评估
#                 users, items, ratings, daytime, weekend, years, lstm_preds, cf_preds = batch
#                 users = users.to(device)
#                 items = items.to(device)
#                 ratings = ratings.to(device)
#                 daytime = daytime.to(device)
#                 weekend = weekend.to(device)
#                 years = years.to(device)
#                 lstm_preds = lstm_preds.to(device)
#                 cf_preds = cf_preds.to(device)
#                 predictions = model(users, items, daytime, weekend, years, lstm_preds, cf_preds)
#                 targets = ratings
            
#             loss = criterion(predictions, targets)
#             total_loss += loss.item()
#             num_batches += 1
    
#     return total_loss / num_batches

# def train_stage(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, stage_name):
#     """单个阶段的训练函数"""
#     best_loss = float('inf')
    
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         num_batches = 0
        
#         for batch in train_loader:
#             optimizer.zero_grad()
            
#             if model.training_stage == 1:
#                 # LSTM训练
#                 sequences, ratings, targets = batch
#                 sequences = sequences.to(device)
#                 ratings = ratings.to(device)
#                 targets = targets.to(device)
#                 predictions = model(sequences, ratings)  # 修改这里
#             elif model.training_stage == 2:
#                 # CF训练
#                 users, items, ratings, daytime, weekend, years = batch
#                 users = users.to(device)
#                 items = items.to(device)
#                 ratings = ratings.to(device)
#                 daytime = daytime.to(device)
#                 weekend = weekend.to(device)
#                 years = years.to(device)
#                 predictions = model(users, items, daytime, weekend, years)
#                 targets = ratings
#             elif model.training_stage == 3:
#                 # MMoE训练
#                 users, items, ratings, daytime, weekend, years, lstm_preds, cf_preds = batch
#                 users = users.to(device)
#                 items = items.to(device)
#                 ratings = ratings.to(device)
#                 daytime = daytime.to(device)
#                 weekend = weekend.to(device)
#                 years = years.to(device)
#                 lstm_preds = lstm_preds.to(device)
#                 cf_preds = cf_preds.to(device)
#                 predictions = model(users, items, daytime, weekend, years, lstm_preds, cf_preds)
#                 targets = ratings
            
#             loss = criterion(predictions, targets)
#             reg_loss = model.get_regularization_loss()
#             total_loss_batch = loss + reg_loss
            
#             total_loss_batch.backward()
            
#             # 梯度裁剪
#             torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
#             optimizer.step()
            
#             total_loss += loss.item()
#             num_batches += 1
        
#         avg_loss = total_loss / num_batches
        
#         # 验证
#         if val_loader is not None:
#             val_loss = evaluate_stage(model, val_loader, criterion, device)
#             print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
#             if val_loss < best_loss:
#                 best_loss = val_loss
#         else:
#             print(f"{stage_name} Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
    
#     return best_loss

# def train_two_stage_model(model, train_data, val_data, device, 
#                          batch_size=256, num_epochs_per_stage=[10, 15, 10], 
#                          learning_rates=[0.001, 0.001, 0.0005]):
#     """两阶段训练函数"""
    
#     # 阶段1：训练LSTM时序建模
#     print("=" * 50)
#     print("阶段1：训练LSTM时序建模")
#     print("=" * 50)
    
#     model.set_training_stage(1)
    
#     # 准备时序数据（按用户和时间排序）
#     sequential_data = prepare_sequential_training_data(train_data)
#     sequential_loader = create_sequential_dataloader(sequential_data, batch_size, model)  # 传入model
    
#     optimizer1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
#                                   lr=learning_rates[0])
#     criterion = nn.MSELoss()
    
#     best_lstm_loss = train_stage(model, sequential_loader, None, criterion, optimizer1, 
#                                 device, num_epochs_per_stage[0], stage_name="LSTM")
    
#     # 阶段2：训练CF网络
#     print("=" * 50)
#     print("阶段2：训练CF网络")
#     print("=" * 50)
    
#     model.set_training_stage(2)
    
#     # 准备乱序数据
#     random_loader = create_random_dataloader(train_data, batch_size)
#     val_loader = create_random_dataloader(val_data, batch_size, shuffle=False)
    
#     optimizer2 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
#                                   lr=learning_rates[1])
    
#     best_cf_loss = train_stage(model, random_loader, val_loader, criterion, optimizer2, 
#                               device, num_epochs_per_stage[1], stage_name="CF")
    
#     # 阶段3：训练MMoE融合
#     print("=" * 50)
#     print("阶段3：训练MMoE融合")
#     print("=" * 50)
    
#     model.set_training_stage(3)
    
#     # 准备融合训练数据
#     fusion_loader = create_fusion_dataloader(model, train_data, sequential_data, batch_size, device)
#     fusion_val_loader = create_fusion_dataloader(model, val_data, sequential_data, batch_size, device, shuffle=False)
    
#     optimizer3 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
#                                   lr=learning_rates[2])
    
#     best_mmoe_loss = train_stage(model, fusion_loader, fusion_val_loader, criterion, optimizer3, 
#                                 device, num_epochs_per_stage[2], stage_name="MMoE")
    
#     print("=" * 50)
#     print("训练完成!")
#     print(f"最佳LSTM损失: {best_lstm_loss:.4f}")
#     print(f"最佳CF损失: {best_cf_loss:.4f}")
#     print(f"最佳MMoE损失: {best_mmoe_loss:.4f}")
#     print("=" * 50)
    
#     return model

# def save_checkpoint(model, optimizer, config, path):
#     """安全保存模型检查点"""
#     try:
#         save_dict = {
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             **config
#         }
#         torch.save(save_dict, path, _use_new_zipfile_serialization=False)
#         logging.info(f'Model saved to {path}')
#     except Exception as e:
#         logging.error(f"保存模型失败: {str(e)}")
#         raise

# def main():
#     """主训练函数"""
#     # 设置日志
#     logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
#     # 配置参数
#     DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     K_FACTORS = 80
#     TIME_FACTORS = 20
#     BATCH_SIZE = 256
#     NUM_EPOCHS_PER_STAGE = [5, 10, 5]  # LSTM, CF, MMoE的训练轮数
#     LEARNING_RATES = [0.001, 0.001, 0.0005]
#     REG_STRENGTH = 0.001
#     NUM_EXPERTS = 4
    
#     logging.info(f'Using device: {DEVICE}')
    
#     # 设置分割数据路径
#     split_path = data_path + 'split_data'
    
#     # 检查是否已有分割数据
#     if check_split_data_exists(split_path):
#         logging.info("Found existing split data, loading...")
#         train_data, val_data, test_data = load_existing_split_data(split_path)
        
#         if train_data is not None and val_data is not None and test_data is not None:
#             logging.info(f'Loaded existing split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
#         else:
#             logging.info("Failed to load existing split data, creating new split...")
#             ratings, users, movies = load_data(data_path + 'ratings.csv',
#                                              data_path + 'users.csv',
#                                              data_path + 'movies.csv')
            
#             logging.info(f'Loaded {len(ratings)} ratings')
#             train_data, val_data, test_data = create_time_aware_split(ratings, random_state=42)
#             save_split_data(train_data, val_data, test_data, split_path)
#             logging.info(f'Created new split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
#     else:
#         logging.info("No existing split data found, creating new split...")
#         ratings, users, movies = load_data(data_path + 'ratings.csv',
#                                          data_path + 'users.csv',
#                                          data_path + 'movies.csv')
        
#         logging.info(f'Loaded {len(ratings)} ratings')
#         train_data, val_data, test_data = create_time_aware_split(ratings, random_state=42)
#         save_split_data(train_data, val_data, test_data, split_path)
#         logging.info(f'Created new split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
    
#     # 获取最大ID
#     if 'ratings' not in locals():
#         ratings_temp = pd.read_csv(data_path + 'ratings.csv', sep='\t', encoding='latin-1')
#         max_userid = ratings_temp['user_id'].max()
#         max_movieid = ratings_temp['movie_id'].max()
#         del ratings_temp
#     else:
#         max_userid = ratings['user_id'].max()
#         max_movieid = ratings['movie_id'].max()
    
#     logging.info(f'Max user ID: {max_userid}, Max movie ID: {max_movieid}')
    
#     # 初始化TwoStageMMoE模型
#     model = TwoStageMMoEModel(
#         max_userid + 1, max_movieid + 1, K_FACTORS, TIME_FACTORS, REG_STRENGTH, NUM_EXPERTS
#     ).to(DEVICE)
    
#     # 两阶段训练
#     model = train_two_stage_model(
#         model, train_data, val_data, DEVICE, 
#         batch_size=BATCH_SIZE, 
#         num_epochs_per_stage=NUM_EPOCHS_PER_STAGE,
#         learning_rates=LEARNING_RATES
#     )
    
#     # 保存模型
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[0])
#     config = {
#         'max_userid': max_userid,
#         'max_movieid': max_movieid,
#         'k_factors': K_FACTORS,
#         'time_factors': TIME_FACTORS,
#         'best_model_state': model.state_dict(),
#         'data_split_path': split_path,
#         'model_type': model.name,
#         'num_experts': NUM_EXPERTS,
#         'reg_strength': REG_STRENGTH
#     }
    
#     save_checkpoint(model, optimizer, config, data_path + 'model/' + 'model_checkpoint_' + model.name + '.pt')
    
#     return model, test_data

# if __name__ == "__main__":
#     main()