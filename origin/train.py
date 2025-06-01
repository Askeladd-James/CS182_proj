import torch
import torch.nn as nn
import torch.optim as optim
from data_process import load_data, split_data, save_split_data, check_split_data_exists, load_existing_split_data, MovieLensDataset, data_path
from model import CFModel
from torch.utils.data import DataLoader
import logging
from pathlib import Path
import math
import pandas as pd

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, patience=10):
    """训练模型，包含早停机制和正则化"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
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
        train_losses.append(avg_train_loss)
        
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
        val_losses.append(avg_val_loss)
        
        # 学习率调度
        old_lr = optimizer.param_groups[0]['lr']
        if scheduler:
            scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # 计算RMSE（与TensorFlow版本保持一致）
        train_rmse = math.sqrt(avg_train_loss)
        val_rmse = math.sqrt(avg_val_loss)
        
        # 打印训练信息
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
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
            logging.info(f'New best validation loss: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch+1}')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # 计算最终的最佳RMSE
    best_rmse = math.sqrt(best_val_loss)
    logging.info(f'Best validation loss: {best_val_loss:.4f}')
    logging.info(f'Best validation RMSE: {best_rmse:.4f}')
    
    return best_model_state, train_losses, val_losses

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    
    # 配置参数 - 更接近TensorFlow版本
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_FACTORS = 100
    BATCH_SIZE = 256  # 增大batch size
    LEARNING_RATE = 0.001  # 使用更标准的学习率
    NUM_EPOCHS = 30
    REG_STRENGTH = 0.0001  # 降低正则化强度
    
    logging.info(f'Using device: {DEVICE}')
    
    # 设置分割数据路径
    split_path = data_path + 'split_data'
    
    # 检查是否已有分割数据
    if check_split_data_exists(split_path):
        logging.info("Found existing split data, loading...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
        
        if train_data is not None and val_data is not None and test_data is not None:
            logging.info(f'Loaded existing split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
        else:
            logging.info("Failed to load existing split data, creating new split...")
            # 加载原始数据并重新分割
            ratings, users, movies = load_data(data_path + 'ratings.csv',
                                             data_path + 'users.csv',
                                             data_path + 'movies.csv')
            
            logging.info(f'Loaded {len(ratings)} ratings')
            train_data, val_data, test_data = split_data(ratings, random_state=42)
            save_split_data(train_data, val_data, test_data, split_path)
            logging.info(f'Created new split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
    else:
        logging.info("No existing split data found, creating new split...")
        # 加载原始数据并分割
        ratings, users, movies = load_data(data_path + 'ratings.csv',
                                         data_path + 'users.csv',
                                         data_path + 'movies.csv')
        
        logging.info(f'Loaded {len(ratings)} ratings')
        train_data, val_data, test_data = split_data(ratings, random_state=42)
        save_split_data(train_data, val_data, test_data, split_path)
        logging.info(f'Created new split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}')
    
    # 创建数据加载器
    train_dataset = MovieLensDataset(
        train_data['user_emb_id'].values,
        train_data['movie_emb_id'].values,
        train_data['rating'].values
    )
    
    val_dataset = MovieLensDataset(
        val_data['user_emb_id'].values,
        val_data['movie_emb_id'].values,
        val_data['rating'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 初始化模型
    # 获取用户和电影的最大ID（需要从原始数据获取，因为分割后的数据可能不包含所有ID）
    if 'ratings' not in locals():
        # 如果没有加载过原始数据，快速加载以获取最大ID
        ratings_temp = pd.read_csv(data_path + 'ratings.csv', sep='\t', encoding='latin-1')
        max_userid = ratings_temp['user_id'].max()
        max_movieid = ratings_temp['movie_id'].max()
        del ratings_temp  # 释放内存
    else:
        max_userid = ratings['user_id'].max()
        max_movieid = ratings['movie_id'].max()
    
    logging.info(f'Max user ID: {max_userid}, Max movie ID: {max_movieid}')
    
    model = CFModel(max_userid + 1, max_movieid + 1, K_FACTORS, REG_STRENGTH).to(DEVICE)
    
    criterion = nn.MSELoss()
    
    # 使用Adam优化器（与TensorFlow AdaMax类似）
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    
    # 更温和的学习率调度 - 移除verbose参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )
    
    # 训练模型
    best_model_state, train_losses, val_losses = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS, patience=10
    )
    
    # 保存模型
    config = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': K_FACTORS,
        'best_model_state': best_model_state,
        'data_split_path': split_path,
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    save_checkpoint(model, optimizer, config, data_path + 'model_checkpoint_origin.pt')
    
    return model, test_data

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