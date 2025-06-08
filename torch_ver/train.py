import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_process import load_data, create_time_aware_split, save_split_data, check_split_data_exists, load_existing_split_data, MovieLensDataset, data_path
from model import IndependentTimeModel, UserTimeModel, UMTimeModel
from torch.utils.data import DataLoader
import logging
import math
import pandas as pd
from pathlib import Path

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=30, patience=10):
    """训练模型，包含早停机制和正则化 - 增强版学习率调度"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []
    learning_rates = []  # 新增：记录学习率历史
    
    for epoch in range(num_epochs):
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
        train_losses.append(avg_train_loss)
        
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
        val_losses.append(avg_val_loss)
        
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # 学习率调度 - 增强版
        old_lr = current_lr
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        
        # 计算RMSE
        train_rmse = math.sqrt(avg_train_loss)
        val_rmse = math.sqrt(avg_val_loss)
        
        # 打印训练信息 - 增强版
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info(f'Train Loss: {avg_train_loss:.4f}, Train RMSE: {train_rmse:.4f}')
        logging.info(f'Val Loss: {avg_val_loss:.4f}, Val RMSE: {val_rmse:.4f}')
        logging.info(f'Learning rate: {new_lr:.6f}')
        
        # 学习率变化提示 - 增强版
        if new_lr != old_lr:
            reduction_percent = ((old_lr - new_lr) / old_lr * 100)
            logging.info(f'  → 学习率调整: {old_lr:.6f} → {new_lr:.6f} (下降 {reduction_percent:.1f}%)')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            logging.info(f'  → 新的最佳验证损失: {avg_val_loss:.4f}')
        else:
            patience_counter += 1
            logging.info(f'  → 验证损失未改善 ({patience_counter}/{patience})')
            
        if patience_counter >= patience:
            logging.info(f'  → 早停：验证损失连续 {patience} 轮未改善')
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            break
    
    # 最终统计和学习率分析
    best_rmse = math.sqrt(best_val_loss)
    
    # 学习率调度效果分析
    if learning_rates:
        initial_lr = learning_rates[0]
        final_lr = learning_rates[-1]
        total_reduction = ((initial_lr - final_lr) / initial_lr * 100) if initial_lr > 0 else 0
        
        logging.info(f'训练完成统计:')
        logging.info(f'  最佳验证损失: {best_val_loss:.4f}')
        logging.info(f'  最佳验证RMSE: {best_rmse:.4f}')
        logging.info(f'  学习率变化: {initial_lr:.6f} → {final_lr:.6f} (总下降 {total_reduction:.1f}%)')
        logging.info(f'  总训练轮数: {epoch + 1}/{num_epochs}')
    else:
        logging.info(f'Best validation loss: {best_val_loss:.4f}')
        logging.info(f'Best validation RMSE: {best_rmse:.4f}')
    
    # 返回增强的训练历史
    training_history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'best_loss': best_val_loss,
        'best_rmse': best_rmse,
        'total_epochs': epoch + 1
    }
    
    return best_model_state, training_history

def main():
    # 设置日志 - 增强版
    log_file = data_path + 'training_single_model.log'
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # 优化的配置参数 - 针对学习率调度器调整
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_FACTORS = 80
    TIME_FACTORS = 20
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    REG_STRENGTH = 0.001
    
    logging.info(f'使用设备: {DEVICE}')
    logging.info(f'配置参数: K_FACTORS={K_FACTORS}, TIME_FACTORS={TIME_FACTORS}, BATCH_SIZE={BATCH_SIZE}')
    logging.info(f'学习率调度器: ReduceLROnPlateau (初始学习率={LEARNING_RATE})')
    
    # 设置分割数据路径
    split_path = data_path + 'split_data'
    
    # 检查是否已有分割数据
    if check_split_data_exists(split_path):
        logging.info("加载现有数据分割...")
        train_data, val_data, test_data = load_existing_split_data(split_path)
        
        if train_data is not None and val_data is not None and test_data is not None:
            logging.info(f'数据分割 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}')
        else:
            logging.info("加载现有数据分割失败，创建新的分割...")
            # 加载原始数据并重新分割
            ratings, users, movies = load_data(data_path + 'ratings.csv',
                                             data_path + 'users.csv',
                                             data_path + 'movies.csv')
            
            logging.info(f'加载了 {len(ratings)} 条评分记录')
            train_data, val_data, test_data = create_time_aware_split(ratings, random_state=42)
            save_split_data(train_data, val_data, test_data, split_path)
            logging.info(f'创建新分割 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}')
    else:
        logging.info("未找到现有数据分割，创建新的分割...")
        # 加载原始数据并分割
        ratings, users, movies = load_data(data_path + 'ratings.csv',
                                         data_path + 'users.csv',
                                         data_path + 'movies.csv')
        
        logging.info(f'加载了 {len(ratings)} 条评分记录')
        train_data, val_data, test_data = create_time_aware_split(ratings, random_state=42)
        save_split_data(train_data, val_data, test_data, split_path)
        logging.info(f'创建新分割 - 训练: {len(train_data)}, 验证: {len(val_data)}, 测试: {len(test_data)}')
    
    # 创建数据加载器
    train_dataset = MovieLensDataset(
        train_data['user_emb_id'].values,
        train_data['movie_emb_id'].values,
        train_data['rating'].values,
        train_data['daytime'].values,
        train_data['is_weekend'].values,
        train_data['year'].values
    )
    
    val_dataset = MovieLensDataset(
        val_data['user_emb_id'].values,
        val_data['movie_emb_id'].values,
        val_data['rating'].values,
        val_data['daytime'].values,
        val_data['is_weekend'].values,
        val_data['year'].values
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 获取最大ID
    if 'ratings' not in locals():
        # 如果没有加载过原始数据，快速加载以获取最大ID
        ratings_temp = pd.read_csv(data_path + 'ratings.csv', sep='\t', encoding='latin-1')
        max_userid = ratings_temp['user_id'].max()
        max_movieid = ratings_temp['movie_id'].max()
        del ratings_temp  # 释放内存
    else:
        max_userid = ratings['user_id'].max()
        max_movieid = ratings['movie_id'].max()
    
    logging.info(f'最大用户ID: {max_userid}, 最大电影ID: {max_movieid}')
    
    ###################   选择模型   #######################
    
    # 使用模型 - 可以选择不同的模型进行测试
    # model = IndependentTimeModel(
    #     max_userid + 1, max_movieid + 1, K_FACTORS, TIME_FACTORS, REG_STRENGTH
    # ).to(DEVICE)
    model = UserTimeModel(
        max_userid + 1, max_movieid + 1, K_FACTORS, TIME_FACTORS, REG_STRENGTH
    ).to(DEVICE)
    # model = UMTimeModel(
    #     max_userid + 1, max_movieid + 1, K_FACTORS, TIME_FACTORS, REG_STRENGTH
    # ).to(DEVICE)
    
    #######################################################
    
    # 统计模型参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f'选择模型: {model.name}')
    logging.info(f'模型参数: 总数={total_params:,}, 可训练={trainable_params:,}')
    
    criterion = nn.MSELoss()
    
    # 使用AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 学习率调度器 - 优化参数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.6, patience=5, min_lr=1e-6
    )
    
    logging.info(f'优化器: AdamW (lr={LEARNING_RATE}, weight_decay=1e-5)')
    logging.info(f'学习率调度器: ReduceLROnPlateau (factor=0.6, patience=5, min_lr=1e-6)')
    
    # 训练模型
    logging.info("开始训练...")
    best_model_state, training_history = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler, 
        DEVICE, NUM_EPOCHS, patience=10
    )
    
    # 保存模型 - 增强版配置
    config = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': K_FACTORS,
        'time_factors': TIME_FACTORS,
        'reg_strength': REG_STRENGTH,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'best_model_state': best_model_state,
        'data_split_path': split_path,
        'training_history': training_history,
        'model_type': model.name,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'scheduler_type': 'ReduceLROnPlateau',
        'has_scheduler': True
    }
    
    model_filename = f'model_checkpoint_{model.name}_with_scheduler.pt'
    save_checkpoint(model, optimizer, config, data_path + model_filename)
    
    logging.info("=" * 60)
    logging.info("训练完成!")
    logging.info(f"模型: {model.name}")
    logging.info(f"最佳验证RMSE: {training_history['best_rmse']:.4f}")
    logging.info(f"学习率变化: {training_history['learning_rates'][0]:.6f} → {training_history['learning_rates'][-1]:.6f}")
    logging.info(f"实际训练轮数: {training_history['total_epochs']}")
    logging.info("=" * 60)
    
    return model, test_data

def save_checkpoint(model, optimizer, config, path):
    """安全保存模型检查点 - 增强版"""
    try:
        # 确保目录存在
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **config
        }
        torch.save(save_dict, path, _use_new_zipfile_serialization=False)
        logging.info(f'模型已保存至: {path}')
        
        # 保存训练历史的JSON版本（便于分析）
        import json
        history_path = path.replace('.pt', '_history.json')
        history_data = {
            'model_type': config['model_type'],
            'training_history': config['training_history'],
            'model_params': {
                'total_params': config['total_params'],
                'trainable_params': config['trainable_params'],
                'k_factors': config['k_factors'],
                'time_factors': config['time_factors']
            }
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        logging.info(f'训练历史已保存至: {history_path}')
        
    except Exception as e:
        logging.error(f"保存模型失败: {str(e)}")
        raise

if __name__ == "__main__":
    main()