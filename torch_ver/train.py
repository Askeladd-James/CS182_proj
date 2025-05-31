import torch
import torch.nn as nn
import torch.optim as optim
from data_process import load_data, create_time_aware_split, save_split_data, MovieLensDataset, data_path
from model import CFModel, TimeAwareCFModel
from torch.utils.data import DataLoader
import logging
from pathlib import Path

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=10):
    """训练模型，包含早停机制"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for user, movie, rating, daytime, weekend, year in train_loader:
            user = user.to(device)
            movie = movie.to(device)
            rating = rating.to(device)
            daytime = daytime.to(device)
            weekend = weekend.to(device)
            year = year.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, movie, daytime, weekend, year)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
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
        
        avg_val_loss = val_loss / len(val_loader)
        
        # 打印训练信息
        logging.info(f'Epoch {epoch+1}/{num_epochs}')
        logging.info(f'Average train loss: {avg_train_loss:.4f}')
        logging.info(f'Average validation loss: {avg_val_loss:.4f}')
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            logging.info(f'Early stopping at epoch {epoch+1}')
            model.load_state_dict(best_model_state)
            break
            
    return best_model_state

def save_checkpoint(model, optimizer, config, path):
    """安全保存模型检查点"""
    try:
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **config
        }
        torch.save(
            save_dict,
            path,
            _use_new_zipfile_serialization=False
        )
    except Exception as e:
        logging.error(f"保存模型失败: {str(e)}")
        raise

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 配置参数
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_FACTORS = 100
    TIME_FACTORS = 10
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    
    # 加载数据（现在包含时间特征）
    ratings, users, movies = load_data(data_path + 'ratings.csv',
                                     data_path + 'users.csv',
                                     data_path + 'movies.csv')
    
    # 使用时间感知的数据分割
    train_data, val_data, test_data = create_time_aware_split(ratings)
    split_path = data_path + 'split_data'
    save_split_data(train_data, val_data, test_data, split_path)
    
    # 创建数据加载器（包含时间特征）
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型（加入时间参数）
    max_userid = ratings['user_id'].max()
    max_movieid = ratings['movie_id'].max()
    # model = CFModel(max_userid + 1, max_movieid + 1, K_FACTORS, TIME_FACTORS).to(DEVICE)
    model = TimeAwareCFModel(max_userid + 1, max_movieid + 1, K_FACTORS, TIME_FACTORS).to(DEVICE)
    
    criterion = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    # 训练模型
    best_model_state = train(model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)
    
    # 保存模型
    config = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': K_FACTORS,
        'time_factors': TIME_FACTORS,
        'best_model_state': best_model_state,
        'data_split_path': split_path
    }
    save_checkpoint(model, optimizer, config, data_path + 'model_checkpoint.pt')
    
    return model, test_data

if __name__ == "__main__":
    main()