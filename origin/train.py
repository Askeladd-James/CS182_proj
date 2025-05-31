import torch
import torch.nn as nn
import torch.optim as optim
from data_process import load_data, split_data, save_split_data, MovieLensDataset, data_path
from model import CFModel
from torch.utils.data import DataLoader
import logging
from pathlib import Path

def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=30, patience=3):
    """训练模型，包含早停机制"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for user, movie, rating in train_loader:
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, movie)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for user, movie, rating in val_loader:
                user, movie, rating = user.to(device), movie.to(device), rating.to(device)
                prediction = model(user, movie)
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
            _use_new_zipfile_serialization=False  # 使用旧的序列化格式以提高兼容性
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
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    
    # 加载数据
    ratings, users, movies = load_data(data_path + 'ratings.csv',
                                     data_path + 'users.csv',
                                     data_path + 'movies.csv')
    train_data, val_data, test_data = split_data(ratings)
    split_path = data_path + 'split_data'
    save_split_data(train_data, val_data, test_data, split_path)
    
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

    # print(test_data.head())  # 打印测试数据的前几行以确认加载正确
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    max_userid = ratings['user_id'].max()
    max_movieid = ratings['movie_id'].max()
    model = CFModel(max_userid + 1, max_movieid + 1, K_FACTORS).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    best_model_state = train(model, train_loader, val_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)
    
    # 保存模型
    config = {
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': K_FACTORS,
        'best_model_state': best_model_state,
        'data_split_path': split_path  # 添加数据分割信息
    }
    save_checkpoint(model, optimizer, config, data_path + 'model_checkpoint.pt')
    
    return model, test_data

if __name__ == "__main__":
    main()