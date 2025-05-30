import torch
import torch.nn as nn
import torch.optim as optim
from data_process import load_data, split_data, MovieLensDataset
from model import CFModel
from torch.utils.data import DataLoader
import logging
from pathlib import Path

def train(model, train_loader, criterion, optimizer, device, num_epochs=30):
    """训练模型"""
    logging.info(f'CPU/GPU: {device}')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for user, movie, rating in train_loader:
            user, movie, rating = user.to(device), movie.to(device), rating.to(device)
            
            optimizer.zero_grad()
            prediction = model(user, movie)
            loss = criterion(prediction, rating)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch {epoch+1}/{num_epochs}, Average loss: {avg_loss:.4f}')

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
    ratings, users, movies = load_data('ratings.csv', 'users.csv', 'movies.csv')
    train_data, test_data = split_data(ratings)
    
    # 创建数据加载器
    train_dataset = MovieLensDataset(
        train_data['user_emb_id'].values,
        train_data['movie_emb_id'].values,
        train_data['rating'].values
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 初始化模型
    max_userid = ratings['user_id'].max()
    max_movieid = ratings['movie_id'].max()
    model = CFModel(max_userid + 1, max_movieid + 1, K_FACTORS).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    train(model, train_loader, criterion, optimizer, DEVICE, NUM_EPOCHS)
    
    # 保存模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'max_userid': max_userid,
        'max_movieid': max_movieid,
        'k_factors': K_FACTORS
    }, 'model_checkpoint.pt')
    
    return model, test_data

if __name__ == "__main__":
    main()