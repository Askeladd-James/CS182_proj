import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
from data_process import load_data, split_data, MovieLensDataset
from model import CFModel

def evaluate_model(model, test_data, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            movie_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            
            pred = model(user_id, movie_id)
            predictions.append(pred.cpu().item())
            actuals.append(row['rating'])
    
    # 计算评估指标
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse_value = np.mean((predictions - actuals) ** 2)
    rmse = np.sqrt(mse_value)
    mae = mean_absolute_error(actuals, predictions)
    
    return {
        'RMSE': rmse,
        'MAE': mae,
        'MSE': mse_value,
        'predictions': predictions.tolist(),
        'actuals': actuals.tolist()
    }

def get_user_recommendations(model, user_id, ratings, movies, device, n_recommendations=10):
    """获取用户推荐"""
    user_history = ratings[ratings['user_id'] == user_id]
    unseen_movies = movies[~movies['movie_id'].isin(user_history['movie_id'])]
    
    predictions = []
    model.eval()
    with torch.no_grad():
        user_emb_id = ratings[ratings['user_id'] == user_id]['user_emb_id'].iloc[0]
        for _, movie in unseen_movies.iterrows():
            movie_emb_id = ratings[ratings['movie_id'] == movie['movie_id']]['movie_emb_id'].iloc[0]
            user_tensor = torch.LongTensor([user_emb_id]).to(device)
            movie_tensor = torch.LongTensor([movie_emb_id]).to(device)
            pred = model(user_tensor, movie_tensor)
            predictions.append((movie['movie_id'], movie['title'], pred.item()))
    
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:n_recommendations]
    return recommendations

def main():
    # 加载保存的模型
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('model_checkpoint.pt')
    
    model = CFModel(
        checkpoint['max_userid'] + 1,
        checkpoint['max_movieid'] + 1,
        checkpoint['k_factors']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载测试数据
    _, test_data = split_data(load_data('ratings.csv', 'users.csv', 'movies.csv')[0])
    
    # 评估模型
    metrics = evaluate_model(model, test_data, device)
    logging.info(f"Test RMSE: {metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {metrics['MAE']:.4f}")

if __name__ == "__main__":
    main()