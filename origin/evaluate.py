import torch
from torch.serialization import add_safe_globals, safe_globals
import numpy as np
import numpy._globals
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
from data_process import load_data, split_data, load_test_data, MovieLensDataset, data_path
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

def load_checkpoint(path):
    """安全加载模型检查点"""
    try:
        # 添加所有需要的安全全局变量
        add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
        ])
        
        # 使用 safe_globals 上下文管理器
        with safe_globals([np.core.multiarray.scalar, np.dtype]):
            return torch.load(
                path,
                weights_only=False,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise

def main():
    # 加载保存的模型
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = load_checkpoint(data_path + 'model_checkpoint.pt')
    
    model = CFModel(
        checkpoint['max_userid'] + 1,
        checkpoint['max_movieid'] + 1,
        checkpoint['k_factors']
    ).to(device)
    model.load_state_dict(checkpoint['best_model_state'])
    
    # 加载测试数据
    test_data = load_test_data(
        data_path + 'ratings.csv',
        checkpoint['data_split_path']
    )

    # print(test_data.head())  # 打印测试数据的前几行以确认加载正确
    
    # 评估模型
    metrics = evaluate_model(model, test_data, device)
    logging.info(f"Test RMSE: {metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {metrics['MAE']:.4f}")

if __name__ == "__main__":
    main()