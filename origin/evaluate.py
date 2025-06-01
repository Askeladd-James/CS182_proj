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
    if user_history.empty:
        logging.warning(f"用户 {user_id} 没有历史评分记录")
        return []
        
    unseen_movies = movies[~movies['movie_id'].isin(user_history['movie_id'])]
    
    predictions = []
    model.eval()
    with torch.no_grad():
        user_emb_id = user_history['user_emb_id'].iloc[0]
        
        for _, movie in unseen_movies.iterrows():
            # 查找电影的嵌入ID
            movie_ratings = ratings[ratings['movie_id'] == movie['movie_id']]
            if movie_ratings.empty:
                continue
                
            movie_emb_id = movie_ratings['movie_emb_id'].iloc[0]
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
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = load_checkpoint(data_path + 'model_checkpoint_origin.pt')
    
    # 使用checkpoint中的参数创建模型
    model = CFModel(
        checkpoint['max_userid'] + 1,
        checkpoint['max_movieid'] + 1,
        checkpoint['k_factors'],
        checkpoint.get('reg_strength', 0.0001)  # 使用保存的正则化强度，如果没有则用默认值
    ).to(device)
    model.load_state_dict(checkpoint['best_model_state'])
    
    logging.info(f"Loaded model with parameters:")
    logging.info(f"  Max user ID: {checkpoint['max_userid']}")
    logging.info(f"  Max movie ID: {checkpoint['max_movieid']}")
    logging.info(f"  K factors: {checkpoint['k_factors']}")
    logging.info(f"  Regularization strength: {checkpoint.get('reg_strength', 0.0001)}")
    
    # 如果有训练历史，显示训练信息
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        
        if train_losses and val_losses:
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            best_val_loss = min(val_losses)
            
            logging.info(f"\n=== Training History ===")
            logging.info(f"Final training loss: {final_train_loss:.4f}")
            logging.info(f"Final validation loss: {final_val_loss:.4f}")
            logging.info(f"Best validation loss: {best_val_loss:.4f}")
            logging.info(f"Training epochs completed: {len(train_losses)}")
    
    # 加载测试数据
    test_data = load_test_data(
        data_path + 'ratings.csv',
        checkpoint['data_split_path']
    )

    logging.info(f"Loaded test data with {len(test_data)} samples")
    
    # 评估模型
    metrics = evaluate_model(model, test_data, device)
    logging.info(f"\n=== Model Performance ===")
    logging.info(f"Test MSE: {metrics['MSE']:.4f}")
    logging.info(f"Test RMSE: {metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {metrics['MAE']:.4f}")
    
    # 示例：为用户1生成推荐
    try:
        ratings, users, movies = load_data(data_path + 'ratings.csv',
                                         data_path + 'users.csv',
                                         data_path + 'movies.csv')
        
        user_id = 1
        logging.info(f"\n=== 用户 {user_id} 的推荐列表 ===")
        recommendations = get_user_recommendations(model, user_id, ratings, movies, device, n_recommendations=10)
        
        if recommendations:
            for i, (movie_id, title, score) in enumerate(recommendations, 1):
                logging.info(f"{i:2d}. {title}: {score:.3f}")
        else:
            logging.warning(f"无法为用户 {user_id} 生成推荐")
            
    except Exception as e:
        logging.error(f"生成推荐时出错: {str(e)}")
    
    return metrics, model

if __name__ == "__main__":
    main()