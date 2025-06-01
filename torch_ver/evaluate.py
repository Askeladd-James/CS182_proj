import torch
from torch.serialization import add_safe_globals, safe_globals
import numpy as np
import numpy._globals
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
from data_process import load_data, load_test_data, MovieLensDataset, data_path
from model import CFModel, TimeAwareCFModel, SimplifiedTimeAwareCFModel, Models

def evaluate_model(model, test_data, device):
    """评估模型性能"""
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            movie_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            daytime = torch.LongTensor([row['daytime']]).to(device)
            weekend = torch.LongTensor([row['is_weekend']]).to(device)
            year = torch.LongTensor([row['year']]).to(device)
            
            pred = model(user_id, movie_id, daytime, weekend, year)
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

def get_user_recommendations(model, user_id, ratings, movies, device, target_daytime=1, target_weekend=0, target_year=10, n_recommendations=10):
    """
    获取用户推荐（考虑时间特征）
    
    参数:
        model: 训练好的模型
        user_id: 用户ID
        ratings: 评分数据
        movies: 电影数据
        device: 设备
        target_daytime: 目标时间段 (0=凌晨, 1=白天, 2=晚上)
        target_weekend: 是否周末 (0=工作日, 1=周末)
        target_year: 目标年份类别 (0-19)
        n_recommendations: 推荐数量
    """
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
            daytime_tensor = torch.LongTensor([target_daytime]).to(device)
            weekend_tensor = torch.LongTensor([target_weekend]).to(device)
            year_tensor = torch.LongTensor([target_year]).to(device)
            
            pred = model(user_tensor, movie_tensor, daytime_tensor, weekend_tensor, year_tensor)
            predictions.append((movie['movie_id'], movie['title'], pred.item()))
    
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:n_recommendations]
    return recommendations

def analyze_time_preferences(model, user_id, sample_movies, ratings, device):
    """
    分析用户在不同时间的偏好差异
    """
    user_history = ratings[ratings['user_id'] == user_id]
    if user_history.empty:
        return None
    
    user_emb_id = user_history['user_emb_id'].iloc[0]
    results = {}
    
    model.eval()
    with torch.no_grad():
        for movie_id, movie_title in sample_movies:
            movie_ratings = ratings[ratings['movie_id'] == movie_id]
            if movie_ratings.empty:
                continue
                
            movie_emb_id = movie_ratings['movie_emb_id'].iloc[0]
            
            user_tensor = torch.LongTensor([user_emb_id]).to(device)
            movie_tensor = torch.LongTensor([movie_emb_id]).to(device)
            year_tensor = torch.LongTensor([10]).to(device)  # 使用中等年份
            
            time_predictions = {}
            for daytime in [0, 1, 2]:  # 凌晨、白天、晚上
                for weekend in [0, 1]:  # 工作日、周末
                    daytime_tensor = torch.LongTensor([daytime]).to(device)
                    weekend_tensor = torch.LongTensor([weekend]).to(device)
                    
                    pred = model(user_tensor, movie_tensor, daytime_tensor, weekend_tensor, year_tensor)
                    
                    time_desc = f"{'周末' if weekend else '工作日'}_{'凌晨' if daytime==0 else ('白天' if daytime==1 else '晚上')}"
                    time_predictions[time_desc] = pred.item()
            
            results[movie_title] = time_predictions
    
    return results

def load_checkpoint(path):
    """安全加载模型检查点"""
    try:
        add_safe_globals([
            np.core.multiarray.scalar,
            np.dtype,
        ])
        
        with safe_globals([np.core.multiarray.scalar, np.dtype]):
            return torch.load(
                path,
                weights_only=False,
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            )
    except Exception as e:
        logging.error(f"加载模型失败: {str(e)}")
        raise

def create_model_from_checkpoint(checkpoint, device):
    """根据checkpoint中的model_type创建正确的模型"""
    model_type = checkpoint.get('model_type', 'SimplifiedTimeAwareCFModel')  # 默认值
    
    max_userid = checkpoint['max_userid']
    max_movieid = checkpoint['max_movieid']
    k_factors = checkpoint['k_factors']
    time_factors = checkpoint.get('time_factors', 10)  # 向后兼容
    reg_strength = checkpoint.get('reg_strength', 0.01)  # 默认正则化强度
    
    logging.info(f"Creating model of type: {model_type}")
    
    if model_type == 'CFModel':
        model = CFModel(
            max_userid + 1,
            max_movieid + 1,
            k_factors,
            time_factors,
            reg_strength
        ).to(device)
    elif model_type == 'TimeAwareCFModel':
        model = TimeAwareCFModel(
            max_userid + 1,
            max_movieid + 1,
            k_factors,
            time_factors,
            reg_strength
        ).to(device)
    elif model_type == 'SimplifiedTimeAwareCFModel':
        model = SimplifiedTimeAwareCFModel(
            max_userid + 1,
            max_movieid + 1,
            k_factors,
            time_factors,
            reg_strength
        ).to(device)
    else:
        # 如果model_type不存在或不匹配，使用默认模型
        logging.warning(f"Unknown model_type: {model_type}, using SimplifiedTimeAwareCFModel as default")
        model = SimplifiedTimeAwareCFModel(
            max_userid + 1,
            max_movieid + 1,
            k_factors,
            time_factors,
            reg_strength
        ).to(device)
    
    return model

def main():
    # 加载保存的模型
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = Models[2] # 更改这个测试不同模型 ["CF", "TimeAwareCF", "SimplifiedTimeAwareCF"]

    checkpoint = load_checkpoint(data_path + 'model_checkpoint_' + model + '.pt')
    
    # 根据checkpoint动态创建模型
    model = create_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint['best_model_state'])
    
    logging.info(f"Loaded model type: {checkpoint.get('model_type', 'Unknown')}")
    logging.info(f"Model parameters: k_factors={checkpoint['k_factors']}, time_factors={checkpoint.get('time_factors', 10)}")
    
    # 如果有训练历史，显示训练信息
    if 'train_losses' in checkpoint and 'val_losses' in checkpoint:
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']
        final_train_loss = train_losses[-1] if train_losses else 'N/A'
        final_val_loss = val_losses[-1] if val_losses else 'N/A'
        best_val_loss = min(val_losses) if val_losses else 'N/A'
        
        logging.info(f"\n=== Training History ===")
        logging.info(f"Final training loss: {final_train_loss}")
        logging.info(f"Final validation loss: {final_val_loss}")
        logging.info(f"Best validation loss: {best_val_loss}")
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
    
    # 示例：分析用户1在不同时间的推荐差异
    try:
        ratings, users, movies = load_data(data_path + 'ratings.csv',
                                        data_path + 'users.csv',
                                        data_path + 'movies.csv')
        
        user_id = 1
        logging.info(f"\n用户 {user_id} 在工作日白天的推荐:")
        workday_recs = get_user_recommendations(model, user_id, ratings, movies, device, 
                                            target_daytime=1, target_weekend=0, n_recommendations=5)
        for movie_id, title, score in workday_recs:
            logging.info(f"  {title}: {score:.3f}")
        
        logging.info(f"\n用户 {user_id} 在周末晚上的推荐:")
        weekend_recs = get_user_recommendations(model, user_id, ratings, movies, device,
                                            target_daytime=2, target_weekend=1, n_recommendations=5)
        for movie_id, title, score in weekend_recs:
            logging.info(f"  {title}: {score:.3f}")
    except Exception as e:
        logging.error(f"生成推荐时出错: {str(e)}")
    
    # 返回评估结果
    return metrics, model

if __name__ == "__main__":
    main()