import torch
from torch.serialization import add_safe_globals, safe_globals
import numpy as np
import numpy._globals
import pandas as pd
from sklearn.metrics import mean_absolute_error
import logging
from data_process import load_data, load_test_data, data_path
from MMOE import TwoStageMMoEModel
from MMOE_train import prepare_sequential_training_data, get_lstm_predictions_for_data, get_cf_predictions_for_data

def evaluate_mmoe_model(model, test_data, device):
    """评估MMoE模型性能"""
    model.eval()
    model.set_training_stage(3)  # 设置为MMoE融合阶段
    
    predictions = []
    actuals = []
    
    # 需要准备时序数据来获取LSTM预测
    # 这里我们使用测试数据本身来构建序列（在实际应用中可能需要历史数据）
    sequential_data = prepare_sequential_training_data(test_data)
    
    with torch.no_grad():
        for _, row in test_data.iterrows():
            user_id = torch.LongTensor([row['user_emb_id']]).to(device)
            movie_id = torch.LongTensor([row['movie_emb_id']]).to(device)
            daytime = torch.LongTensor([row['daytime']]).to(device)
            weekend = torch.LongTensor([row['is_weekend']]).to(device)
            year = torch.LongTensor([row['year']]).to(device)
            
            # 获取LSTM和CF的预测
            lstm_pred = get_single_lstm_prediction(model, row, sequential_data, device)
            cf_pred = get_single_cf_prediction(model, row, device)
            
            lstm_pred_tensor = torch.FloatTensor([lstm_pred]).to(device)
            cf_pred_tensor = torch.FloatTensor([cf_pred]).to(device)
            
            # MMoE融合预测
            pred = model(user_id, movie_id, daytime, weekend, year, lstm_pred_tensor, cf_pred_tensor)
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

def get_single_lstm_prediction(model, row, sequential_data, device):
    """为单个样本获取LSTM预测"""
    model.set_training_stage(1)
    
    user_id = row['user_emb_id']
    
    # 查找该用户的序列数据
    user_sequences = [item for item in sequential_data if item['user_id'] == user_id]
    
    if user_sequences:
        # 使用最新的序列
        latest_sequence = user_sequences[-1]
        sequence = latest_sequence['sequence']
        
        # 构建特征序列
        features = []
        for _, seq_row in sequence.iterrows():
            feature_vector = [
                seq_row['user_emb_id'],
                seq_row['daytime'],
                seq_row['is_weekend'],
                seq_row['year'],
                seq_row['rating']
            ]
            features.append(feature_vector)
        
        # 填充到固定长度
        max_seq_len = 20
        if len(features) > max_seq_len:
            features = features[-max_seq_len:]
        else:
            while len(features) < max_seq_len:
                features.insert(0, [0, 0, 0, 0, 0])
        
        sequence_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(sequence_tensor)
        return pred.item()
    else:
        return 3.0  # 默认评分

def get_single_cf_prediction(model, row, device):
    """为单个样本获取CF预测"""
    model.set_training_stage(2)
    
    user_tensor = torch.LongTensor([row['user_emb_id']]).to(device)
    item_tensor = torch.LongTensor([row['movie_emb_id']]).to(device)
    daytime_tensor = torch.LongTensor([row['daytime']]).to(device)
    weekend_tensor = torch.LongTensor([row['is_weekend']]).to(device)
    year_tensor = torch.LongTensor([row['year']]).to(device)
    
    with torch.no_grad():
        pred = model(user_tensor, item_tensor, daytime_tensor, weekend_tensor, year_tensor)
    return pred.item()

def get_user_recommendations_mmoe(model, user_id, ratings, movies, device, 
                                 target_daytime=1, target_weekend=0, target_year=10, 
                                 n_recommendations=10):
    """使用MMoE模型获取用户推荐"""
    user_history = ratings[ratings['user_id'] == user_id]
    if user_history.empty:
        logging.warning(f"用户 {user_id} 没有历史评分记录")
        return []
    
    unseen_movies = movies[~movies['movie_id'].isin(user_history['movie_id'])]
    
    predictions = []
    model.eval()
    model.set_training_stage(3)
    
    # 准备用户的序列数据
    user_data = ratings[ratings['user_id'] == user_id].copy()
    sequential_data = prepare_sequential_training_data(user_data)
    
    with torch.no_grad():
        user_emb_id = user_history['user_emb_id'].iloc[0]
        
        for _, movie in unseen_movies.iterrows():
            movie_ratings = ratings[ratings['movie_id'] == movie['movie_id']]
            if movie_ratings.empty:
                continue
                
            movie_emb_id = movie_ratings['movie_emb_id'].iloc[0]
            
            # 创建一个虚拟的row来获取预测
            virtual_row = {
                'user_emb_id': user_emb_id,
                'movie_emb_id': movie_emb_id,
                'daytime': target_daytime,
                'is_weekend': target_weekend,
                'year': target_year
            }
            
            # 获取LSTM和CF预测
            lstm_pred = get_single_lstm_prediction(model, virtual_row, sequential_data, device)
            cf_pred = get_single_cf_prediction(model, virtual_row, device)
            
            # MMoE融合预测
            user_tensor = torch.LongTensor([user_emb_id]).to(device)
            movie_tensor = torch.LongTensor([movie_emb_id]).to(device)
            daytime_tensor = torch.LongTensor([target_daytime]).to(device)
            weekend_tensor = torch.LongTensor([target_weekend]).to(device)
            year_tensor = torch.LongTensor([target_year]).to(device)
            lstm_pred_tensor = torch.FloatTensor([lstm_pred]).to(device)
            cf_pred_tensor = torch.FloatTensor([cf_pred]).to(device)
            
            pred = model(user_tensor, movie_tensor, daytime_tensor, weekend_tensor, year_tensor, 
                        lstm_pred_tensor, cf_pred_tensor)
            predictions.append((movie['movie_id'], movie['title'], pred.item()))
    
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:n_recommendations]
    return recommendations

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

def create_mmoe_model_from_checkpoint(checkpoint, device):
    """根据checkpoint创建MMoE模型"""
    max_userid = checkpoint['max_userid']
    max_movieid = checkpoint['max_movieid']
    k_factors = checkpoint['k_factors']
    time_factors = checkpoint.get('time_factors', 20)
    reg_strength = checkpoint.get('reg_strength', 0.01)
    num_experts = checkpoint.get('num_experts', 4)
    
    logging.info(f"Creating MMoE model with {num_experts} experts")
    
    model = TwoStageMMoEModel(
        max_userid + 1,
        max_movieid + 1,
        k_factors,
        time_factors,
        reg_strength,
        num_experts
    ).to(device)
    
    return model

def main():
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载保存的MMoE模型
    model_name = "TwoStage_MMoE"
    checkpoint = load_checkpoint(data_path + 'model/' + 'model_checkpoint_' + model_name + '.pt')
    
    # 创建模型
    model = create_mmoe_model_from_checkpoint(checkpoint, device)
    model.load_state_dict(checkpoint['best_model_state'])
    
    logging.info(f"Loaded MMoE model with {checkpoint.get('num_experts', 4)} experts")
    logging.info(f"Model parameters: k_factors={checkpoint['k_factors']}, time_factors={checkpoint.get('time_factors', 20)}")
    
    # 加载测试数据
    test_data = load_test_data(
        data_path + 'ratings.csv',
        checkpoint['data_split_path']
    )
    
    logging.info(f"Loaded test data with {len(test_data)} samples")
    
    # 评估模型
    metrics = evaluate_mmoe_model(model, test_data, device)
    logging.info(f"\n=== MMoE Model Performance ===")
    logging.info(f"Test MSE: {metrics['MSE']:.4f}")
    logging.info(f"Test RMSE: {metrics['RMSE']:.4f}")
    logging.info(f"Test MAE: {metrics['MAE']:.4f}")
    
    # 示例：生成推荐
    try:
        ratings, users, movies = load_data(data_path + 'ratings.csv',
                                        data_path + 'users.csv',
                                        data_path + 'movies.csv')
        
        user_id = 1
        logging.info(f"\n用户 {user_id} 的MMoE推荐 (工作日白天):")
        workday_recs = get_user_recommendations_mmoe(model, user_id, ratings, movies, device, 
                                                   target_daytime=1, target_weekend=0, n_recommendations=5)
        for movie_id, title, score in workday_recs:
            logging.info(f"  {title}: {score:.3f}")
        
        logging.info(f"\n用户 {user_id} 的MMoE推荐 (周末晚上):")
        weekend_recs = get_user_recommendations_mmoe(model, user_id, ratings, movies, device,
                                                   target_daytime=2, target_weekend=1, n_recommendations=5)
        for movie_id, title, score in weekend_recs:
            logging.info(f"  {title}: {score:.3f}")
    except Exception as e:
        logging.error(f"生成推荐时出错: {str(e)}")
    
    return metrics, model

if __name__ == "__main__":
    main()