import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import logging
from datetime import datetime
import pytz

data_path = r'./data/'

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings, daytime, weekend, years):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)
        self.daytime = torch.LongTensor(daytime)
        self.weekend = torch.LongTensor(weekend)
        self.years = torch.LongTensor(years)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return (self.users[idx], self.movies[idx], self.ratings[idx], 
                self.daytime[idx], self.weekend[idx], self.years[idx])

def load_data(ratings_path, users_path, movies_path):
    """加载并预处理数据"""
    # 读取数据（需要包含timestamp列）
    ratings = pd.read_csv(ratings_path, sep='\t', encoding='latin-1')
    users = pd.read_csv(users_path, sep='\t', encoding='latin-1',
                       usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
    movies = pd.read_csv(movies_path, sep='\t', encoding='latin-1',
                        usecols=['movie_id', 'title', 'genres'])
    
    # 添加时间特征
    ratings['daytime'] = ratings['timestamp'].apply(daytime)
    ratings['is_weekend'] = ratings['timestamp'].apply(is_weekend)
    ratings['year'] = ratings['timestamp'].apply(get_year_category)
    
    return ratings, users, movies

def daytime(timestamp):
    """判断时间段"""
    try:
        us_tz = pytz.timezone('US/Eastern')
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        dt = dt.astimezone(us_tz)
        hour = dt.hour
        
        if 0 <= hour < 6:
            return 0  # 凌晨
        elif 6 <= hour < 18:
            return 1  # 白天
        else:
            return 2  # 晚上
    except:
        return 1

def is_weekend(timestamp):
    """判断是否周末"""
    try:
        us_tz = pytz.timezone('US/Eastern')
        dt = datetime.fromtimestamp(timestamp, tz=pytz.UTC)
        dt = dt.astimezone(us_tz)
        weekday = dt.weekday()
        return 1 if weekday >= 5 else 0
    except:
        return 0

def get_year_category(timestamp):
    """将时间戳转换为年份类别（相对于数据集起始年份）"""
    try:
        dt = datetime.fromtimestamp(timestamp)
        year = dt.year
        # 假设数据集从1995年开始，映射到0-19的范围
        base_year = 1995
        year_category = max(0, min(19, year - base_year))
        return year_category
    except:
        return 0
    
    
def create_time_aware_split(ratings, test_ratio=0.1, val_ratio=0.1, random_state=42):
    """
    创建时间感知的数据分割
    对每个用户随机抽取一定比例的评分作为测试集
    """
    np.random.seed(random_state)
    
    train_data = []
    val_data = []
    test_data = []
    
    # 按用户分组
    for user_id in ratings['user_id'].unique():
        user_ratings = ratings[ratings['user_id'] == user_id].copy()
        
        # 按时间排序
        user_ratings = user_ratings.sort_values('timestamp')
        
        n_ratings = len(user_ratings)
        if n_ratings < 3:  # 如果用户评分太少，全部作为训练集
            train_data.append(user_ratings)
            continue
        
        # 随机选择测试集索引
        test_size = max(1, int(n_ratings * test_ratio))
        val_size = max(1, int(n_ratings * val_ratio))
        
        # 随机选择索引
        all_indices = list(range(n_ratings))
        test_indices = np.random.choice(all_indices, test_size, replace=False)
        remaining_indices = [i for i in all_indices if i not in test_indices]
        
        if len(remaining_indices) > val_size:
            val_indices = np.random.choice(remaining_indices, val_size, replace=False)
            train_indices = [i for i in remaining_indices if i not in val_indices]
        else:
            val_indices = remaining_indices[:val_size]
            train_indices = remaining_indices[val_size:]
        
        # 分割数据 - 修复这里的问题
        test_data.append(user_ratings.iloc[test_indices])
        if len(val_indices) > 0:  # 修改：检查长度而不是直接用 if val_indices
            val_data.append(user_ratings.iloc[val_indices])
        if len(train_indices) > 0:  # 修改：检查长度而不是直接用 if train_indices
            train_data.append(user_ratings.iloc[train_indices])
    
    # 合并所有用户的数据
    train_df = pd.concat(train_data, ignore_index=True) if train_data else pd.DataFrame()
    val_df = pd.concat(val_data, ignore_index=True) if val_data else pd.DataFrame()
    test_df = pd.concat(test_data, ignore_index=True) if test_data else pd.DataFrame()
    
    return train_df, val_df, test_df

def save_split_data(train_data, val_data, test_data, split_path):
    """保存分割后的数据"""
    Path(split_path).mkdir(parents=True, exist_ok=True)
    train_data.to_csv(f'{split_path}/train.csv', sep='\t', index=False)
    val_data.to_csv(f'{split_path}/val.csv', sep='\t', index=False)
    test_data.to_csv(f'{split_path}/test.csv', sep='\t', index=False)

def load_test_data(ratings_path, split_path):
    """加载测试数据"""
    test_data = pd.read_csv(f'{split_path}/test.csv', sep='\t')
    # 确保测试数据包含必要的时间特征
    if 'daytime' not in test_data.columns:
        test_data['daytime'] = test_data['timestamp'].apply(daytime)
    if 'is_weekend' not in test_data.columns:
        test_data['is_weekend'] = test_data['timestamp'].apply(is_weekend)
    if 'year' not in test_data.columns:
        test_data['year'] = test_data['timestamp'].apply(get_year_category)
    return test_data

def load_existing_split_data(split_path):
    """加载已存在的分割数据"""
    try:
        train_data = pd.read_csv(f'{split_path}/train.csv', sep='\t')
        val_data = pd.read_csv(f'{split_path}/val.csv', sep='\t')
        test_data = pd.read_csv(f'{split_path}/test.csv', sep='\t')
        
        # 确保包含必要的时间特征列
        for data, name in [(train_data, 'train'), (val_data, 'val'), (test_data, 'test')]:
            if 'daytime' not in data.columns or 'is_weekend' not in data.columns or 'year' not in data.columns:
                logging.warning(f"{name} data missing time features, will recreate split")
                return None, None, None
        
        logging.info(f'Loaded existing split data from {split_path}')
        return train_data, val_data, test_data
    except Exception as e:
        logging.info(f'Failed to load existing split data: {str(e)}')
        return None, None, None

def check_split_data_exists(split_path):
    """检查分割数据是否存在且完整"""
    required_files = ['train.csv', 'val.csv', 'test.csv']
    split_dir = Path(split_path)
    
    if not split_dir.exists():
        return False
    
    for file in required_files:
        file_path = split_dir / file
        if not file_path.exists() or file_path.stat().st_size == 0:
            return False
    
    return True