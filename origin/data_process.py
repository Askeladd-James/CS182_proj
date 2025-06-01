import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from pathlib import Path
import logging

data_path = r'./data/'

class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

def load_data(ratings_path, users_path, movies_path):
    """加载并预处理数据"""
    # 读取数据
    ratings = pd.read_csv(ratings_path, sep='\t', encoding='latin-1',
                         usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
    users = pd.read_csv(users_path, sep='\t', encoding='latin-1',
                       usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])
    movies = pd.read_csv(movies_path, sep='\t', encoding='latin-1',
                        usecols=['movie_id', 'title', 'genres'])
    
    return ratings, users, movies

def split_data(ratings, val_size=0.1, test_size=0.2, random_state=42):
    """将数据集分割为训练集、验证集和测试集"""
    # 随机打乱数据
    shuffled_ratings = ratings.sample(frac=1., random_state=random_state)
    
    # 计算分割点
    n_samples = len(shuffled_ratings)
    test_idx = int(n_samples * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    # 分割数据
    train_data = shuffled_ratings[:val_idx]
    val_data = shuffled_ratings[val_idx:test_idx]
    test_data = shuffled_ratings[test_idx:]
    
    return train_data, val_data, test_data

def save_split_data(train_data, val_data, test_data, base_path):
    """保存分割后的数据集"""
    try:
        # 确保目录存在
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        # for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        #     save_path = f"{base_path}_{name}.csv"
        #     data.to_csv(save_path, sep='\t', index=False, encoding='latin-1')
            # logging.info(f"保存{name}数据到: {save_path}")
            
        # 保存分割索引
        split_indices = {
            'train_indices': train_data.index.tolist(),
            'val_indices': val_data.index.tolist(),
            'test_indices': test_data.index.tolist()
        }
        torch.save(split_indices, base_path + '_indices.pt')
        
    except Exception as e:
        logging.error(f"保存数据集失败: {str(e)}")
        raise

def load_test_data(ratings_path, split_indices_path):
    """加载测试数据"""
    try:
        # 使用正确的分隔符加载原始数据
        ratings = pd.read_csv(ratings_path, sep='\t', encoding='latin-1')
        
        # 确保列名正确（移除可能的前缀）
        ratings.columns = ratings.columns.str.strip('\t')
        
        # 加载分割索引
        split_indices = torch.load(split_indices_path + '_indices.pt')
        test_indices = split_indices['test_indices']
        
        # 获取测试数据
        test_data = ratings.loc[test_indices].copy()
        
        # 重置索引
        test_data = test_data.reset_index(drop=True)
        
        # 数据格式验证
        # logging.info("测试数据加载完成，验证数据格式...")
        # logging.info(f"列名: {test_data.columns.tolist()}")
        # logging.info("\n" + str(test_data.head()))
        
        return test_data
        
    except Exception as e:
        logging.error(f"加载测试数据失败: {str(e)}")
        raise
    
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