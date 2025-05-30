import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

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

def split_data(ratings, test_size=0.2, random_state=42):
    """分割训练集和测试集"""
    # 随机打乱数据
    shuffled_ratings = ratings.sample(frac=1., random_state=random_state)
    
    # 分割数据
    train_size = int(len(shuffled_ratings) * (1 - test_size))
    train_data = shuffled_ratings[:train_size]
    test_data = shuffled_ratings[train_size:]
    
    return train_data, test_data