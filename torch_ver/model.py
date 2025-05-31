import torch
import torch.nn as nn
import numpy as np

class CFModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors, time_factors=10):
        super(CFModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 时间特征的嵌入层
        self.daytime_embedding = nn.Embedding(3, time_factors)  # 0,1,2 三个时间段
        self.weekend_embedding = nn.Embedding(2, time_factors)  # 0,1 工作日/周末
        
        # 年份嵌入 - 捕捉不同年代的评分模式
        self.year_embedding = nn.Embedding(20, time_factors)  # 假设覆盖20年
        
        # 用户在不同时间段的偏好变化
        self.user_time_bias = nn.Embedding(n_users, 3)  # 用户在3个时间段的偏好偏置
        
        # 全连接层融合特征
        self.fc = nn.Sequential(
            nn.Linear(k_factors + 3 * time_factors, k_factors // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(k_factors // 2, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
        nn.init.normal_(self.user_time_bias.weight, std=0.01)

    def forward(self, user_input, item_input, daytime_input, weekend_input, year_input):
        # 基础嵌入
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # 时间特征嵌入
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        # 用户-物品交互
        user_item_interaction = user_embedded * item_embedded
        
        # 用户在特定时间段的偏好偏置
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        # 融合所有特征
        combined_features = torch.cat([
            user_item_interaction,
            daytime_embedded,
            weekend_embedded,
            year_embedded
        ], dim=1)
        
        # 预测基础评分
        base_rating = self.fc(combined_features).squeeze()
        
        # 加上时间偏置
        final_rating = base_rating + user_daytime_bias
        
        return final_rating
    


class TimeAwareCFModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors, time_factors=10):
        super(TimeAwareCFModel, self).__init__()
        
        # 基础嵌入
        self.user_base_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 用户偏好随时间的演化
        self.user_evolution = nn.LSTM(k_factors, k_factors, batch_first=True)
        
        # 时间特征
        self.daytime_embedding = nn.Embedding(3, time_factors)
        self.weekend_embedding = nn.Embedding(2, time_factors)
        
        # 电影年代效应 - 不同年代电影的基础受欢迎程度
        self.movie_year_bias = nn.Embedding(20, 1)
        
        self.fc = nn.Sequential(
            nn.Linear(k_factors + 2 * time_factors, k_factors // 2),
            nn.ReLU(),
            nn.Linear(k_factors // 2, 1)
        )
    
    def forward(self, user_input, item_input, daytime_input, weekend_input, 
                year_input, user_historical_features=None):
        """
        user_historical_features: 用户历史行为的时序特征（可选）
        """
        # 获取用户基础嵌入
        user_base = self.user_base_embedding(user_input)
        
        # 如果有历史特征，通过LSTM建模用户偏好演化
        if user_historical_features is not None:
            user_evolved, _ = self.user_evolution(user_historical_features)
            user_embedded = user_evolved[:, -1, :]  # 取最后一个时间步
        else:
            user_embedded = user_base
        
        # 物品嵌入
        item_embedded = self.item_embedding(item_input)
        
        # 时间特征
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        
        # 电影年代偏置
        movie_year_bias = self.movie_year_bias(year_input).squeeze()
        
        # 用户-物品交互
        interaction = user_embedded * item_embedded
        
        # 融合特征
        combined = torch.cat([interaction, daytime_embedded, weekend_embedded], dim=1)
        
        # 预测评分
        rating = self.fc(combined).squeeze() + movie_year_bias
        
        return rating