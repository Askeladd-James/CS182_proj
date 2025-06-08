import torch
import torch.nn as nn
import numpy as np
from MMOE import TwoStageMMoEModel
from origin_model import CFModel

Models = ["IndependentTime", "UserTime", "UMTime", "TwoStage_MMoE", "CF"]

class IndependentTimeModel(nn.Module):
    """简化的时间感知协同过滤模型 - 更稳定的版本"""
    def __init__(self, n_users, m_items, k_factors=100, time_factors=20, reg_strength=0.01):
        super(IndependentTimeModel, self).__init__()
        self.name = Models[2]
        self.reg_strength = reg_strength
        
        # 基础嵌入层 - 降低维度防止过拟合
        self.user_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 偏差项 - 关键组件
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 简化的时间特征嵌入
        self.daytime_embedding = nn.Embedding(3, time_factors)  # 3个时间段
        self.weekend_embedding = nn.Embedding(2, time_factors)  # 工作日/周末
        self.year_embedding = nn.Embedding(20, time_factors)    # 年份
        
        # 时间相关的偏差项
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        
        # 简化的融合层
        self.time_fusion = nn.Sequential(
            nn.Linear(3 * time_factors, time_factors),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(time_factors, 1)
        )
        
        # 强化dropout
        self.dropout = nn.Dropout(0.5)
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        # 使用更小的初始化值
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        
        # 偏差项初始化
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        
        # 时间嵌入初始化
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)

    def forward(self, user_input, item_input, daytime_input, weekend_input, year_input):
        # 基础嵌入
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # 应用dropout（仅在训练时）
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 基础交互 - 点积
        base_interaction = torch.sum(user_embedded * item_embedded, dim=1)
        
        # 偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        
        # 时间特征
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        # 时间偏差
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        
        # 融合时间特征
        time_features = torch.cat([daytime_embedded, weekend_embedded, year_embedded], dim=1)
        time_effect = self.time_fusion(time_features).squeeze()
        
        # 最终预测
        prediction = (base_interaction + user_bias + item_bias + 
                     daytime_bias + weekend_bias + time_effect + self.global_bias)
        
        return prediction
    
    def get_regularization_loss(self):
        """计算L2正则化损失"""
        user_reg = torch.norm(self.user_embedding.weight)
        item_reg = torch.norm(self.item_embedding.weight)
        time_reg = (torch.norm(self.daytime_embedding.weight) + 
                   torch.norm(self.weekend_embedding.weight) + 
                   torch.norm(self.year_embedding.weight))
        return self.reg_strength * (user_reg + item_reg + time_reg * 0.1)

# 回来改个名字，这三个都是和时间相关的
class UserTimeModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors, time_factors=10, reg_strength=0.01):
        super(UserTimeModel, self).__init__()
        self.reg_strength = reg_strength  # 添加正则化强度参数
        self.name = Models[0]
        
        # 基础嵌入层 - 降低维度防止过拟合
        self.user_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 关键：添加偏差项（原模型缺少这个重要组件）
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 时间特征的嵌入层
        self.daytime_embedding = nn.Embedding(3, time_factors)  # 0,1,2 三个时间段
        self.weekend_embedding = nn.Embedding(2, time_factors)  # 0,1 工作日/周末
        self.year_embedding = nn.Embedding(20, time_factors)    # 年份嵌入
        
        # 时间相关的偏差项（更细粒度的时间偏差）
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        self.year_bias = nn.Embedding(20, 1)
        
        # 用户在不同时间段的偏好变化
        self.user_time_bias = nn.Embedding(n_users, 3)  # 用户在3个时间段的偏好偏置
        
        # 简化并强化正则化的全连接层
        self.fc = nn.Sequential(
            nn.Linear(3 * time_factors, time_factors),  # 只处理时间特征
            nn.ReLU(),
            nn.Dropout(0.4),  # 增强dropout
            nn.Linear(time_factors, 1)
        )
        
        # 主要的dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 改进权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化 - 使用更小的初始化值"""
        # 主要嵌入层使用较小的标准差
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        
        # 偏差项初始化为更小的值
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        
        # 时间特征嵌入
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
        
        # 时间偏差项
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)
        nn.init.normal_(self.year_bias.weight, std=0.01)
        nn.init.normal_(self.user_time_bias.weight, std=0.01)

    def forward(self, user_input, item_input, daytime_input, weekend_input, year_input):
        # 基础嵌入
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # 应用dropout（仅在训练时）
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 核心交互：用户-物品点积（类似矩阵分解）
        base_interaction = torch.sum(user_embedded * item_embedded, dim=1)
        
        # 基础偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        
        # 时间特征嵌入
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        # 时间偏差项
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        year_bias = self.year_bias(year_input).squeeze()
        
        # 用户在特定时间段的偏好偏置
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        # 融合时间特征（只通过FC层处理时间特征，不混合用户-物品交互）
        time_features = torch.cat([
            daytime_embedded,
            weekend_embedded,
            year_embedded
        ], dim=1)
        
        # 时间特征的非线性变换
        time_effect = self.fc(time_features).squeeze()
        
        # 最终预测：基础交互 + 各种偏差项 + 时间效应
        final_rating = (base_interaction + user_bias + item_bias + 
                       daytime_bias + weekend_bias + year_bias + 
                       user_daytime_bias + time_effect + self.global_bias)
        
        return final_rating
    
    def get_regularization_loss(self):
        """计算L2正则化损失"""
        user_reg = torch.norm(self.user_embedding.weight)
        item_reg = torch.norm(self.item_embedding.weight)
        time_reg = (torch.norm(self.daytime_embedding.weight) + 
                   torch.norm(self.weekend_embedding.weight) + 
                   torch.norm(self.year_embedding.weight))
        return self.reg_strength * (user_reg + item_reg + time_reg * 0.1)
    
    def rate(self, user_id, item_id, daytime=1, weekend=0, year=10):
        """预测单个用户对单个物品的评分（兼容接口）"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            daytime_tensor = torch.LongTensor([daytime])
            weekend_tensor = torch.LongTensor([weekend])
            year_tensor = torch.LongTensor([year])
            prediction = self.forward(user_tensor, item_tensor, daytime_tensor, 
                                    weekend_tensor, year_tensor)
            return prediction.item()

class UMTimeModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors, time_factors=20, reg_strength=0.01):
        super(UMTimeModel, self).__init__()
        self.reg_strength = reg_strength
        self.name = Models[1]
        self.k_factors = k_factors
        self.time_factors = time_factors
        
        # 基础嵌入层 - 保持完整维度
        self.user_base_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 偏差项 - 与UserTimeModel保持一致
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # 🔧 修复2: 补全时间特征，与UserTimeModel对齐
        self.daytime_embedding = nn.Embedding(3, time_factors)
        self.weekend_embedding = nn.Embedding(2, time_factors)
        self.year_embedding = nn.Embedding(20, time_factors)  # 补充年份嵌入
        
        # 🔧 修复3: 完整的时间偏差项
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        self.year_bias = nn.Embedding(20, 1)  # 补充年份偏差
        
        # 🔧 修复4: 添加用户时间交互偏差（参考UserTimeModel）
        self.user_time_bias = nn.Embedding(n_users, 3)  # 用户在不同时间段的偏好
        
        # 🔧 修复5: 简化FC层，移除复杂的维度投影
        # 输入维度: k_factors(用户演化) + 3*time_factors(时间特征)
        self.time_fusion = nn.Sequential(
            nn.Linear(k_factors + 3 * time_factors, time_factors * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(time_factors * 2, 1)
        )
        
        # 主要dropout层
        self.dropout = nn.Dropout(0.3)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        # 基础嵌入层
        nn.init.normal_(self.user_base_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        
        # 偏差项
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        
        # 时间特征嵌入
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
        
        # 时间偏差项
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)
        nn.init.normal_(self.year_bias.weight, std=0.01)
        nn.init.normal_(self.user_time_bias.weight, std=0.01)
    
    def forward(self, user_input, item_input, daytime_input, weekend_input, 
                year_input, user_historical_features=None):
        
        # 🔧 修复6: 改进用户演化建模
        user_base = self.user_base_embedding(user_input)
        
        if self.training:
            user_base = self.dropout(user_base)
        
        # 用户偏好演化建模
        
        user_embedded = user_base
        
        # 物品嵌入
        item_embedded = self.item_embedding(item_input)
        if self.training:
            item_embedded = self.dropout(item_embedded)
        
        # 🔧 修复7: 核心交互 - 直接点积，避免维度转换
        base_interaction = torch.sum(user_embedded * item_embedded, dim=1)
        
        # 基础偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        
        # 🔧 修复8: 完整的时间特征处理
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        # 时间偏差项
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        year_bias = self.year_bias(year_input).squeeze()
        
        # 🔧 修复9: 用户时间交互偏差
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        # 🔧 修复10: 改进的时间特征融合
        # 结合用户演化特征和完整时间特征
        time_features = torch.cat([
            daytime_embedded,
            weekend_embedded,
            year_embedded
        ], dim=1)
        
        # 将用户演化特征与时间特征结合
        combined_features = torch.cat([user_embedded, time_features], dim=1)
        time_interaction = self.time_fusion(combined_features).squeeze()
        
        # 🔧 修复11: 最终预测 - 参考UserTimeModel的成功模式
        final_rating = (base_interaction + user_bias + item_bias + 
                       daytime_bias + weekend_bias + year_bias + 
                       user_daytime_bias + time_interaction + self.global_bias)
        
        return final_rating
    
    def get_regularization_loss(self):
        """计算L2正则化损失"""
        user_reg = torch.norm(self.user_base_embedding.weight)
        item_reg = torch.norm(self.item_embedding.weight)
        time_reg = (torch.norm(self.daytime_embedding.weight) + 
                   torch.norm(self.weekend_embedding.weight) + 
                   torch.norm(self.year_embedding.weight))
        
        return self.reg_strength * (user_reg + item_reg + time_reg * 0.1)
    
    def rate(self, user_id, item_id, daytime=1, weekend=0, year=10):
        """预测单个用户对单个物品的评分（兼容接口）"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            daytime_tensor = torch.LongTensor([daytime])
            weekend_tensor = torch.LongTensor([weekend])
            year_tensor = torch.LongTensor([year])
            prediction = self.forward(user_tensor, item_tensor, daytime_tensor, 
                                    weekend_tensor, year_tensor)
            return prediction.item()