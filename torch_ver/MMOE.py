import torch
import torch.nn as nn
import numpy as np

class TwoStageMMoEModel(nn.Module):
    """两阶段MMoE模型 - 优化时序建模"""
    def __init__(self, n_users, m_items, k_factors=80, time_factors=20, reg_strength=0.01, num_experts=4):
        super(TwoStageMMoEModel, self).__init__()
        self.reg_strength = reg_strength
        self.name = "TwoStage_MMoE"
        self.k_factors = k_factors
        self.time_factors = time_factors
        self.num_experts = num_experts
        
        # ============= 共享嵌入层 =============
        self.user_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 时间特征嵌入 - 增加维度
        self.daytime_embedding = nn.Embedding(3, time_factors)
        self.weekend_embedding = nn.Embedding(2, time_factors)
        self.year_embedding = nn.Embedding(20, time_factors)
        
        # 基础偏差项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # ============= 阶段1：改进的时序建模网络 =============
        # 增加输入特征的丰富度
        temporal_input_dim = k_factors + k_factors + time_factors * 3 + 5  # user + item + time + history
        
        # 使用更浅但更宽的网络，添加BatchNorm和残差连接
        self.temporal_feature_network = nn.Sequential(
            nn.Linear(temporal_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # 添加注意力机制
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        # 最终预测层
        self.temporal_prediction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # 用户历史特征增强网络
        self.user_history_enhancer = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # ============= 阶段2：改进的协同过滤网络 =============
        # 参考UMTimeModel的设计，但做优化
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        self.movie_year_bias = nn.Embedding(20, 1)
        self.user_time_bias = nn.Embedding(n_users, 3)
        
        # 用户嵌入处理网络 - 增加残差连接
        self.user_projection = nn.Sequential(
            nn.Linear(k_factors, k_factors),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.user_resize = nn.Sequential(
            nn.Linear(k_factors, k_factors // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 时间特征融合网络
        time_features_dim = time_factors * 2  # daytime + weekend
        combined_dim = k_factors // 2 + time_features_dim
        
        self.cf_time_fc = nn.Sequential(
            nn.Linear(combined_dim, combined_dim),
            nn.BatchNorm1d(combined_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(combined_dim // 2, 1)
        )
        
        # ============= MMoE融合层 =============
        expert_input_dim = 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络
        gate_input_dim = k_factors + time_factors * 3
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 最终输出层
        self.final_layer = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.dropout = nn.Dropout(0.1)  # 减少dropout率
        self._init_weights()
        
        # 训练阶段标志
        self.training_stage = 1
    
    def _init_weights(self):
        """改进的权重初始化"""
        # 使用Xavier初始化主要嵌入层
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        # 偏差项初始化为小值
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        nn.init.constant_(self.global_bias, 0.0)
        
        # 时间特征嵌入使用更小的初始化
        nn.init.normal_(self.daytime_embedding.weight, std=0.02)
        nn.init.normal_(self.weekend_embedding.weight, std=0.02)
        nn.init.normal_(self.year_embedding.weight, std=0.02)
        
        # 偏差项
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)
        nn.init.normal_(self.movie_year_bias.weight, std=0.01)
        nn.init.normal_(self.user_time_bias.weight, std=0.01)
        
        # 线性层使用He初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def forward_temporal_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                              user_history_features=None):
        """
        改进的时序建模前向传播
        """
        batch_size = user_input.size(0)
        
        # 获取嵌入 - 包含物品信息
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)  # 添加物品嵌入
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 改进历史特征处理
        if user_history_features is None:
            # 基于用户ID生成更有意义的伪历史特征
            device = user_input.device
            user_history_features = torch.zeros(batch_size, 5).to(device)
            
            # 使用用户ID作为种子生成差异化的历史特征
            user_ids_float = user_input.float()
            user_history_features[:, 0] = 2.5 + 1.0 * torch.sin(user_ids_float * 0.1)  # 平均评分变化
            user_history_features[:, 1] = 0.5 + 0.3 * torch.cos(user_ids_float * 0.15)  # 标准差变化
            user_history_features[:, 2] = torch.clamp(torch.log(user_ids_float + 1), 1, 100)  # 评分数量
            user_history_features[:, 3] = user_history_features[:, 0] + 0.2 * torch.randn_like(user_ids_float)  # 最近评分
            user_history_features[:, 4] = 0.1 * torch.sin(user_ids_float * 0.05)  # 评分趋势
        
        # 增强历史特征
        enhanced_history = self.user_history_enhancer(user_history_features)
        
        # 组合所有特征 - 包含用户、物品、时间、历史
        combined_features = torch.cat([
            user_embedded,
            item_embedded,  # 添加物品嵌入
            daytime_embedded, 
            weekend_embedded,
            year_embedded,
            user_history_features  # 原始历史特征
        ], dim=1)
        
        # 特征网络处理
        temporal_features = self.temporal_feature_network(combined_features)
        
        # 注意力机制 - 自注意力
        temporal_features_reshaped = temporal_features.unsqueeze(1)  # [batch, 1, 128]
        attended_features, _ = self.temporal_attention(
            temporal_features_reshaped, 
            temporal_features_reshaped, 
            temporal_features_reshaped
        )
        attended_features = attended_features.squeeze(1)  # [batch, 128]
        
        # 残差连接
        final_features = temporal_features + attended_features
        
        # 最终预测
        prediction = self.temporal_prediction(final_features)
        return prediction.squeeze()
    
    def forward_cf_stage(self, user_input, item_input, daytime_input, weekend_input, year_input):
        """改进的CF前向传播"""
        # 获取基础嵌入
        user_base = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_base = self.dropout(user_base)
            item_embedded = self.dropout(item_embedded)
        
        # 用户嵌入处理 - 添加残差连接
        user_projected = self.user_projection(user_base)
        user_projected = user_projected + user_base  # 残差连接
        
        # 核心交互
        base_interaction = torch.sum(user_projected * item_embedded, dim=1)
        
        # 偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        movie_year_bias = self.movie_year_bias(year_input).squeeze()
        
        # 用户时间偏好
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        # 时间特征融合
        user_resized = self.user_resize(user_projected)
        time_features = torch.cat([daytime_embedded, weekend_embedded], dim=1)
        combined_features = torch.cat([user_resized, time_features], dim=1)
        
        time_interaction = self.cf_time_fc(combined_features).squeeze()
        
        # 最终预测
        final_prediction = (base_interaction + user_bias + item_bias + 
                           daytime_bias + weekend_bias + movie_year_bias + 
                           user_daytime_bias + time_interaction + self.global_bias)
        
        return final_prediction
    
    # ... 保留其他方法不变 ...
    
    def get_regularization_loss(self):
        """调整正则化损失"""
        reg_loss = 0
        
        # 降低嵌入层正则化强度
        reg_loss += torch.norm(self.user_embedding.weight) * 0.5
        reg_loss += torch.norm(self.item_embedding.weight) * 0.5
        
        # 时间嵌入正则化
        reg_loss += torch.norm(self.daytime_embedding.weight) * 0.1
        reg_loss += torch.norm(self.weekend_embedding.weight) * 0.1
        reg_loss += torch.norm(self.year_embedding.weight) * 0.1
        
        if self.training_stage == 1:
            # 时序网络正则化 - 降低强度
            reg_loss += sum(torch.norm(param) for param in self.temporal_feature_network.parameters()) * 0.01
            reg_loss += sum(torch.norm(param) for param in self.temporal_prediction.parameters()) * 0.01
        elif self.training_stage == 2:
            # CF网络正则化
            reg_loss += sum(torch.norm(param) for param in self.cf_time_fc.parameters()) * 0.01
        elif self.training_stage == 3:
            # MMoE正则化
            reg_loss += sum(torch.norm(param) for expert in self.experts for param in expert.parameters()) * 0.01
        
        return self.reg_strength * reg_loss