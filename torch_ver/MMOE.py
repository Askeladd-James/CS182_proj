import torch
import torch.nn as nn
import numpy as np

class TwoStageMMoEModel(nn.Module):
    """两阶段MMoE模型 - CF层参考UMTimeModel"""
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
        
        # 时间特征嵌入 - 简化
        self.daytime_embedding = nn.Embedding(3, time_factors // 2)  # 减少维度
        self.weekend_embedding = nn.Embedding(2, time_factors // 2)
        self.year_embedding = nn.Embedding(20, time_factors // 2)
        
        # 基础偏差项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # ============= 阶段1：简化的时序建模网络 =============
        # 用更简单的方式替代复杂的LSTM序列处理
        # 使用用户的历史评分统计特征而不是完整序列
        temporal_input_dim = k_factors + time_factors // 2 * 3 + 5  # 用户嵌入 + 时间特征 + 历史统计特征
        
        self.temporal_network = nn.Sequential(
            nn.Linear(temporal_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 用户历史特征嵌入 - 替代LSTM
        self.user_history_network = nn.Sequential(
            nn.Linear(k_factors, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # ============= 阶段2：改进的协同过滤网络（参考UMTimeModel） =============
        
        # 参考UMTimeModel：添加时间相关的偏差项
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        self.movie_year_bias = nn.Embedding(20, 1)  # 电影年代效应
        
        # 参考UMTimeModel：用户在不同时间段的偏好变化
        self.user_time_bias = nn.Embedding(n_users, 3)  # 用户在3个时间段的偏好偏置
        
        # 参考UMTimeModel：用户嵌入的维度调整网络
        self.user_projection = nn.Linear(k_factors, k_factors)  # 用户嵌入维度投影
        self.user_resize = nn.Linear(k_factors, k_factors // 2)   # 用户嵌入维度调整
        
        # 参考UMTimeModel：分离时间特征处理的FC层
        time_features_dim = time_factors // 2 * 2  # daytime + weekend (不包括year，因为year用作偏差)
        combined_dim = k_factors // 2 + time_features_dim
        
        self.cf_time_fc = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.4),  # 增强dropout
            nn.Linear(combined_dim // 2, 1)
        )
        
        # ============= MMoE融合层（简化） =============
        expert_input_dim = 2  # 时序网络输出 + CF输出
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, 16),  # 减少专家网络大小
                nn.ReLU(),
                nn.Linear(16, 8)
            ) for _ in range(num_experts)
        ])
        
        # 门控网络（简化）
        gate_input_dim = k_factors + time_factors // 2 * 3
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 16),  # 减少门控网络大小
            nn.ReLU(),
            nn.Linear(16, num_experts),
            nn.Softmax(dim=1)
        )
        
        # 最终输出层
        self.final_layer = nn.Sequential(
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
        
        self.dropout = nn.Dropout(0.3)
        self._init_weights()
        
        # 训练阶段标志
        self.training_stage = 1
    
    def set_training_stage(self, stage):
        """设置训练阶段"""
        self.training_stage = stage
        
        if stage == 1:
            # 阶段1：只训练时序网络
            self._freeze_parameters(['cf_time_fc', 'user_projection', 'user_resize', 
                                    'daytime_bias', 'weekend_bias', 'movie_year_bias', 'user_time_bias',
                                    'experts', 'gate_network', 'final_layer'])
        elif stage == 2:
            # 阶段2：只训练CF网络（包括所有CF相关组件）
            self._freeze_parameters(['temporal_network', 'user_history_network', 'experts', 'gate_network', 'final_layer'])
        elif stage == 3:
            # 阶段3：只训练MMoE融合层
            self._freeze_parameters(['temporal_network', 'user_history_network', 'cf_time_fc', 
                                    'user_projection', 'user_resize', 'daytime_bias', 'weekend_bias', 
                                    'movie_year_bias', 'user_time_bias'])
        else:
            # 全部解冻
            self._unfreeze_all_parameters()
    
    def _freeze_parameters(self, module_names):
        """冻结指定模块的参数"""
        for name, param in self.named_parameters():
            if any(module_name in name for module_name in module_names):
                param.requires_grad = False
            else:
                param.requires_grad = True
    
    def _unfreeze_all_parameters(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def _init_weights(self):
        """权重初始化 - 参考UMTimeModel"""
        # 主要嵌入层使用较小的标准差
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        
        # 偏差项初始化为小值
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        
        # 时间特征嵌入
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
        
        # CF层的时间偏差项
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)
        nn.init.normal_(self.movie_year_bias.weight, std=0.01)
        nn.init.normal_(self.user_time_bias.weight, std=0.01)
        
        # 线性层初始化
        nn.init.xavier_normal_(self.user_projection.weight)
        nn.init.constant_(self.user_projection.bias, 0)
        nn.init.xavier_normal_(self.user_resize.weight)
        nn.init.constant_(self.user_resize.bias, 0)
    
    def forward_temporal_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                              user_history_features):
        """
        阶段1：简化的时序建模前向传播
        
        Args:
            user_input: 用户ID
            item_input: 物品ID (用于获取历史交互)
            daytime_input, weekend_input, year_input: 时间特征
            user_history_features: [batch_size, 5] - 用户历史统计特征
                                  [平均评分, 评分标准差, 评分数量, 最近评分, 评分趋势]
        """
        # 获取嵌入
        user_embedded = self.user_embedding(user_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_embedded = self.dropout(user_embedded)
        
        # 处理用户历史特征
        if user_history_features is None:
            # 如果没有历史特征，使用默认值
            batch_size = user_input.size(0)
            user_history_features = torch.zeros(batch_size, 5).to(user_input.device)
            user_history_features[:, 0] = 3.0  # 默认平均评分
            user_history_features[:, 1] = 1.0  # 默认标准差
            user_history_features[:, 2] = 1.0  # 默认评分数量
        
        # 组合特征
        combined_features = torch.cat([
            user_embedded,
            daytime_embedded, 
            weekend_embedded,
            year_embedded,
            user_history_features  # 历史统计特征
        ], dim=1)
        
        # 时序网络预测
        prediction = self.temporal_network(combined_features)
        return prediction.squeeze()
    
    def forward_cf_stage(self, user_input, item_input, daytime_input, weekend_input, year_input):
        """阶段2：改进的CF前向传播 - 参考UMTimeModel"""
        # 获取基础嵌入
        user_base = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_base = self.dropout(user_base)
            item_embedded = self.dropout(item_embedded)
        
        # 参考UMTimeModel：用户嵌入的演化处理
        # 首先通过投影层处理用户嵌入
        user_projected = self.user_projection(user_base)
        
        # 参考UMTimeModel：核心交互 - 用户-物品点积
        base_interaction = torch.sum(user_projected * item_embedded, dim=1)
        
        # 参考UMTimeModel：添加基础偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        
        # 参考UMTimeModel：时间偏差项
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        movie_year_bias = self.movie_year_bias(year_input).squeeze()
        
        # 参考UMTimeModel：用户在特定时间段的偏好偏置
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        # 参考UMTimeModel：分离时间特征处理
        # 调整用户嵌入维度用于时间特征融合
        user_resized = self.user_resize(user_projected)  # 降维到k_factors//2
        
        # 融合用户演化特征和时间特征（不包括year，因为year作为偏差处理）
        time_features = torch.cat([daytime_embedded, weekend_embedded], dim=1)
        combined_features = torch.cat([user_resized, time_features], dim=1)
        
        # 通过FC层学习复杂的时间交互
        time_interaction = self.cf_time_fc(combined_features).squeeze()
        
        # 参考UMTimeModel：最终预测使用加法模式
        final_prediction = (base_interaction + user_bias + item_bias + 
                           daytime_bias + weekend_bias + movie_year_bias + 
                           user_daytime_bias + time_interaction + self.global_bias)
        
        return final_prediction
    
    def forward_mmoe_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                          temporal_predictions, cf_predictions):
        """阶段3：MMoE融合的前向传播"""
        # 获取用户和时间特征用于门控网络
        user_embedded = self.user_embedding(user_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        gate_input = torch.cat([
            user_embedded, daytime_embedded, weekend_embedded, year_embedded
        ], dim=1)
        
        # 门控权重
        gate_weights = self.gate_network(gate_input)  # [batch_size, num_experts]
        
        # 专家网络输入：时序预测 + CF预测
        expert_input = torch.stack([temporal_predictions, cf_predictions], dim=1)  # [batch_size, 2]
        
        # 计算专家输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(expert_input)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_output_dim]
        
        # 加权聚合
        weighted_output = torch.bmm(
            gate_weights.unsqueeze(1),  # [batch_size, 1, num_experts]
            expert_outputs  # [batch_size, num_experts, expert_output_dim]
        ).squeeze(1)  # [batch_size, expert_output_dim]
        
        # 最终预测
        final_prediction = self.final_layer(weighted_output).squeeze()
        return final_prediction
    
    def forward(self, *args, **kwargs):
        """根据训练阶段选择相应的前向传播"""
        if self.training_stage == 1:
            # 时序阶段：期望 (user_input, item_input, daytime_input, weekend_input, year_input, user_history_features)
            return self.forward_temporal_stage(*args, **kwargs)
        elif self.training_stage == 2:
            # CF阶段：期望 (user_input, item_input, daytime_input, weekend_input, year_input)
            return self.forward_cf_stage(*args, **kwargs)
        elif self.training_stage == 3:
            # MMoE阶段：期望 (user_input, item_input, daytime_input, weekend_input, year_input, temporal_preds, cf_preds)
            return self.forward_mmoe_stage(*args, **kwargs)
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
    
    def get_regularization_loss(self):
        """计算正则化损失"""
        reg_loss = 0
        
        # 基础嵌入正则化
        reg_loss += torch.norm(self.user_embedding.weight)
        reg_loss += torch.norm(self.item_embedding.weight)
        
        # 时间嵌入正则化
        reg_loss += torch.norm(self.daytime_embedding.weight) * 0.1
        reg_loss += torch.norm(self.weekend_embedding.weight) * 0.1
        reg_loss += torch.norm(self.year_embedding.weight) * 0.1
        
        if self.training_stage == 1:
            # 时序网络正则化
            reg_loss += sum(torch.norm(param) for param in self.temporal_network.parameters()) * 0.05
        elif self.training_stage == 2:
            # CF网络正则化（包括新增的CF组件）
            reg_loss += sum(torch.norm(param) for param in self.cf_time_fc.parameters()) * 0.05
            reg_loss += torch.norm(self.user_projection.weight) * 0.05
            reg_loss += torch.norm(self.user_resize.weight) * 0.05
            reg_loss += torch.norm(self.user_time_bias.weight) * 0.1
        elif self.training_stage == 3:
            # MMoE正则化
            reg_loss += sum(torch.norm(param) for expert in self.experts for param in expert.parameters()) * 0.05
        
        return self.reg_strength * reg_loss