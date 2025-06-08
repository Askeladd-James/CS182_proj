import torch
import torch.nn as nn
import numpy as np

class TwoStageMMoEModel(nn.Module):
    """两阶段MMoE模型 - 修复评估阶段维度错误"""
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
        
        # 时间特征嵌入
        self.daytime_embedding = nn.Embedding(3, time_factors)
        self.weekend_embedding = nn.Embedding(2, time_factors)
        self.year_embedding = nn.Embedding(20, time_factors)
        
        # 基础偏差项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # ============= 阶段1：改进的时序建模网络 =============
        temporal_input_dim = k_factors + k_factors + time_factors * 3 + 5
        
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
        
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=128, 
            num_heads=4, 
            dropout=0.1,
            batch_first=True
        )
        
        self.temporal_prediction = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        self.user_history_enhancer = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # ============= 阶段2：改进的协同过滤网络 =============
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        self.movie_year_bias = nn.Embedding(20, 1)
        self.user_time_bias = nn.Embedding(n_users, 3)
        
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
        
        time_features_dim = time_factors * 2
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
        expert_input_dim = 2  # temporal_pred + cf_pred
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
        
        self.final_layer = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        self.dropout = nn.Dropout(0.1)
        self._init_weights()
        
        self.training_stage = 1
    
    def _init_weights(self):
        """改进的权重初始化"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        nn.init.constant_(self.global_bias, 0.0)
        
        nn.init.normal_(self.daytime_embedding.weight, std=0.02)
        nn.init.normal_(self.weekend_embedding.weight, std=0.02)
        nn.init.normal_(self.year_embedding.weight, std=0.02)
        
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)
        nn.init.normal_(self.movie_year_bias.weight, std=0.01)
        nn.init.normal_(self.user_time_bias.weight, std=0.01)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.01)
    
    def set_training_stage(self, stage):
        """设置训练阶段"""
        self.training_stage = stage
        
        if stage == 1:
            # 时序阶段：只训练时序网络
            for param in self.parameters():
                param.requires_grad = False
            for param in self.temporal_feature_network.parameters():
                param.requires_grad = True
            for param in self.temporal_attention.parameters():
                param.requires_grad = True
            for param in self.temporal_prediction.parameters():
                param.requires_grad = True
            for param in self.user_history_enhancer.parameters():
                param.requires_grad = True
                
        elif stage == 2:
            # CF阶段：只训练CF网络
            for param in self.parameters():
                param.requires_grad = False
            for param in self.user_projection.parameters():
                param.requires_grad = True
            for param in self.user_resize.parameters():
                param.requires_grad = True
            for param in self.cf_time_fc.parameters():
                param.requires_grad = True
                
        elif stage == 3:
            # MMoE阶段：只训练MMoE网络
            for param in self.parameters():
                param.requires_grad = False
            for param in self.experts.parameters():
                param.requires_grad = True
            for param in self.gate_network.parameters():
                param.requires_grad = True
            for param in self.final_layer.parameters():
                param.requires_grad = True
                
        elif stage == 4:
            # 解冻所有参数用于评估
            for param in self.parameters():
                param.requires_grad = True
    
    def forward_temporal_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                              user_history_features=None):
        """时序建模前向传播"""
        batch_size = user_input.size(0)
        
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        if user_history_features is None:
            device = user_input.device
            user_history_features = torch.zeros(batch_size, 5).to(device)
            
            user_ids_float = user_input.float()
            user_history_features[:, 0] = 2.5 + 1.0 * torch.sin(user_ids_float * 0.1)
            user_history_features[:, 1] = 0.5 + 0.3 * torch.cos(user_ids_float * 0.15)
            user_history_features[:, 2] = torch.clamp(torch.log(user_ids_float + 1), 1, 100)
            user_history_features[:, 3] = user_history_features[:, 0] + 0.2 * torch.randn_like(user_ids_float)
            user_history_features[:, 4] = 0.1 * torch.sin(user_ids_float * 0.05)
        
        enhanced_history = self.user_history_enhancer(user_history_features)
        
        combined_features = torch.cat([
            user_embedded,
            item_embedded,
            daytime_embedded, 
            weekend_embedded,
            year_embedded,
            user_history_features
        ], dim=1)
        
        temporal_features = self.temporal_feature_network(combined_features)
        
        temporal_features_reshaped = temporal_features.unsqueeze(1)
        attended_features, _ = self.temporal_attention(
            temporal_features_reshaped, 
            temporal_features_reshaped, 
            temporal_features_reshaped
        )
        attended_features = attended_features.squeeze(1)
        
        final_features = temporal_features + attended_features
        
        prediction = self.temporal_prediction(final_features)
        return prediction.squeeze()
    
    def forward_cf_stage(self, user_input, item_input, daytime_input, weekend_input, year_input):
        """CF前向传播"""
        user_base = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_base = self.dropout(user_base)
            item_embedded = self.dropout(item_embedded)
        
        user_projected = self.user_projection(user_base)
        user_projected = user_projected + user_base
        
        base_interaction = torch.sum(user_projected * item_embedded, dim=1)
        
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        movie_year_bias = self.movie_year_bias(year_input).squeeze()
        
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        user_resized = self.user_resize(user_projected)
        time_features = torch.cat([daytime_embedded, weekend_embedded], dim=1)
        combined_features = torch.cat([user_resized, time_features], dim=1)
        
        time_interaction = self.cf_time_fc(combined_features).squeeze()
        
        final_prediction = (base_interaction + user_bias + item_bias + 
                           daytime_bias + weekend_bias + movie_year_bias + 
                           user_daytime_bias + time_interaction + self.global_bias)
        
        return final_prediction
    
    def forward_mmoe_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                          temporal_pred, cf_pred):
        """MMoE融合前向传播 - 修复维度错误"""
        # 确保预测值是标量或1D张量
        if temporal_pred.dim() == 0:
            temporal_pred = temporal_pred.unsqueeze(0)
        if cf_pred.dim() == 0:
            cf_pred = cf_pred.unsqueeze(0)
            
        # 如果是批处理，确保维度正确
        if temporal_pred.dim() > 1:
            temporal_pred = temporal_pred.squeeze()
        if cf_pred.dim() > 1:
            cf_pred = cf_pred.squeeze()
            
        # 确保是1D张量，并且长度匹配
        batch_size = user_input.size(0)
        if temporal_pred.size(0) != batch_size:
            temporal_pred = temporal_pred.expand(batch_size)
        if cf_pred.size(0) != batch_size:
            cf_pred = cf_pred.expand(batch_size)
        
        # 组合专家输入 [batch_size, 2]
        expert_input = torch.stack([temporal_pred, cf_pred], dim=1)
        
        # 门控网络输入
        user_embedded = self.user_embedding(user_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        gate_input = torch.cat([
            user_embedded, 
            daytime_embedded, 
            weekend_embedded, 
            year_embedded
        ], dim=1)
        
        # 计算门控权重
        gate_weights = self.gate_network(gate_input)  # [batch_size, num_experts]
        
        # 专家网络输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(expert_input)  # [batch_size, 8]
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)  # [batch_size, 8, num_experts]
        
        # 加权融合：使用bmm进行批量矩阵乘法
        gate_weights = gate_weights.unsqueeze(1)  # [batch_size, 1, num_experts]
        weighted_output = torch.bmm(gate_weights, expert_outputs.transpose(1, 2))  # [batch_size, 1, 8]
        weighted_output = weighted_output.squeeze(1)  # [batch_size, 8]
        
        # 最终预测
        final_prediction = self.final_layer(weighted_output)
        return final_prediction.squeeze()
    
    def forward(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                temporal_pred=None, cf_pred=None, user_history_features=None):
        """前向传播 - 修复所有阶段的调用"""
        if self.training_stage == 1:
            return self.forward_temporal_stage(
                user_input, item_input, daytime_input, weekend_input, year_input, user_history_features
            )
        elif self.training_stage == 2:
            return self.forward_cf_stage(
                user_input, item_input, daytime_input, weekend_input, year_input
            )
        elif self.training_stage == 3:
            # MMoE阶段需要两个预测输入
            if temporal_pred is None or cf_pred is None:
                raise ValueError("MMoE stage requires both temporal_pred and cf_pred")
            return self.forward_mmoe_stage(
                user_input, item_input, daytime_input, weekend_input, year_input, temporal_pred, cf_pred
            )
        elif self.training_stage == 4:
            # 评估阶段：依次调用三个阶段
            # 阶段1：时序预测
            self.training_stage = 1
            temporal_pred = self.forward_temporal_stage(
                user_input, item_input, daytime_input, weekend_input, year_input, user_history_features
            )
            
            # 阶段2：CF预测
            self.training_stage = 2
            cf_pred = self.forward_cf_stage(
                user_input, item_input, daytime_input, weekend_input, year_input
            )
            
            # 阶段3：MMoE融合
            self.training_stage = 3
            final_pred = self.forward_mmoe_stage(
                user_input, item_input, daytime_input, weekend_input, year_input, temporal_pred, cf_pred
            )
            
            # 恢复评估阶段
            self.training_stage = 4
            return final_pred
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
    
    def get_regularization_loss(self):
        """调整正则化损失"""
        reg_loss = 0
        
        reg_loss += torch.norm(self.user_embedding.weight) * 0.5
        reg_loss += torch.norm(self.item_embedding.weight) * 0.5
        
        reg_loss += torch.norm(self.daytime_embedding.weight) * 0.1
        reg_loss += torch.norm(self.weekend_embedding.weight) * 0.1
        reg_loss += torch.norm(self.year_embedding.weight) * 0.1
        
        if self.training_stage == 1:
            reg_loss += sum(torch.norm(param) for param in self.temporal_feature_network.parameters()) * 0.01
            reg_loss += sum(torch.norm(param) for param in self.temporal_prediction.parameters()) * 0.01
        elif self.training_stage == 2:
            reg_loss += sum(torch.norm(param) for param in self.cf_time_fc.parameters()) * 0.01
        elif self.training_stage == 3:
            reg_loss += sum(torch.norm(param) for expert in self.experts for param in expert.parameters()) * 0.01
        
        return self.reg_strength * reg_loss