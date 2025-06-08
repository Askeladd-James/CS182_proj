import torch
import torch.nn as nn
import numpy as np

class TwoStageMMoEModel(nn.Module):
    """改进的两阶段MMoE模型 - 参考UMTimeModel的成功设计"""
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
        
        # 🔧 修复1: 完整的时间特征嵌入 - 参考UMTimeModel
        self.daytime_embedding = nn.Embedding(3, time_factors)
        self.weekend_embedding = nn.Embedding(2, time_factors)
        self.year_embedding = nn.Embedding(20, time_factors)
        
        # 基础偏差项
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # ============= 阶段1：简化的时序建模网络 =============
        # 🔧 修复2: 简化时序网络，参考UMTimeModel的直接设计
        temporal_input_dim = k_factors + k_factors + time_factors * 3  # 移除模拟历史特征
        
        self.temporal_feature_network = nn.Sequential(
            nn.Linear(temporal_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # ============= 阶段2：简化的协同过滤网络 - 参考UMTimeModel =============
        # 🔧 修复3: 完整的时间偏差项
        self.daytime_bias = nn.Embedding(3, 1)
        self.weekend_bias = nn.Embedding(2, 1)
        self.year_bias = nn.Embedding(20, 1)  # 添加年份偏差
        self.user_time_bias = nn.Embedding(n_users, 3)  # 用户时间交互偏差
        
        # 🔧 修复4: 简化CF网络 - 移除复杂的投影和维度变换
        # 直接使用时间特征融合，参考UMTimeModel的成功模式
        self.cf_time_fusion = nn.Sequential(
            nn.Linear(k_factors + 3 * time_factors, time_factors * 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(time_factors * 2, 1)
        )
        
        # ============= 阶段3：MMoE融合层 =============
        expert_input_dim = 2  # temporal_pred + cf_pred
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_input_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 8)
            ) for _ in range(num_experts)
        ])
        
        gate_input_dim = k_factors + time_factors * 3
        self.gate_network = nn.Sequential(
            nn.Linear(gate_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.final_layer = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1)
        )
        
        self.dropout = nn.Dropout(0.3)
        self._init_weights()
        
        self.training_stage = 1
    
    def _init_weights(self):
        """改进的权重初始化 - 参考UMTimeModel"""
        # 🔧 修复5: 使用更保守的初始化策略
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        nn.init.constant_(self.global_bias, 0.0)
        
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
        
        nn.init.normal_(self.daytime_bias.weight, std=0.01)
        nn.init.normal_(self.weekend_bias.weight, std=0.01)
        nn.init.normal_(self.year_bias.weight, std=0.01)
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
            # 🔧 修复6: 包含共享嵌入层的训练
            for param in self.user_embedding.parameters():
                param.requires_grad = True
            for param in self.item_embedding.parameters():
                param.requires_grad = True
            for param in self.daytime_embedding.parameters():
                param.requires_grad = True
            for param in self.weekend_embedding.parameters():
                param.requires_grad = True
            for param in self.year_embedding.parameters():
                param.requires_grad = True
            for param in self.temporal_feature_network.parameters():
                param.requires_grad = True
                
        elif stage == 2:
            # CF阶段：只训练CF网络和偏差项
            for param in self.parameters():
                param.requires_grad = False
            for param in self.user_bias.parameters():
                param.requires_grad = True
            for param in self.item_bias.parameters():
                param.requires_grad = True
            for param in self.daytime_bias.parameters():
                param.requires_grad = True
            for param in self.weekend_bias.parameters():
                param.requires_grad = True
            for param in self.year_bias.parameters():
                param.requires_grad = True
            for param in self.user_time_bias.parameters():
                param.requires_grad = True
            for param in self.cf_time_fusion.parameters():
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
    
    def forward_temporal_stage(self, user_input, item_input, daytime_input, weekend_input, year_input):
        """🔧 修复7: 简化时序建模前向传播 - 移除模拟历史特征"""
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 🔧 直接组合用户、物品和时间特征
        combined_features = torch.cat([
            user_embedded,
            item_embedded,
            daytime_embedded, 
            weekend_embedded,
            year_embedded
        ], dim=1)
        
        # 简化的时序预测
        prediction = self.temporal_feature_network(combined_features)
        return prediction.squeeze()
    
    def forward_cf_stage(self, user_input, item_input, daytime_input, weekend_input, year_input):
        """🔧 修复8: 简化CF前向传播 - 参考UMTimeModel设计"""
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 🔧 核心交互 - 直接点积，参考UMTimeModel
        base_interaction = torch.sum(user_embedded * item_embedded, dim=1)
        
        # 🔧 完整的偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        daytime_bias = self.daytime_bias(daytime_input).squeeze()
        weekend_bias = self.weekend_bias(weekend_input).squeeze()
        year_bias = self.year_bias(year_input).squeeze()
        
        # 🔧 用户时间交互偏差 - 参考UMTimeModel
        time_bias = self.user_time_bias(user_input)
        user_daytime_bias = torch.gather(time_bias, 1, daytime_input.unsqueeze(1)).squeeze()
        
        # 🔧 简化的时间特征融合 - 参考UMTimeModel
        time_features = torch.cat([
            daytime_embedded, 
            weekend_embedded, 
            year_embedded
        ], dim=1)
        
        # 将用户特征与时间特征结合
        combined_features = torch.cat([user_embedded, time_features], dim=1)
        time_interaction = self.cf_time_fusion(combined_features).squeeze()
        
        # 🔧 最终预测 - 参考UMTimeModel的成功模式
        final_prediction = (base_interaction + user_bias + item_bias + 
                           daytime_bias + weekend_bias + year_bias + 
                           user_daytime_bias + time_interaction + self.global_bias)
        
        return final_prediction
    
    def forward_mmoe_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                          temporal_pred, cf_pred):
        """MMoE融合前向传播"""
        # 确保预测值维度正确
        if temporal_pred.dim() == 0:
            temporal_pred = temporal_pred.unsqueeze(0)
        if cf_pred.dim() == 0:
            cf_pred = cf_pred.unsqueeze(0)
            
        if temporal_pred.dim() > 1:
            temporal_pred = temporal_pred.squeeze()
        if cf_pred.dim() > 1:
            cf_pred = cf_pred.squeeze()
            
        batch_size = user_input.size(0)
        if temporal_pred.size(0) != batch_size:
            temporal_pred = temporal_pred.expand(batch_size)
        if cf_pred.size(0) != batch_size:
            cf_pred = cf_pred.expand(batch_size)
        
        # 组合专家输入
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
        gate_weights = self.gate_network(gate_input)
        
        # 专家网络输出
        expert_outputs = []
        for expert in self.experts:
            expert_out = expert(expert_input)
            expert_outputs.append(expert_out)
        
        expert_outputs = torch.stack(expert_outputs, dim=2)
        
        # 加权融合
        gate_weights = gate_weights.unsqueeze(1)
        weighted_output = torch.bmm(gate_weights, expert_outputs.transpose(1, 2))
        weighted_output = weighted_output.squeeze(1)
        
        # 最终预测
        final_prediction = self.final_layer(weighted_output)
        return final_prediction.squeeze()
    
    def forward(self, user_input, item_input, daytime_input, weekend_input, year_input, 
                temporal_pred=None, cf_pred=None, user_history_features=None):
        """前向传播"""
        if self.training_stage == 1:
            return self.forward_temporal_stage(
                user_input, item_input, daytime_input, weekend_input, year_input
            )
        elif self.training_stage == 2:
            return self.forward_cf_stage(
                user_input, item_input, daytime_input, weekend_input, year_input
            )
        elif self.training_stage == 3:
            if temporal_pred is None or cf_pred is None:
                raise ValueError("MMoE stage requires both temporal_pred and cf_pred")
            return self.forward_mmoe_stage(
                user_input, item_input, daytime_input, weekend_input, year_input, temporal_pred, cf_pred
            )
        elif self.training_stage == 4:
            # 评估阶段：依次调用三个阶段
            self.training_stage = 1
            temporal_pred = self.forward_temporal_stage(
                user_input, item_input, daytime_input, weekend_input, year_input
            )
            
            self.training_stage = 2
            cf_pred = self.forward_cf_stage(
                user_input, item_input, daytime_input, weekend_input, year_input
            )
            
            self.training_stage = 3
            final_pred = self.forward_mmoe_stage(
                user_input, item_input, daytime_input, weekend_input, year_input, temporal_pred, cf_pred
            )
            
            self.training_stage = 4
            return final_pred
        else:
            raise ValueError(f"Unknown training stage: {self.training_stage}")
    
    def get_regularization_loss(self):
        """🔧 修复9: 改进的正则化损失"""
        reg_loss = 0
        
        # 基础嵌入正则化
        reg_loss += torch.norm(self.user_embedding.weight) * 0.1
        reg_loss += torch.norm(self.item_embedding.weight) * 0.1
        
        # 时间特征正则化
        reg_loss += torch.norm(self.daytime_embedding.weight) * 0.05
        reg_loss += torch.norm(self.weekend_embedding.weight) * 0.05
        reg_loss += torch.norm(self.year_embedding.weight) * 0.05
        
        # 根据训练阶段调整正则化
        if self.training_stage == 1:
            reg_loss += sum(torch.norm(param) for param in self.temporal_feature_network.parameters()) * 0.01
        elif self.training_stage == 2:
            reg_loss += sum(torch.norm(param) for param in self.cf_time_fusion.parameters()) * 0.01
        elif self.training_stage == 3:
            reg_loss += sum(torch.norm(param) for expert in self.experts for param in expert.parameters()) * 0.005
            reg_loss += sum(torch.norm(param) for param in self.gate_network.parameters()) * 0.005
            reg_loss += sum(torch.norm(param) for param in self.final_layer.parameters()) * 0.005
        
        return self.reg_strength * reg_loss
    
    def rate(self, user_id, item_id, daytime=1, weekend=0, year=10):
        """预测单个用户对单个物品的评分（兼容接口）"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            daytime_tensor = torch.LongTensor([daytime])
            weekend_tensor = torch.LongTensor([weekend])
            year_tensor = torch.LongTensor([year])
            
            # 设置为评估阶段
            original_stage = self.training_stage
            self.training_stage = 4
            
            prediction = self.forward(user_tensor, item_tensor, daytime_tensor, 
                                    weekend_tensor, year_tensor)
            
            # 恢复原始阶段
            self.training_stage = original_stage
            
            return prediction.item()