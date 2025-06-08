import torch
import torch.nn as nn
import numpy as np

class TwoStageMMoEModel(nn.Module):
    """优化后的两阶段MMoE模型 - 简化LSTM部分"""
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
        
        # ============= 阶段2：传统协同过滤网络（简化） =============
        cf_input_dim = k_factors * 2 + time_factors // 2 * 3
        self.cf_network = nn.Sequential(
            nn.Linear(cf_input_dim, 64),  # 减少网络复杂度
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
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
            self._freeze_parameters(['cf_network', 'experts', 'gate_network', 'final_layer'])
        elif stage == 2:
            # 阶段2：只训练CF网络
            self._freeze_parameters(['temporal_network', 'user_history_network', 'experts', 'gate_network', 'final_layer'])
        elif stage == 3:
            # 阶段3：只训练MMoE融合层
            self._freeze_parameters(['temporal_network', 'user_history_network', 'cf_network'])
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
        """权重初始化"""
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)
        nn.init.normal_(self.daytime_embedding.weight, std=0.01)
        nn.init.normal_(self.weekend_embedding.weight, std=0.01)
        nn.init.normal_(self.year_embedding.weight, std=0.01)
    
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
        """阶段2：传统CF的前向传播"""
        # 获取嵌入
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        daytime_embedded = self.daytime_embedding(daytime_input)
        weekend_embedded = self.weekend_embedding(weekend_input)
        year_embedded = self.year_embedding(year_input)
        
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 组合特征
        combined_features = torch.cat([
            user_embedded, item_embedded, 
            daytime_embedded, weekend_embedded, year_embedded
        ], dim=1)
        
        # CF网络预测
        cf_output = self.cf_network(combined_features)
        
        # 添加偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        
        final_prediction = cf_output.squeeze() + user_bias + item_bias + self.global_bias
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
            # CF网络正则化
            reg_loss += sum(torch.norm(param) for param in self.cf_network.parameters()) * 0.05
        elif self.training_stage == 3:
            # MMoE正则化
            reg_loss += sum(torch.norm(param) for expert in self.experts for param in expert.parameters()) * 0.05
        
        return self.reg_strength * reg_loss


# import torch
# import torch.nn as nn
# import numpy as np

# class TwoStageMMoEModel(nn.Module):
#     """两阶段训练的MMoE模型"""
#     def __init__(self, n_users, m_items, k_factors=80, time_factors=20, reg_strength=0.01, num_experts=4):
#         super(TwoStageMMoEModel, self).__init__()
#         self.reg_strength = reg_strength
#         self.name = "TwoStage_MMoE"
#         self.k_factors = k_factors
#         self.time_factors = time_factors
#         self.num_experts = num_experts
        
#         # ============= 共享嵌入层 =============
#         self.user_embedding = nn.Embedding(n_users, k_factors)
#         self.item_embedding = nn.Embedding(m_items, k_factors)
        
#         # 时间特征嵌入
#         self.daytime_embedding = nn.Embedding(3, time_factors)
#         self.weekend_embedding = nn.Embedding(2, time_factors)
#         self.year_embedding = nn.Embedding(20, time_factors)
        
#         # 基础偏差项
#         self.user_bias = nn.Embedding(n_users, 1)
#         self.item_bias = nn.Embedding(m_items, 1)
#         self.global_bias = nn.Parameter(torch.zeros(1))
        
#         # ============= 阶段1：时序建模网络 (LSTM) =============
#         lstm_input_dim = k_factors + time_factors * 3 + 1  # 用户嵌入 + 时间特征 + 评分
#         self.temporal_lstm = nn.LSTM(
#             input_size=lstm_input_dim,
#             hidden_size=64,
#             num_layers=2,
#             batch_first=True,
#             dropout=0.3
#         )
        
#         # LSTM的输出层（用于阶段1训练）
#         self.lstm_output_layer = nn.Sequential(
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(32, 1)
#         )
        
#         # ============= 阶段2：传统协同过滤网络 =============
#         cf_input_dim = k_factors * 2 + time_factors * 3
#         self.cf_network = nn.Sequential(
#             nn.Linear(cf_input_dim, 128),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1)
#         )
        
#         # ============= MMoE融合层 =============
#         # 专家网络：处理两个任务的输出
#         expert_input_dim = 2  # LSTM输出 + CF输出
#         self.experts = nn.ModuleList([
#             nn.Sequential(
#                 nn.Linear(expert_input_dim, 32),
#                 nn.ReLU(),
#                 nn.Dropout(0.2),
#                 nn.Linear(32, 16)
#             ) for _ in range(num_experts)
#         ])
        
#         # 门控网络
#         gate_input_dim = k_factors + time_factors * 3  # 用户特征 + 时间特征
#         self.gate_network = nn.Sequential(
#             nn.Linear(gate_input_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, num_experts),
#             nn.Softmax(dim=1)
#         )
        
#         # 最终输出层
#         self.final_layer = nn.Sequential(
#             nn.Linear(16, 8),
#             nn.ReLU(),
#             nn.Dropout(0.2),
#             nn.Linear(8, 1)
#         )
        
#         self.dropout = nn.Dropout(0.3)
#         self._init_weights()
        
#         # 训练阶段标志
#         self.training_stage = 1  # 1: LSTM训练, 2: CF训练, 3: MMoE融合训练
    
#     def set_training_stage(self, stage):
#         """设置训练阶段"""
#         self.training_stage = stage
        
#         if stage == 1:
#             # 阶段1：只训练LSTM相关参数
#             self._freeze_parameters(['cf_network', 'experts', 'gate_network', 'final_layer'])
#         elif stage == 2:
#             # 阶段2：只训练CF网络
#             self._freeze_parameters(['temporal_lstm', 'lstm_output_layer', 'experts', 'gate_network', 'final_layer'])
#         elif stage == 3:
#             # 阶段3：只训练MMoE融合层，冻结LSTM和CF
#             self._freeze_parameters(['temporal_lstm', 'lstm_output_layer', 'cf_network'])
#         else:
#             # 全部解冻
#             self._unfreeze_all_parameters()
    
#     def _freeze_parameters(self, module_names):
#         """冻结指定模块的参数"""
#         for name, param in self.named_parameters():
#             if any(module_name in name for module_name in module_names):
#                 param.requires_grad = False
#             else:
#                 param.requires_grad = True
    
#     def _unfreeze_all_parameters(self):
#         """解冻所有参数"""
#         for param in self.parameters():
#             param.requires_grad = True
    
#     def _init_weights(self):
#         """权重初始化"""
#         nn.init.normal_(self.user_embedding.weight, std=0.05)
#         nn.init.normal_(self.item_embedding.weight, std=0.05)
#         nn.init.normal_(self.user_bias.weight, std=0.01)
#         nn.init.normal_(self.item_bias.weight, std=0.01)
#         nn.init.normal_(self.daytime_embedding.weight, std=0.01)
#         nn.init.normal_(self.weekend_embedding.weight, std=0.01)
#         nn.init.normal_(self.year_embedding.weight, std=0.01)
        
#         # LSTM初始化
#         for name, param in self.temporal_lstm.named_parameters():
#             if 'weight' in name:
#                 nn.init.xavier_normal_(param)
#             elif 'bias' in name:
#                 nn.init.constant_(param, 0)
    
#     def forward_lstm_stage(self, sequence_ids, sequence_ratings):
#         """
#         阶段1：LSTM时序建模的前向传播
        
#         Args:
#             sequence_ids: [batch_size, seq_len, 4] - 用户ID, daytime, weekend, year
#             sequence_ratings: [batch_size, seq_len] - 对应的评分
#         """
#         batch_size, seq_len, _ = sequence_ids.shape
        
#         # 获取嵌入
#         user_ids = sequence_ids[:, :, 0]  # [batch_size, seq_len]
#         daytime_ids = sequence_ids[:, :, 1]
#         weekend_ids = sequence_ids[:, :, 2] 
#         year_ids = sequence_ids[:, :, 3]
        
#         # 处理零填充的位置（用户ID为0的位置）
#         mask = user_ids != 0  # [batch_size, seq_len]
        
#         # 获取嵌入向量
#         user_embedded = self.user_embedding(user_ids)  # [batch_size, seq_len, k_factors]
#         daytime_embedded = self.daytime_embedding(daytime_ids)  # [batch_size, seq_len, time_factors]
#         weekend_embedded = self.weekend_embedding(weekend_ids)  # [batch_size, seq_len, time_factors]
#         year_embedded = self.year_embedding(year_ids)  # [batch_size, seq_len, time_factors]
        
#         # 组合特征
#         lstm_input = torch.cat([
#             user_embedded,
#             daytime_embedded, 
#             weekend_embedded,
#             year_embedded,
#             sequence_ratings.unsqueeze(-1)  # [batch_size, seq_len, 1]
#         ], dim=-1)  # [batch_size, seq_len, k_factors + 3*time_factors + 1]
        
#         # 应用mask（零填充位置设为0）
#         lstm_input = lstm_input * mask.unsqueeze(-1).float()
        
#         # LSTM前向传播
#         lstm_output, _ = self.temporal_lstm(lstm_input)
        
#         # 取最后一个有效时间步的输出
#         # 找到每个序列的最后一个非零位置
#         seq_lengths = mask.sum(dim=1)  # [batch_size]
#         batch_indices = torch.arange(batch_size).to(lstm_output.device)
#         last_outputs = lstm_output[batch_indices, seq_lengths - 1]  # [batch_size, hidden_size]
        
#         # 通过输出层
#         prediction = self.lstm_output_layer(last_outputs)
#         return prediction.squeeze()
    
#     def forward_cf_stage(self, user_input, item_input, daytime_input, weekend_input, year_input):
#         """阶段2：传统CF的前向传播"""
#         # 获取嵌入
#         user_embedded = self.user_embedding(user_input)
#         item_embedded = self.item_embedding(item_input)
#         daytime_embedded = self.daytime_embedding(daytime_input)
#         weekend_embedded = self.weekend_embedding(weekend_input)
#         year_embedded = self.year_embedding(year_input)
        
#         if self.training:
#             user_embedded = self.dropout(user_embedded)
#             item_embedded = self.dropout(item_embedded)
        
#         # 组合特征
#         combined_features = torch.cat([
#             user_embedded, item_embedded, 
#             daytime_embedded, weekend_embedded, year_embedded
#         ], dim=1)
        
#         # CF网络预测
#         cf_output = self.cf_network(combined_features)
        
#         # 添加偏差项
#         user_bias = self.user_bias(user_input).squeeze()
#         item_bias = self.item_bias(item_input).squeeze()
        
#         final_prediction = cf_output.squeeze() + user_bias + item_bias + self.global_bias
#         return final_prediction
    
#     def forward_mmoe_stage(self, user_input, item_input, daytime_input, weekend_input, year_input, 
#                           lstm_predictions, cf_predictions):
#         """阶段3：MMoE融合的前向传播"""
#         # 获取用户和时间特征用于门控网络
#         user_embedded = self.user_embedding(user_input)
#         daytime_embedded = self.daytime_embedding(daytime_input)
#         weekend_embedded = self.weekend_embedding(weekend_input)
#         year_embedded = self.year_embedding(year_input)
        
#         gate_input = torch.cat([
#             user_embedded, daytime_embedded, weekend_embedded, year_embedded
#         ], dim=1)
        
#         # 门控权重
#         gate_weights = self.gate_network(gate_input)  # [batch_size, num_experts]
        
#         # 专家网络输入：LSTM预测 + CF预测
#         expert_input = torch.stack([lstm_predictions, cf_predictions], dim=1)  # [batch_size, 2]
        
#         # 计算专家输出
#         expert_outputs = []
#         for expert in self.experts:
#             expert_output = expert(expert_input)
#             expert_outputs.append(expert_output)
#         expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, expert_output_dim]
        
#         # 加权聚合
#         weighted_output = torch.bmm(
#             gate_weights.unsqueeze(1),  # [batch_size, 1, num_experts]
#             expert_outputs  # [batch_size, num_experts, expert_output_dim]
#         ).squeeze(1)  # [batch_size, expert_output_dim]
        
#         # 最终预测
#         final_prediction = self.final_layer(weighted_output).squeeze()
#         return final_prediction
    
#     def forward(self, *args, **kwargs):
#         """根据训练阶段选择相应的前向传播"""
#         if self.training_stage == 1:
#             # LSTM阶段：期望 (sequence_ids, sequence_ratings)
#             return self.forward_lstm_stage(*args, **kwargs)
#         elif self.training_stage == 2:
#             # CF阶段：期望 (user_input, item_input, daytime_input, weekend_input, year_input)
#             return self.forward_cf_stage(*args, **kwargs)
#         elif self.training_stage == 3:
#             # MMoE阶段：期望 (user_input, item_input, daytime_input, weekend_input, year_input, lstm_preds, cf_preds)
#             return self.forward_mmoe_stage(*args, **kwargs)
#         else:
#             raise ValueError(f"Unknown training stage: {self.training_stage}")
    
#     def get_lstm_prediction(self, user_sequences):
#         """获取LSTM预测（用于阶段3）"""
#         self.eval()
#         with torch.no_grad():
#             return self.forward_lstm_stage(user_sequences)
    
#     def get_cf_prediction(self, user_input, item_input, daytime_input, weekend_input, year_input):
#         """获取CF预测（用于阶段3）"""
#         self.eval()
#         with torch.no_grad():
#             return self.forward_cf_stage(user_input, item_input, daytime_input, weekend_input, year_input)
    
#     def get_regularization_loss(self):
#         """计算正则化损失"""
#         reg_loss = 0
        
#         # 基础嵌入正则化
#         reg_loss += torch.norm(self.user_embedding.weight)
#         reg_loss += torch.norm(self.item_embedding.weight)
        
#         # 时间嵌入正则化
#         reg_loss += torch.norm(self.daytime_embedding.weight) * 0.1
#         reg_loss += torch.norm(self.weekend_embedding.weight) * 0.1
#         reg_loss += torch.norm(self.year_embedding.weight) * 0.1
        
#         if self.training_stage == 1:
#             # LSTM正则化
#             reg_loss += sum(torch.norm(param) for param in self.temporal_lstm.parameters()) * 0.05
#         elif self.training_stage == 2:
#             # CF网络正则化
#             reg_loss += sum(torch.norm(param) for param in self.cf_network.parameters()) * 0.05
#         elif self.training_stage == 3:
#             # MMoE正则化
#             reg_loss += sum(torch.norm(param) for expert in self.experts for param in expert.parameters()) * 0.05
        
#         return self.reg_strength * reg_loss