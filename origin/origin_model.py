import torch
import torch.nn as nn

class CFModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors, reg_strength=0.01):
        super(CFModel, self).__init__()
        self.reg_strength = reg_strength  # 添加这个属性
        
        self.user_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(m_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Dropout 正则化
        self.dropout = nn.Dropout(0.2)
        
        # 初始化权重
        self._init_weights()
        self.name = "CF"
        
    def _init_weights(self):
        # 使用Xavier初始化，更适合协同过滤
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)
        
        # 偏差项初始化为小值
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self, user_input, item_input):
        # 获取嵌入向量
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # 应用dropout（仅在训练时）
        if self.training:
            user_embedded = self.dropout(user_embedded)
            item_embedded = self.dropout(item_embedded)
        
        # 计算点积
        dot_product = torch.sum(user_embedded * item_embedded, dim=1)
        
        # 添加偏差项
        user_bias = self.user_bias(user_input).squeeze()
        item_bias = self.item_bias(item_input).squeeze()
        
        # 最终预测
        output = dot_product + user_bias + item_bias + self.global_bias
        
        return output
    
    def get_regularization_loss(self):
        """计算L2正则化损失"""
        user_reg = torch.norm(self.user_embedding.weight)
        item_reg = torch.norm(self.item_embedding.weight)
        return self.reg_strength * (user_reg + item_reg)
    
    def rate(self, user_id, item_id):
        """预测单个用户对单个物品的评分（为了与TensorFlow版本兼容）"""
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id])
            item_tensor = torch.LongTensor([item_id])
            prediction = self.forward(user_tensor, item_tensor)
            return prediction.item()