import torch
import torch.nn as nn
import numpy as np

class CFModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors):
        super(CFModel, self).__init__()
        
        # 用户嵌入层
        self.user_embedding = nn.Embedding(n_users, k_factors)
        # 电影嵌入层
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 初始化权重
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_input, item_input):
        # 获取嵌入
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        
        # 计算点积
        output = torch.sum(user_embedded * item_embedded, dim=1)
        return output

    def rate(self, user_id, item_id):
        # 预测评分
        with torch.no_grad():
            user_input = torch.LongTensor([user_id]).cuda()  # 如果使用GPU
            item_input = torch.LongTensor([item_id]).cuda()  # 如果使用GPU
            prediction = self.forward(user_input, item_input)
            return prediction.cpu().numpy()[0]