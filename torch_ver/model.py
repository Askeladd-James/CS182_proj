import torch
import torch.nn as nn

class CFModel(nn.Module):
    def __init__(self, n_users, m_items, k_factors):
        super(CFModel, self).__init__()
        self.user_embedding = nn.Embedding(n_users, k_factors)
        self.item_embedding = nn.Embedding(m_items, k_factors)
        
        # 初始化权重
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_input, item_input):
        user_embedded = self.user_embedding(user_input)
        item_embedded = self.item_embedding(item_input)
        output = torch.sum(user_embedded * item_embedded, dim=1)
        return output