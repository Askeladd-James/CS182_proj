import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 创建数据集类
class MovieLensDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = torch.LongTensor(users)
        self.movies = torch.LongTensor(movies)
        self.ratings = torch.FloatTensor(ratings)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# 创建数据加载器
train_dataset = MovieLensDataset(Users, Movies, Ratings)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CFModel(max_userid + 1, max_movieid + 1, K_FACTORS).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 训练循环
n_epochs = 30
for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    for user, movie, rating in train_loader:
        user = user.to(device)
        movie = movie.to(device)
        rating = rating.to(device)
        
        # 前向传播
        prediction = model(user, movie)
        loss = criterion(prediction, rating)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f'Epoch {epoch+1}/{n_epochs}')
    print(f'Average loss: {total_loss/len(train_loader):.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

def predict_rating(user_id, movie_id):
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        user = torch.LongTensor([user_id]).to(device)
        movie = torch.LongTensor([movie_id]).to(device)
        prediction = model(user, movie)
        return prediction.cpu().numpy()[0]


print(predict_rating(1, 1))