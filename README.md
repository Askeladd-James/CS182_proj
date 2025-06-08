# CS182_proj

1.  
conda env create -f environment.yml

2.  
进入新创建的 `182proj` 环境  

3.  
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

如果不运行ipynb的几个文件，那么只需要安装torch即可

三个ipynb都是原作者的代码，origin文件夹里的是将原代码更改为torch版本的代码  
torch_ver是我们自己的代码，里面包含四个模型，IndependentTimeModel，UserTimeModel，UMTimeModel，TwoStageMMoEModel(MMOE)
MMOE里的两个网络是LSTM和UMTimeModel
