# CS182_proj

1.  
conda env create -f environment.yml

2.  
进入新创建的 `182proj` 环境  

3.  
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

上面的可能还是有点问题，建议找到关键库的版本，单独安装即可，全部安装也很慢  

如果不运行ipynb的几个文件，那么只需要安装torch即可  

三个ipynb都是原作者的代码，origin文件夹里的是将原代码更改为torch版本的代码, 作为我们的baseline   

torch_ver是我们自己的代码，里面包含四个模型，IndependentTimeModel，UserTimeModel，UMTimeModel，TwoStageMMoEModel(MMOE)  

MMOE里的两个网络是LSTM和UMTimeModel  

训练方法写在 report 的 readme 里了  