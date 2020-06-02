## requirement
PyTorch

## Train
`python train.py --data_path=YOUR-DATA-PATH`  
 需要修改训练文件路径 

## Test
`python camear.py`    
在weights中已经包含模型无需训练，但是这个模型训练次数非常少
## state
沙雕的py3.8中有个库没法和torchvision中C++实现的nms函数兼容，我用python实现的nms非常慢。
python 3.7以上不能使用torchvision中的nms,torchvision中的nms通过c++和c实现速度很快。
使用python写的nms非常缓慢。
## 测试
![tt.png](/img/tt.png)
## References  
[paper](https://arxiv.org/abs/1911.09070)  
[lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)  
[yetEfficientNet-PyTorch ](https://github.com/zylo117/Yet-Another-Efficient-Pytorch)  
