![Licence](https://img.shields.io/github/license/lawRossi/recommendation)
![Python](https://img.shields.io/badge/Python->=3.6-blue)

## 简介
跟踪推荐系统研究，用pytorch实现部分推荐模型并进行实验。

## 已实现模型

### 召回模型

|模型名称| 说明| 参考文献|
| --  | -- | -- |
Youtube-Net | Youtube提出的经典的召回模型 | Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
GES | 阿里提出的Graph Embedding with Side-information模型 | Wang, Jizhe, et al. "Billion-scale commodity embedding for e-commerce recommendation in alibaba." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.
EGES | 阿里提出的Enhanced Graph Embedding with Side-information模型 |Wang, Jizhe, et al. "Billion-scale commodity embedding for e-commerce recommendation in alibaba." Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018.



### 排序模型

模型名称| description| reference
| --  | -- | -- |
NRMS | 微软研究研究提出的新闻推荐模型 | Wu, Chuhan, et al. "Neural news recommendation with multi-head self-attention." Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019.
BST | 阿里提出的Behavior Sequence Transformer模型|Chen, Qiwei, et al. "Behavior sequence transformer for e-commerce recommendation in alibaba." Proceedings of the 1st International Workshop on Deep Learning Practice for High-Dimensional Sparse Data. 2019.

## TODO
1. 实现DIN模型
2. 拿实验数据对比各个模型效果

## 模型效果

1. 新闻推荐

### 实验设置
1. 数据集
使用微软发布的新闻推荐数据集


2. 模型参数

最大历史点击数：30  
标题最大长度（词数): 25  
dropout: 0.2  
embedding 维度300  
注意力头：20  
addative attention隐含层维度：200  


模型名称|dev-set auc|备注
--| -- | --
NRMS| 0.671 |论文提出的原始模型
NRMS-cosine| 0.684 | 基于原始模型修改了相似度计算和loss计算
