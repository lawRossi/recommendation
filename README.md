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

### TODO
1. 实现DIN模型
2. 拿实验数据对比各个模型效果
