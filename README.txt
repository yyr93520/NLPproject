项目：基于酒店评论数据的情感分析工具

==================================
环境：
Ubuntu 14.04 LTS
Python 2.7.6
需要安装的依赖库(运行如下命令)：
pip install numpy sklearn scipy nltk jieba

==================================
使用方法（进入NLPproject目录）：
python train_model.py [opts]
-t input_file output_file: input_file给出测试数据文件（eg: ./data/input_file.txt），output_file给出结果文件（默认为./data/output_file.txt）
-r: 输出模型融合的测试结果
-d: 输出5个模型分别的测试结果

说明：
-t参数可以根据给定的「酒店评论」数据文件分析出评论的情感倾向是积极（pos）还是消极（neg），并输出结果，项目中NLPproject/data/input_file.txt文件已给出例子;
-r参数将输出我们的最终模型在测试数据上的评估结果，包括准确率、正确率、召回率、F1值等评价结果;
-d参数将输出我们采用的5种模型分别的评估结果，五种分类模型为：伯努利朴素贝叶斯模型、多项式朴素贝叶斯模型、随机森林、Logistic回归、线性SVM分类器。

==================================
附项目的文件结构
NLPproject/
    |——data/  存放所有数据
        |——model/  存放模型文件   xxx.pkl
        |——neg/  消极情感样本
        |——pos/  积极情感样本
        |——neg_data.txt  消极情感分词结果
        |——pos_data.txt  积极情感分词结果
        |——neg_word_seq.pkl 用于训练的消极情感序列化文件
        |——pos_word_seq.pkl 用于训练的积极情感序列化文件
    |——data_preprocess.py  数据预处理，获得样本分词结果
    |——feature_extrac.py  将每个样本用特征来表示
    |——train_model.py  用样本训练和测试模型


