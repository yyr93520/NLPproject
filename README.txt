
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

scikit-learn.org/stable/modules/classes.html