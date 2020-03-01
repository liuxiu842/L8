#!pip install surprise
from surprise import KNNWithMeans
from surprise import KNNBasic
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import Dataset, Reader
from surprise import accuracy
from surprise.model_selection import KFold


# 数据读取
reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)
data = Dataset.load_from_file('/content/drive/My Drive/Colab Notebooks/ratings.csv', reader=reader)
trainset = data.build_full_trainset()

# ItemCF 计算得分
# 取最相似的用户计算时，只取最相似的k个
# algo = KNNWithMeans(k=50, sim_options={'user_based': True, 'verbose': 'True'})
# algo = KNNBasic(k=50, sim_options={'user_based': True, 'verbose': 'True'})
# algo = KNNWithZScore(k=50, sim_options={'user_based': True, 'verbose': 'True'})
algo = KNNBaseline(k=50, sim_options={'user_based': True, 'verbose': 'True'})
algo.fit(trainset)

# 定义K折交叉验证迭代器，K=3
kf = KFold(n_splits=3)
for trainset, testset in kf.split(data):
    # 训练并预测
    algo.fit(trainset)
    predictions = algo.test(testset)
    # 计算RMSE
    accuracy.rmse(predictions, verbose=True)
    accuracy.mae(predictions, verbose=True)
