import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , OneHotEncoder


def read():
    data = pd.read_csv("data/train.csv")
    data = data.drop(["Cabin", "Name"], axis=1)
    # 处理数据中的空值
    age = int(np.mean(data["Age"].fillna(0).values))
    value = {"Age": age, "Embarked": "un"}
    cols = ["Pclass", "Name", "Age", "SibSp", "Parch", "Ticket", "Fare", "Embarked"]
    # 将数据中的字符串转换为数值
    coder = LabelEncoder()
    data = data[["Sex", "Embarked"]].apply(lambda item: coder.fit_transform(item))