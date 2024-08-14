import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/1.salary.csv')

# 데이터 값만 불러오기
#  : -> 전부 선택하겠다.
array = data.values
X = array[:, 0]
Y = array[:, 1]
#
X = X.reshape(-1,1)