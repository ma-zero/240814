import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('./data/1.salary.csv')


#  : -> 전부 선택하겠다. 행 열
array = data.values
array.shape
X = array[:, 0] # 독립변수 : 종속변수에 영향을 주는 변수
Y = array[:, 1] # 종속변수 : 독립 변수에 영향을 받는 변수

# 근속연수 * 연봉
XR = X.reshape(-1,1)
# 데이터 분할
# testsize 0.2 = 20% 만 테스트에 쓰겠다.
X_train, X_test, Y_train, Y_test = train_test_split(XR,Y , test_size=0.2)


# # 모델 선택 및 분할
model = LinearRegression()
model.fit(X_train, Y_train)

# X_test로 값을 예측해봐
y_pred = model.predict(X_test)
print(y_pred)


plt.figure(figsize=(10,6))
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Actual Values', marker='o')
plt.plot(range(len(y_pred)), y_pred, color='red', label='Predicted Values', marker='x')
plt.show()

plt.clf()
plt.scatter(X, Y, color='green', label='Predicted Values', marker='*')

plt.title("Years, Salary data")
plt.xlabel("Years")
plt.ylabel("Salary")
plt.legend()
plt.show()
plt.savefig("./results/scatter.png")

# 모델 정확도 계산
mean = mean_absolute_error(Y_test, y_pred)
print(mean)

# -------------------------------------------------------------------------------------

