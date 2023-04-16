import pandas as pd
from sklearn.preprocessing import StandardScaler

# CSVファイルからデータを読み込む
data = pd.read_csv("transactions.csv")

# 取引先の住所を頻度で置き換える
address_counts = data['address'].value_counts()
address_counts = address_counts[address_counts >= 10]
data['address'] = data['address'].apply(lambda x: x if x in address_counts else "Other")

# 頻度で置き換えた住所をOne-Hotエンコーディングする
one_hot = pd.get_dummies(data['address'])
data = pd.concat([data, one_hot], axis=1)
data = data.drop(['address'], axis=1)

# 欠損値を処理する
data = data.dropna()

# 正規化する
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.drop(['Class'], axis=1))
data_scaled = pd.DataFrame(data_scaled, columns=data.drop(['Class'], axis=1).columns)
data_scaled['Class'] = data['Class']

# モデルを構築する
from sklearn.ensemble import IsolationForest

clf = IsolationForest(n_estimators=100, contamination=0.01)
clf.fit(data_scaled.drop(['Class'], axis=1))
y_pred = clf.predict(data_scaled.drop(['Class'], axis=1))

# 結果を表示する
anomalies = data_scaled.iloc[np.where(y_pred == -1)]
print(anomalies)
