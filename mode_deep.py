import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

# 正常データと異常データに分割する
normal_data = data[data['Class'] == 0].drop(['Class'], axis=1)
anomaly_data = data[data['Class'] == 1].drop(['Class'], axis=1)

# 正常データを学習用データとテスト用データに分割する
train_data = normal_data.sample(frac=0.8, random_state=123)
test_data = normal_data.drop(train_data.index)

# 学習用データを正常データと異常データに分割する
train_normal_data = train_data.values
train_anomaly_data = anomaly_data.values

# Autoencoderモデルを構築する
input_dim = train_normal_data.shape[1]
encoding_dim = 20
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh", activity_regularizer=tf.keras.regularizers.l1(10e-5))(input_layer)
encoder = tf.keras.layers.Dense(int(encoding_dim / 2), activation="relu")(encoder)
decoder = tf.keras.layers.Dense(int(encoding_dim / 2), activation='tanh')(encoder)
decoder = tf.keras.layers.Dense(input_dim, activation='relu')(decoder)
autoencoder = tf.keras.models.Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Autoencoderモデルを学習する
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
autoencoder.fit(train_normal_data, train_normal_data, epochs=50, batch_size=32, shuffle=True, validation_data=(test_data, test_data), callbacks=[es])

# 異常検知を行う
test_anomaly_data = anomaly_data.values
y_pred = autoencoder.predict(test_anomaly_data)
mse = np.mean(np.power(test_anomaly_data - y_pred, 2), axis=1)
mse_df = pd.DataFrame({'mse': mse, 'true_class': 1})
threshold = mse_df['mse'].quantile(0.99)
anomalies = mse_df[mse_df['mse'] > threshold]

# 結果を表示する
print(anomalies)
