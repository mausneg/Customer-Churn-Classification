import pandas as pd
import tensorflow as tf
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df_churn = pd.read_csv('data\customer_churn_dataset-training-master.csv')
df_churn = pd.concat([df_churn, pd.read_csv('data\customer_churn_dataset-testing-master.csv')], ignore_index=True)
df_churn.dropna(inplace=True)
df_churn.reset_index(drop=True, inplace=True)

df_churn.drop(columns=['CustomerID','Last Interaction','Tenure','Usage Frequency'],inplace=True)

encoder = LabelEncoder()
df_churn['Gender'] = encoder.fit_transform(df_churn['Gender'])

encoder = OneHotEncoder(sparse_output=False)
columns_to_encode = ['Subscription Type','Contract Length']
encoded_df = pd.DataFrame(encoder.fit_transform(df_churn[columns_to_encode]), columns=encoder.get_feature_names_out(columns_to_encode))
df_churn.drop(columns_to_encode, axis=1, inplace=True)
df_churn = pd.concat([df_churn, encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(df_churn.drop(columns=['Churn']), df_churn['Churn'], test_size = 0.2, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size = 0.5, random_state = 42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1024,activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, verbose=2, batch_size=512, validation_data=(X_test, y_test))
val_loss, val_accuracy = model.evaluate(X_val, y_val, batch_size=512)

print(f'Training accuracy: {history.history["accuracy"][-1]}')
print(f'Testing accuracy: {history.history["val_accuracy"][-1]}')
print(f'Validation accuracy: {val_accuracy}')