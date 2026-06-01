import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb
import joblib

print("正在從 Kaggle 下載信用卡詐騙資料集...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("下載完成，路徑:", path)

df = pd.read_csv(os.path.join(path, "creditcard.csv"))
print(f"資料讀入完成：{df.shape[0]} 筆，{df.shape[1]} 個欄位")

processing_df = df.copy()
scaler = StandardScaler()
processing_df[['Time', 'Amount']] = scaler.fit_transform(processing_df[['Time', 'Amount']])

X = processing_df.drop('Class', axis=1)
y = processing_df['Class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"訓練集: {X_train.shape[0]} 筆 / 測試集: {X_test.shape[0]} 筆")

over_sampler = BorderlineSMOTE(sampling_strategy=0.01, random_state=42, kind='borderline-1')
under_sampler = RandomUnderSampler(sampling_strategy=0.05, random_state=42)
xgb_model = xgb.XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.05,
    scale_pos_weight=5, subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', n_jobs=-1
)
optimized_pipeline = Pipeline([
    ('smote', over_sampler),
    ('under', under_sampler),
    ('classifier', xgb_model)
])

print("正在訓練模型（約需 30-60 秒）...")
optimized_pipeline.fit(X_train, y_train)
print("模型訓練完成！")

joblib.dump(optimized_pipeline, 'fraud_model_pipeline.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("已儲存：fraud_model_pipeline.pkl、test_data.pkl、scaler.pkl")
