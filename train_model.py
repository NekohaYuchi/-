import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.pipeline import Pipeline
import xgboost as xgb
import joblib

print("正在從 Kaggle 下載信用卡詐騙資料集...")
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("下載完成，路徑:", path)

df = pd.read_csv(os.path.join(path, "creditcard.csv"))
print(f"資料讀入完成：{df.shape[0]} 筆，{df.shape[1]} 個欄位")

# 先切分特徵與標籤，再標準化，避免 Data Leakage
X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train[['Time', 'Amount']] = scaler.fit_transform(X_train[['Time', 'Amount']])
X_test[['Time', 'Amount']] = scaler.transform(X_test[['Time', 'Amount']])

print(f"訓練集: {X_train.shape[0]} 筆 / 測試集: {X_test.shape[0]} 筆")

# BorderlineSMOTE 過採樣，讓詐騙:正常 ≈ 1:10
over_sampler = BorderlineSMOTE(
    sampling_strategy=0.1,
    random_state=42,
    kind='borderline-1'
)

xgb_model = xgb.XGBClassifier(
    n_estimators=150, max_depth=6, learning_rate=0.05,
    scale_pos_weight=1, subsample=0.8, colsample_bytree=0.8,
    random_state=42, eval_metric='logloss', n_jobs=-1
)

optimized_pipeline = Pipeline([
    ('smote', over_sampler),
    ('classifier', xgb_model)
])

# 從訓練集切出 15% 作為 Validation Set，供門檻搜尋使用
X_train_final, X_val, y_train_final, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
)

print("正在訓練模型（約需 1-2 分鐘）...")
optimized_pipeline.fit(X_train_final, y_train_final)

# 5-Fold 交叉驗證評估模型穩定性
print("正在執行 5-Fold 交叉驗證（約需數分鐘）...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1 = cross_val_score(optimized_pipeline, X_train_final, y_train_final,
                         cv=skf, scoring='f1', n_jobs=-1)
cv_recall = cross_val_score(optimized_pipeline, X_train_final, y_train_final,
                             cv=skf, scoring='recall', n_jobs=-1)

print(f"5-Fold CV 結果：")
print(f"  F1     : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")
print(f"  Recall : {cv_recall.mean():.4f} ± {cv_recall.std():.4f}")

print("模型訓練完成！")

joblib.dump(optimized_pipeline, 'fraud_model_pipeline.pkl')
joblib.dump((X_test, y_test), 'test_data.pkl')
joblib.dump((X_val, y_val), 'val_data.pkl')
joblib.dump((X_train_final, y_train_final), 'train_data.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("已儲存：fraud_model_pipeline.pkl、test_data.pkl、val_data.pkl、train_data.pkl、scaler.pkl")
