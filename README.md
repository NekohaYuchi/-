# 銀行智慧型信用卡詐騙偵測決策支援系統

**工資系 商管程式設計 期末專題**
組員：陳祥恩、劉邦佑、劉耿宏、伍埞承、張睿

本系統結合金融合規、描述性統計與 XGBoost 演算法，在個資完全去識別化的前提下，協助銀行在營運利潤與客戶體驗間取得最佳商業平衡。

---

## 專案結構

```
├── Credit_Card_Fraud_Detection.ipynb   # 模型訓練 Notebook
├── train_model.py                      # 本地訓練腳本
├── system_dash.py                      # Dash 儀表板（側邊欄三頁面）
├── setup.py                            # 一鍵下載預訓練模型
└── README.md
```

> `fraud_model_pipeline.pkl`、`test_data.pkl`、`scaler.pkl` 不在版本庫中，可用 `setup.py` 自動下載或執行 `train_model.py` 自行訓練產生。

---

## 快速開始（推薦）

### Step 1：安裝套件

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost joblib
pip install dash dash-bootstrap-components plotly openpyxl
```

### Step 2：下載預訓練模型

```bash
python setup.py
```

自動從 GitHub Releases 下載 `fraud_model_pipeline.pkl`（47MB）、`test_data.pkl`（16MB）、`scaler.pkl`。

### Step 3：啟動系統

```bash
python system_dash.py
```

開啟瀏覽器：`http://127.0.0.1:8050`

---

## 自行訓練模型（進階）

```bash
python train_model.py
```

需要 Kaggle 帳號，資料集自動下載，訓練約需 2–5 分鐘（含 5-Fold CV）。

---

## 環境需求

- Python 3.x（Anaconda 環境建議）

---

## 系統功能

| 功能 | 說明 |
|------|------|
| 即時交易模擬 | 從測試集隨機抽樣，使用 XGBoost 即時預測詐騙機率 |
| 動態風險閥值 | 滑桿調整判定敏感度（0.01–0.99），KPI 卡即時更新 |
| 銀行策略切換 | 損益平衡 / 嚴格資安防禦 / 客戶體驗優先 三種模式 |
| 商業利潤試算 | 依漏抓成本（FN）與誤判成本（FP）計算銀行淨防護收益 |
| 動態利潤曲線 | 顯示不同閥值下的預期獲利，標記當前決策點 |
| XAI 特徵解釋 | XGBoost 前 10 大核心風控特徵重要性（可解釋 AI） |
| 每日營運報表 | 一鍵產出今日交易績效摘要並下載 Excel（含三個工作表） |
| 交易詳細面板 | 點選監控表格任一列，展開該筆交易的風險分析與 SOP |

### 側邊欄三頁面導覽

- **監控**：即時交易流、判定結果、SOP、利潤試算
- **分析**：XAI 特徵重要性圖、動態利潤最佳化曲線
- **報表**：每日營運摘要、高風險交易清單、Excel 下載

---

## 模型架構

```
BorderlineSMOTE（過採樣，sampling_strategy=0.1）
  ↓
XGBoost Classifier
  n_estimators=150, max_depth=6, learning_rate=0.05
```

**訓練流程：**
1. 先切分訓練 / 測試集（8:2，stratify），再標準化（避免 Data Leakage）
2. 從訓練集切出 15% 作為 Validation Set（供門檻搜尋）
3. 5-Fold 交叉驗證評估模型穩定性

資料集：[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)（284,807 筆歐洲信用卡交易，詐騙率 0.17%）
