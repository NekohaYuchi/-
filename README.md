# 銀行智慧型信用卡詐騙偵測決策支援系統

**工資系 商管程式設計 期末專題**
組員：陳祥恩、劉邦佑、劉耿宏、伍埞承、張睿

本系統結合金融合規、描述性統計與 XGBoost 演算法，在個資完全去識別化的前提下，協助銀行在營運利潤與客戶體驗間取得最佳商業平衡。

---

## 專案結構

```
├── Credit_Card_Fraud_Detection.ipynb   # 模型訓練 Notebook（張睿）
├── system.py                           # Gradio 版儀表板
├── system_dash.py                      # Dash 版儀表板（側邊欄三頁面）
├── setup.py                            # 一鍵下載預訓練模型
└── README.md
```

> `fraud_model_pipeline.pkl` 與 `test_data.pkl` 不在版本庫中，可用 `setup.py` 自動下載或手動訓練產生。

---

## 快速開始（推薦）

### Step 1：安裝套件

```bash
pip install pandas numpy scikit-learn xgboost joblib plotly
pip install dash dash-bootstrap-components   # Dash 版
pip install gradio                           # Gradio 版（可選）
```

### Step 2：下載預訓練模型

```bash
python setup.py
```

自動從 GitHub Releases 下載 `fraud_model_pipeline.pkl`（56MB）與 `test_data.pkl`（16MB）。

### Step 3：啟動系統

**Dash 版（推薦）**
```bash
python system_dash.py
```
開啟瀏覽器：`http://127.0.0.1:8050`

**Gradio 版（原版）**
```bash
python system.py
```
開啟瀏覽器：`http://127.0.0.1:7860/?__theme=dark`

---

## 自行訓練模型（進階）

不想使用預訓練模型的話，可執行 `Credit_Card_Fraud_Detection.ipynb` 全部 cell 自行訓練（需要 Kaggle 帳號，約 30–60 秒）。

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
| 每日營運報表 | 一鍵產出今日交易績效摘要並下載 CSV |

### Dash 版額外特色

- 側邊欄三頁面導覽：**監控 / 分析 / 報表**
- 頁面載入後自動填入資料（無需手動點擊）
- 所有控制項常駐側邊欄，切換頁面不中斷操作

---

## 模型架構

```
BorderlineSMOTE（過採樣）
  ↓
RandomUnderSampler（欠採樣）
  ↓
XGBoost Classifier
  n_estimators=150, max_depth=6, learning_rate=0.05
```

資料集：[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)（284,807 筆歐洲信用卡交易，詐騙率 0.17%）
