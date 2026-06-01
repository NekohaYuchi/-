# Dash 儀表板改寫設計文件

**日期**：2026-06-01  
**專案**：銀行智慧型信用卡詐騙偵測決策支援系統  
**目標**：將現有 `system.py`（Gradio）改寫為 `system_dash.py`（Dash），提升視覺品質與架構彈性。

---

## 決策摘要

| 項目 | 決定 |
|------|------|
| 框架 | Dash + dash-bootstrap-components |
| 版面方向 | 側邊欄導覽（固定左側，頁面內容在右） |
| 主題 | 深海藍 Slate Dark（`#0f172a` / `#1e293b`，強調色 `#3b82f6`） |
| 頁面數量 | 3 個（監控、分析、報表） |
| 實作結構 | 單一檔案 `system_dash.py`，`dcc.Location` 路由 |
| Bug 修復範圍 | 僅修「啟動時畫面空白」，其餘 bug 不動 |
| 新增/取代 | 新增 `system_dash.py`，原 `system.py` 保留不動 |

---

## 版面架構

```
┌─────────────────────────────────────────────────────┐
│  側邊欄（70px）  │  頁面內容區                       │
│  ─────────────  │  ──────────────────────────────── │
│  🏦 系統標題     │  KPI 卡列（今日攔截 / 防護金額 /  │
│                 │           AUC / 當前閥值）          │
│  [監控]         │  ──────────────────────────────── │
│  [分析]         │  頁面內容（依側邊欄切換）           │
│  [報表]         │                                    │
└─────────────────────────────────────────────────────┘
```

---

## 三個頁面規格

### 頁面 1：監控（/）

- 左欄（4 col）：
  - 銀行營運方針下拉選單（3 個策略選項）
  - 動態風險判定閥值滑桿（0.01–0.99）
  - 漏抓成本 FN Cost 滑桿（50–1000）
  - 誤判成本 FP Cost 滑桿（1–100）
  - 「接收即時交易」按鈕
  - 單筆判定結果文字
  - SOP 處置說明文字
- 右欄（8 col）：
  - 即時交易監控表格（固定 5 行，DataTable）
  - 商業利潤試算文字

### 頁面 2：分析（/analysis）

- 左半（6 col）：XGBoost 特徵重要性橫條圖（Plotly，啟動時預計算，不隨 callback 重繪）
- 右半（6 col）：動態利潤最佳化曲線（Plotly，隨閥值與成本參數更新）

> 分析頁的滑桿數值從 `dcc.Store` 讀取，與監控頁共享狀態。

### 頁面 3：報表（/report）

- 「產生管理報表」按鈕
- 今日風控摘要文字（總交易量、攔截筆數、誤攔筆數、防護金額、最佳閥值、AUC）
- 「下載報表 CSV」按鈕（`dcc.Download`，點擊觸發）
- CSV 檔名：`daily_report.csv`（固定，不加時間戳）

---

## KPI 卡（頂部，全頁面共用）

四張卡，跟著閥值與成本滑桿即時更新：

| 卡片 | 數值來源 |
|------|----------|
| 今日攔截筆數 | `confusion_matrix` TP，以當前閥值計算 |
| 防護金額 | TP × FN Cost |
| 模型 AUC | `roc_auc_score`，固定值，啟動時算一次 |
| 當前閥值 | 直接讀閥值滑桿 |

---

## 資料流與 State 管理

### 預載資料（啟動時，server-side 全域）

```python
model_pipeline     # 從 fraud_model_pipeline.pkl 載入
X_test, y_test     # 從 test_data.pkl 載入
y_proba_all        # 預先計算全部預測機率
curve_stats        # 預先計算 99 個閥值的混淆矩陣統計
fig_importance     # 預先繪製特徵重要性圖（固定，不重繪）
auc_score          # 啟動時算一次
recent_transaction_indices  # 15k 筆，server-side 全域變數
```

### Client-side State（`dcc.Store`）

| Store ID | 內容 | 用途 |
|----------|------|------|
| `tx-state` | `{id, score}` | 記憶當前交易，滑桿移動時不重新抽樣 |
| `history-store` | 最近 5 筆交易列表 | 交易表格資料來源 |
| `settings-store` | `{threshold, cost_fn, cost_fp}` | 監控頁滑桿數值，供分析頁利潤曲線跨頁讀取 |
| `report-store` | `{filename, content}` | 暫存報表 CSV 內容供下載 |

### Callback 清單

```
[dcc.Interval(interval=1, max_intervals=1)]  ← 觸發一次初始化（修復空白畫面）
[simulate-btn 點擊]
[threshold-slider / cost-fn-slider / cost-fp-slider 變動]
    → 更新：KPI卡 × 4、交易表格、判定文字、SOP、利潤文字、利潤曲線
    → 更新：tx-state、history-store

[scenario-dropdown 變動]
    → 更新：threshold-slider 數值

[report-btn 點擊]
    → 更新：report-summary 文字、report-store、download-btn 顯示

[download-btn 點擊]
    → 觸發：dcc.Download（從 report-store 讀取）
```

---

## 配色規格

```python
COLORS = {
    "bg_page":    "#0f172a",   # 頁面背景
    "bg_card":    "#1e293b",   # 卡片背景
    "bg_sidebar": "#1e293b",   # 側邊欄背景
    "accent":     "#3b82f6",   # 主強調色（藍）
    "text":       "#e2e8f0",   # 主文字
    "text_muted": "#64748b",   # 次要文字
    "fraud":      "#ef4444",   # 詐騙警示（紅）
    "safe":       "#22c55e",   # 安全放行（綠）
    "border":     "#334155",   # 邊框
}
```

---

## 相依套件

需額外安裝（base Anaconda 環境）：
- `dash`
- `dash-bootstrap-components`

其餘套件（`plotly`、`pandas`、`numpy`、`joblib`、`scikit-learn`、`xgboost`）已安裝。
