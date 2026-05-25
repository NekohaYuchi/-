import os
os.environ["GRADIO_THEME"] = "dark"
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix

# ==========================================
# 1. 載入模型 Pipeline 與測試數據
# ==========================================
print("正在載入隨機森林模型管線與測試數據...")
try:
    model_pipeline = joblib.load('fraud_model_pipeline.pkl')
    X_test, y_test = joblib.load('test_data.pkl')
    print("數據載入成功！")
except FileNotFoundError as e:
    print(f"錯誤：找不到模型檔案，請確保檔案在相同目錄下。細節: {e}")
    raise

# 預先計算整體的預測機率，優化介面連動效能
y_proba_all = model_pipeline.predict_proba(X_test)[:, 1]

# ==========================================
# 2. 數據視覺化前處理 (使用 NumPy 陣列確保前端圖表 100% 穩定)
# ==========================================
xgb_classifier = model_pipeline.named_steps['classifier']
importances = xgb_classifier.feature_importances_
feature_names = X_test.columns

# 提取前 10 大核心風控特徵
indices = np.argsort(importances)[::-1][:10]
top_features = feature_names[indices]
top_importances = importances[indices]

# 提取特徵空間投影數據
v1_all = X_test['V1'].values
v2_all = X_test['V2'].values
amount_all = X_test['Amount'].values # 供金額分析圖使用
y_all = y_test.values

fraud_positions = np.where(y_all == 1)[0]
safe_positions = np.where(y_all == 0)[0]

np.random.seed(42)
sampled_safe_positions = np.random.choice(safe_positions, 500, replace=False)
plot_positions = np.concatenate([fraud_positions, sampled_safe_positions])

# 建立極度乾淨、無 Pandas Index 衝突的繪圖資料表
df_pca_plot = pd.DataFrame({
    'V1': v1_all[plot_positions],
    'V2': v2_all[plot_positions],
    'Amount': amount_all[plot_positions],
    'Class': ['Fraud (詐騙)' if y == 1 else 'Safe (正常)' for y in y_all[plot_positions]]
})

# 新增功能 2：預先試算全體閥值的利潤曲線數據，避免移動滑桿時即時大重算導致卡頓
thresholds = np.arange(0.01, 1.00, 0.01)
profit_list = []
cost_per_fn_fixed = 250
cost_per_fp_fixed = 15

for t in thresholds:
    y_pred_t = (y_proba_all >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    if cm_t.size == 4:
        tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    else:
        tn_t, fp_t, fn_t, tp_t = len(y_test) - sum(y_pred_t), 0, 0, sum(y_pred_t)
    benefit_t = (tp_t * cost_per_fn_fixed) - ((fp_t * cost_per_fp_fixed) + (fn_t * cost_per_fn_fixed))
    profit_list.append(benefit_t)

df_profit_curve = pd.DataFrame({
    'Threshold': thresholds,
    'Profit': profit_list
})

history_data = []

# ==========================================
# 3. 核心決策與商管邏輯引擎
# ==========================================
# 系統的短期記憶，用來記住畫面上最新的一筆交易
current_tx_state = {"id": None, "score": 0.0} 

def handle_scenario(scenario_name):
    """商管場景快捷鍵：防呆優化版，自動捕捉關鍵字"""
    if "嚴格資安防禦" in scenario_name: return 0.20
    elif "客戶體驗優先" in scenario_name: return 0.80
    else: return 0.50

def process_ui_updates(threshold, is_new_transaction=False):
    """引擎本體：根據 is_new_transaction 決定要不要抽新牌"""
    global history_data, current_tx_state
    
    # 3.1 抽樣與記憶邏輯 (報告 Demo 專用 30% 高機率版)
    if is_new_transaction or current_tx_state["id"] is None:
        # 情況 A：使用者按了按鈕，必須抽一筆新交易 (使用作弊機率)
        if np.random.rand() < 0.30:
            fraud_indices = np.where(y_all == 1)[0]
            sample_idx = np.random.choice(fraud_indices)
        else:
            safe_indices = np.where(y_all == 0)[0]
            sample_idx = np.random.choice(safe_indices)
            
        current_sample = X_test.iloc[[sample_idx]]
        risk_score = float(model_pipeline.predict_proba(current_sample)[0][1])
        tx_id = f"#TX-{np.random.randint(10000, 99999)}"
        
        # 把這筆新交易存進系統記憶中
        current_tx_state["id"] = tx_id
        current_tx_state["score"] = risk_score
    else:
        # 情況 B：使用者只是拉動滑桿，直接從記憶中提取當前這筆交易
        tx_id = current_tx_state["id"]
        risk_score = current_tx_state["score"]
    
    # 3.2 狀態判定與 SOP 生成
    is_system_fraud = risk_score >= threshold
    status_text = "🚨 詐騙警示 (FRAUD)" if is_system_fraud else "✅ 安全交易 (SAFE)"
    
    if is_system_fraud:
        action_sop = (
            "[風控系統核心響應機制觸發]:\n"
            "1. 該筆交易已被即時攔截，暫停授權發放。\n"
            "2. 系統已自動透過 LINE/簡訊 發送消費確認通知給持卡人。\n"
            "3. 風控中心已自動將此案派發工單至『一線客服部』進行人工外撥核對。"
        )
    else:
        action_sop = (
            "[標準商務流程放行]:\n"
            "1. 交易通過隨機森林演算法風控比對，毫秒級授權成功。\n"
            "2. 計入當日持卡人正常消費信用額度，保障刷卡順暢體驗。"
        )
    
    # 新增功能 1：計算人工審查工單處置優先級
    if risk_score > 0.90:
        priority_text = "Level 1 (緊急阻斷)"
    elif risk_score >= threshold:
        priority_text = "Level 2 (人工確認)"
    else:
        priority_text = "Level 3 (自動化驗證)"
        
    # 3.3 監控表格更新邏輯 (納入功能 1 欄位)
    if is_new_transaction or len(history_data) == 0:
        # 如果是新交易，就把資料插入到表格最上方
        history_data.insert(0, [tx_id, round(risk_score, 4), status_text, priority_text])
        if len(history_data) > 5:
            history_data.pop()
    else:
        # 如果只是拉滑桿，就只修改表格最上方那一筆的判定狀態與優先級，不增加新列
        history_data[0][2] = status_text
        history_data[0][3] = priority_text
        
    df_stream = pd.DataFrame(history_data, columns=["交易ID", "風險評分", "系統判定狀態", "處置優先級"])
    
    # 3.4 商業利潤模型試算 (每次都依據新閥值重新計算全域)
    cost_per_fn = 250
    cost_per_fp = 15
    y_pred_all = (y_proba_all >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_all)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = len(y_test) - sum(y_pred_all), 0, 0, sum(y_pred_all)
        
    total_fraud_loss_prevented = tp * cost_per_fn
    total_operational_friction_cost = (fp * cost_per_fp) + (fn * cost_per_fn)
    net_benefit = total_fraud_loss_prevented - total_operational_friction_cost
    
    profit_text = (
        f"全局財務效益試算:\n"
        f"- 成功防堵呆帳: +${total_fraud_loss_prevented:,} USD (攔截 {tp} 筆)\n"
        f"- 營收摩擦與漏抓成本: -${total_operational_friction_cost:,} USD\n"
        f"銀行風控純防護收益淨值: ${net_benefit:,} USD"
    )

    # 3.5 圖表繪製
    fig_importance = px.bar(
        x=top_importances, y=top_features, orientation='h',
        title="XGBoost 全局核心風控特徵 (Top 10)",
        labels={'x': '相對重要性得分 (Relative Importance)', 'y': '加密特徵欄位 (PCA)'},
        color=top_importances, color_continuous_scale='Viridis'
    )
    fig_importance.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(l=50, r=20, b=40, t=40))

    fig_amount = px.box(
        df_pca_plot, x='Class', y='Amount', color='Class',
        color_discrete_map={'Safe (正常)': '#22c55e', 'Fraud (詐騙)': '#ef4444'},
        title="正常交易 vs 詐騙交易之金額規模分佈分析",
        labels={'Amount': '標準化金額 (Amount Scaled)', 'Class': '交易類別'}
    )
    fig_amount.update_layout(margin=dict(l=40, r=20, b=40, t=40), showlegend=False)

    # 新增功能 2：動態利潤最佳化曲線圖繪製
    fig_curve = px.line(
        df_profit_curve, x='Threshold', y='Profit',
        title="動態利潤最佳化曲線圖",
        labels={'Threshold': '判定閥值 (Threshold)', 'Profit': '預期淨利潤 (USD)'}
    )
    # 在當前閥值位置加上一個顯眼的定位紅點
    fig_curve.add_scatter(
        x=[threshold], y=[net_benefit],
        mode='markers',
        marker=dict(size=12, color='#ef4444'),
        name='當前決策點'
    )
    
    # 找出全體利潤的最大與最小值，用來計算固定的 Y 軸邊界留白
    min_p = min(profit_list)
    max_p = max(profit_list)
    p_range = max_p - min_p if (max_p - min_p) != 0 else 10000
    
    # 鎖定 X 軸與 Y 軸的範圍，強迫座標軸固定，讓紅點沿著曲線移動
    fig_curve.update_layout(
        xaxis=dict(range=[0, 1], fixedrange=True),
        yaxis=dict(range=[min_p - p_range * 0.05, max_p + p_range * 0.05], fixedrange=True),
        margin=dict(l=40, r=20, b=40, t=40),
        showlegend=False
    )

    return status_text, action_sop, profit_text, df_stream, fig_importance, fig_amount, fig_curve

def on_simulate_click(threshold):
    return process_ui_updates(threshold, is_new_transaction=True)

def on_threshold_change(threshold):
    return process_ui_updates(threshold, is_new_transaction=False)

# ==========================================
# 4. 建立 Gradio 介面視覺版面
# ==========================================

# 已移除不相容的 js 參數，改在程式結尾由終端機印出深色模式網址
with gr.Blocks(theme=gr.themes.Default(), title="銀行智慧型信用卡防詐系統") as demo:
    
    gr.Markdown(
        """
        # 銀行智慧型信用卡詐騙偵測決策支援系統 (Intelligent Fraud Decision Support System)
        **工資系 商管程式設計 期末專題報告展示控制台 (組員：陳祥恩、劉邦佑、劉耿宏、伍埞承、張睿)**
        本智慧系統結合 金融合規、數據描述性統計與XGBoost演算法，展示如何在個資完全去識別化的加密前提下，協助銀行在營運利潤與客戶體驗間取得最佳商業平衡。
        """
    )
    
    with gr.Row():
        # 左側控制面板 (1/3 寬度)
        with gr.Column(scale=1):
            gr.Markdown("### 決策策略控制中心")
            
            scenario_dropdown = gr.Dropdown(
                choices=[
                    "損益平衡最優化 (F1-Score 最佳化)", 
                    "嚴格資安防禦 (寧可錯殺，絕不放過)", 
                    "客戶體驗優先 (降低誤擋率與摩擦)"
                ],
                value="損益平衡最優化 (F1-Score 最佳化)",
                label="銀行當前營運方針 (Dynamic Strategy)"
            )
            
            threshold_slider = gr.Slider(
                minimum=0.01, maximum=0.99, value=0.50, step=0.01, 
                label="動態風險判定閥值 (Sensitivity Threshold)"
            )
            simulate_btn = gr.Button("接收即時交易 (Simulate Transaction)", variant="primary")
            
            gr.Markdown("### 本次決策核心響應")
            out_status = gr.Textbox(label="單筆交易判定結果", text_align="center")
            
            out_sop = gr.Textbox(label="系統自動化處置與營運 SOP (Action Trigger)", lines=4)
            
        # 右側實時監控與利潤試算 (2/3 寬度)
        with gr.Column(scale=2):
            gr.Markdown("### 銀行後台實時交易風險監控流")
            out_stream = gr.Dataframe(headers=["交易ID", "風險評分", "系統判定狀態", "處置優先級"], interactive=False)
            
            gr.Markdown("### 商業利潤動態分析")
            out_profit = gr.Textbox(label="商管利潤模擬器 (動態財務決策支援)", lines=4)

    with gr.Row():
        # 下方視覺化三圖表並列
        with gr.Column(scale=1):
            out_importance_plot = gr.Plot(label="AI 決策黑盒解碼 (XAI)")
        with gr.Column(scale=1):
            out_amount_plot = gr.Plot(label="金融風險暴露分析")
        with gr.Column(scale=1):
            out_curve_plot = gr.Plot(label="動態利潤最佳化曲線")

    # ─── 變數與事件連動綁定 ───
    scenario_dropdown.change(fn=handle_scenario, inputs=[scenario_dropdown], outputs=[threshold_slider])

    # 點擊按鈕時：強制抽新交易 (is_new_transaction=True)
    simulate_btn.click(
        fn=on_simulate_click, inputs=[threshold_slider],
        outputs=[out_status, out_sop, out_profit, out_stream, out_importance_plot, out_amount_plot, out_curve_plot]
    )
    
    # 拖動滑桿時：只重新計算畫面上的舊交易與利潤 (is_new_transaction=False)
    threshold_slider.change(
        fn=on_threshold_change, inputs=[threshold_slider],
        outputs=[out_status, out_sop, out_profit, out_stream, out_importance_plot, out_amount_plot, out_curve_plot]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("深色模式網址已生成，請複製此連結至瀏覽器開啟：")
    print("http://127.0.0.1:7860/?__theme=dark")
    print("="*60 + "\n")
    demo.launch(share=False)