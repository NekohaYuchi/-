import os
os.environ["GRADIO_THEME"] = "dark"
import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score

# ==========================================
# 1. 載入模型 Pipeline 與測試數據
# ==========================================
print("正在載入xgboost模型與測試數據...")
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
# 2. 數據視覺化前處理
# ==========================================
xgb_classifier = model_pipeline.named_steps['classifier']
importances = xgb_classifier.feature_importances_
feature_names = X_test.columns

# 提取前 10 大核心風控特徵
indices = np.argsort(importances)[::-1][:10]
top_features = feature_names[indices]
top_importances = importances[indices]

y_all = y_test.values

# ==========================================
# 預先試算全體閥值的混淆矩陣參數，避免移動滑桿時大重算導致卡頓
# ==========================================
thresholds_array = np.arange(0.01, 1.00, 0.01)
tp_array = np.zeros_like(thresholds_array)
fp_array = np.zeros_like(thresholds_array)
fn_array = np.zeros_like(thresholds_array)
tn_array = np.zeros_like(thresholds_array)

for i, t in enumerate(thresholds_array):
    y_pred_t = (y_proba_all >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    if cm_t.size == 4:
        tn_array[i], fp_array[i], fn_array[i], tp_array[i] = cm_t.ravel()
    else:
        tn_array[i], fp_array[i], fn_array[i], tp_array[i] = len(y_test) - sum(y_pred_t), 0, 0, sum(y_pred_t)

curve_stats = pd.DataFrame({
    'Threshold': thresholds_array,
    'TP': tp_array,
    'FP': fp_array,
    'FN': fn_array,
    'TN': tn_array
})

history_data = []

# 模擬銀行最近交易池 (每日營運報表基礎資料)
WINDOW_SIZE = 15786
recent_transaction_indices = list(
    np.random.choice(
        np.arange(len(y_test)),
        WINDOW_SIZE,
        replace=False
    )
)

# ==========================================
# 3. 核心決策與商管邏輯引擎
# ==========================================
current_tx_state = {"id": None, "score": 0.0} 

def handle_scenario(scenario_name):
    if "嚴格資安防禦" in scenario_name: return 0.20
    elif "客戶體驗優先" in scenario_name: return 0.80
    else: return 0.50

def process_ui_updates(threshold, cost_fn, cost_fp, is_new_transaction=False):
    global history_data, current_tx_state
    
    # 3.1 抽樣與記憶邏輯
    if is_new_transaction or current_tx_state["id"] is None:
        if np.random.rand() < 0.30:
            fraud_indices = np.where(y_all == 1)[0]
            sample_idx = np.random.choice(fraud_indices)
            recent_transaction_indices.append(sample_idx)
            if len(recent_transaction_indices) > WINDOW_SIZE:
                recent_transaction_indices.pop(0)
        else:
            safe_indices = np.where(y_all == 0)[0]
            sample_idx = np.random.choice(safe_indices)
            
        current_sample = X_test.iloc[[sample_idx]]
        risk_score = float(model_pipeline.predict_proba(current_sample)[0][1])
        tx_id = f"#TX-{np.random.randint(10000, 99999)}"
        
        current_tx_state["id"] = tx_id
        current_tx_state["score"] = risk_score
    else:
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
            "1. 交易通過 XGBoost 演算法風控比對，毫秒級授權成功。\n"
            "2. 計入當日持卡人正常消費信用額度，保障刷卡順暢體驗。"
        )
    
    if risk_score > 0.90:
        priority_text = "Level 1 (緊急阻斷)"
    elif risk_score >= threshold:
        priority_text = "Level 2 (人工確認)"
    else:
        priority_text = "Level 3 (自動化驗證)"
        
    # 3.3 監控表格更新邏輯
    if is_new_transaction or len(history_data) == 0:
        history_data.insert(0, [tx_id, round(risk_score, 4), status_text, priority_text])
        if len(history_data) > 5:
            history_data.pop()
    else:
        history_data[0][2] = status_text
        history_data[0][3] = priority_text
        
    # 為了讓前端固定顯示 5 筆，若不足 5 筆則補上空值 (解決 Gradio 渲染問題)
    display_data = history_data.copy()
    while len(display_data) < 5:
        display_data.append(["-", "-", "-", "-"])
        
    df_stream = pd.DataFrame(display_data, columns=["交易ID", "風險評分", "系統判定狀態", "處置優先級"])
    
    # 3.4 商業利潤模型試算
    y_pred_all = (y_proba_all >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_all)
    
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = len(y_test) - sum(y_pred_all), 0, 0, sum(y_pred_all)
        
    total_fraud_loss_prevented = tp * cost_fn
    total_operational_friction_cost = (fp * cost_fp) + (fn * cost_fn)
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

    # 動態利潤最佳化曲線圖繪製
    df_profit_curve = curve_stats.copy()
    df_profit_curve['Profit'] = (df_profit_curve['TP'] * cost_fn) - \
                                ((df_profit_curve['FP'] * cost_fp) + (df_profit_curve['FN'] * cost_fn))

    fig_curve = px.line(
        df_profit_curve, x='Threshold', y='Profit',
        title="動態利潤最佳化曲線圖",
        labels={'Threshold': '判定閥值 (Threshold)', 'Profit': '預期淨利潤 (USD)'}
    )
    fig_curve.add_scatter(
        x=[threshold], y=[net_benefit],
        mode='markers',
        marker=dict(size=12, color='#ef4444'),
        name='當前決策點'
    )
    
    min_p = df_profit_curve['Profit'].min()
    max_p = df_profit_curve['Profit'].max()
    p_range = max_p - min_p if (max_p - min_p) != 0 else 10000
    
    fig_curve.update_layout(
        xaxis=dict(range=[0, 1], fixedrange=True),
        yaxis=dict(range=[min_p - p_range * 0.05, max_p + p_range * 0.05], fixedrange=True),
        margin=dict(l=40, r=20, b=40, t=40),
        showlegend=False
    )

    return status_text, action_sop, profit_text, df_stream, fig_importance, fig_curve

def on_simulate_click(threshold, cost_fn, cost_fp):
    return process_ui_updates(threshold, cost_fn, cost_fp, is_new_transaction=True)

def on_threshold_change(threshold, cost_fn, cost_fp):
    return process_ui_updates(threshold, cost_fn, cost_fp, is_new_transaction=False)

# ==========================================
# 新增功能 9：每日營運報表生成邏輯
# ==========================================
def generate_daily_report(threshold, cost_fn, cost_fp):
    # 取出今日的 500 筆模擬交易池
    recent_y = y_test.iloc[recent_transaction_indices]
    recent_proba = y_proba_all[recent_transaction_indices]
    
    # 計算今日交易在當前閥值下的表現
    y_pred = (recent_proba >= threshold).astype(int)
    cm = confusion_matrix(recent_y, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = len(recent_y) - sum(y_pred), 0, 0, sum(y_pred)
        
    saved_loss = tp * cost_fn
    
    # 計算模型 AUC
    auc_score = roc_auc_score(y_test, y_proba_all)
    
    # 尋找全局最佳閥值
    profit_array = (curve_stats['TP'] * cost_fn) - ((curve_stats['FP'] * cost_fp) + (curve_stats['FN'] * cost_fn))
    best_idx = np.argmax(profit_array)
    best_threshold = curve_stats['Threshold'].iloc[best_idx]
    
    summary_text = (
        f"=== 今日風控摘要 ===\n\n"
        f"總交易量：{len(recent_y)} 筆\n"
        f"攔截詐騙：{tp} 筆\n"
        f"誤攔交易：{fp} 筆\n"
        f"成功阻止損失：${saved_loss:,}\n"
        f"最佳Threshold：{best_threshold:.2f}\n"
        f"模型AUC：{auc_score:.3f}"
    )
    
    # 產生詳細報表 DataFrame
    report_df = pd.DataFrame({
        "Transaction_ID": [f"TX-{np.random.randint(100000, 999999)}" for _ in range(len(recent_y))],
        "Risk_Score": np.round(recent_proba, 4),
        "Is_Fraud_Real": recent_y.values,
        "System_Blocked": y_pred
    })
    
    # 增加營運標籤
    report_df["Status"] = report_df.apply(
        lambda x: "True Positive (成功攔截)" if x["Is_Fraud_Real"]==1 and x["System_Blocked"]==1 else 
                  "False Positive (客戶誤擋)" if x["Is_Fraud_Real"]==0 and x["System_Blocked"]==1 else 
                  "False Negative (漏抓損失)" if x["Is_Fraud_Real"]==1 and x["System_Blocked"]==0 else 
                  "True Negative (安全放行)", axis=1
    )
    
    # 匯出 CSV 檔案，加上 utf-8-sig 以確保 Excel 開啟中文不會亂碼
    file_path = "daily_report.csv"
    report_df.to_csv(file_path, index=False, encoding="utf-8-sig")
    
    return summary_text, file_path

# ==========================================
# 4. 建立 Gradio 介面視覺版面
# ==========================================
with gr.Blocks(theme=gr.themes.Default(), title="銀行智慧型信用卡防詐系統") as demo:
    
    gr.Markdown(
        """
        # 銀行智慧型信用卡詐騙偵測決策支援系統 (Intelligent Fraud Decision Support System)
        **工資系 商管程式設計 期末專題報告展示控制台 (組員：陳祥恩、劉邦佑、劉耿宏、伍埞承、張睿)**
        本智慧系統結合 金融合規、數據描述性統計與XGBoost演算法，展示如何在個資完全去識別化的加密前提下，協助銀行在營運利潤與客戶體驗間取得最佳商業平衡。
        """
    )
    
    with gr.Row():
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
            
            gr.Markdown("#### 誤判成本動態設定 (Cost Analysis)")
            cost_fn_slider = gr.Slider(
                minimum=50, maximum=1000, value=250, step=10, 
                label="漏抓成本 (FN Cost - 詐騙損失/筆)"
            )
            cost_fp_slider = gr.Slider(
                minimum=1, maximum=100, value=15, step=1, 
                label="誤判成本 (FP Cost - 營運摩擦/筆)"
            )

            simulate_btn = gr.Button("接收即時交易 (Simulate Transaction)", variant="primary")
            
            gr.Markdown("### 本次決策核心響應")
            out_status = gr.Textbox(label="單筆交易判定結果", text_align="center")
            out_sop = gr.Textbox(label="系統自動化處置與營運 SOP (Action Trigger)", lines=4)
            
        with gr.Column(scale=2):
            gr.Markdown("### 銀行後台實時交易風險監控流")
            # 加上 row_count=(5, "fixed") 強制固定顯示列數
            out_stream = gr.Dataframe(headers=["交易ID", "風險評分", "系統判定狀態", "處置優先級"], interactive=False, row_count=(5, "fixed"))
            
            gr.Markdown("### 商業利潤動態分析")
            out_profit = gr.Textbox(label="商管利潤模擬器 (動態財務決策支援)", lines=4)

    with gr.Row():
        # 左：XAI 解釋圖
        with gr.Column(scale=1):
            out_importance_plot = gr.Plot(label="AI 決策黑盒解碼 (XAI)")
            
        # 中：新增的營運報表生成區塊 (取代了原本的金額暴露圖)
        with gr.Column(scale=1):
            gr.Markdown("### 每日營運報表")
            gr.Markdown("一鍵產出今日的績效報告及 CSV 清單供後端查帳使用。")
            generate_report_btn = gr.Button("產生管理報表", variant="secondary")
            report_summary = gr.Textbox(label="今日風控摘要", interactive=False, lines=8)
            report_file = gr.File(label="下載詳細報表 (.csv)")
            
        # 右：利潤最佳化曲線
        with gr.Column(scale=1):
            out_curve_plot = gr.Plot(label="動態利潤最佳化曲線")

    # ─── 變數與事件連動綁定 ───
    scenario_dropdown.change(fn=handle_scenario, inputs=[scenario_dropdown], outputs=[threshold_slider])

    # 參數統整傳遞
    update_inputs = [threshold_slider, cost_fn_slider, cost_fp_slider]
    update_outputs = [out_status, out_sop, out_profit, out_stream, out_importance_plot, out_curve_plot]

    simulate_btn.click(
        fn=on_simulate_click, inputs=update_inputs,
        outputs=update_outputs
    )
    
    threshold_slider.change(fn=on_threshold_change, inputs=update_inputs, outputs=update_outputs)
    cost_fn_slider.change(fn=on_threshold_change, inputs=update_inputs, outputs=update_outputs)
    cost_fp_slider.change(fn=on_threshold_change, inputs=update_inputs, outputs=update_outputs)

    # 綁定產生報表按鈕的事件
    generate_report_btn.click(
        fn=generate_daily_report,
        inputs=[threshold_slider, cost_fn_slider, cost_fp_slider],
        outputs=[report_summary, report_file]
    )

if __name__ == "__main__":
    print("\n" + "="*60)
    print("深色模式網址已生成，請複製此連結至瀏覽器開啟：")
    print("http://127.0.0.1:7860/?__theme=dark")
    print("="*60 + "\n")
    demo.launch(share=False)