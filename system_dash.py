import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.metrics import confusion_matrix, roc_auc_score
import dash
from dash import dcc, html, dash_table, Input, Output, State, ctx, no_update
import dash_bootstrap_components as dbc

# ── 配色常數 ──────────────────────────────────────────────
COLORS = {
    "bg_page":    "#0B111A",
    "bg_card":    "#1F2937",
    "bg_sidebar": "#111827",
    "bg_input":   "#273142",
    "accent":     "#1E9FFB",
    "cyan":       "#22D3EE",
    "text":       "#F5F7FA",
    "text_muted": "#9CA3AF",
    "fraud":      "#EF4444",
    "safe":       "#00D084",
    "warning":    "#F59E0B",
    "border":     "#2F3B4A",
}

# ── 載入模型與測試資料 ─────────────────────────────────────
print("載入模型與測試數據...")
model_pipeline = joblib.load("fraud_model_pipeline.pkl")
X_test, y_test = joblib.load("test_data.pkl")
print("載入成功")

y_proba_all = model_pipeline.predict_proba(X_test)[:, 1]
y_all = y_test.values

# 固定 AUC（只算一次）
AUC_SCORE = round(float(roc_auc_score(y_test, y_proba_all)), 3)

# 預先計算 99 個閥值的混淆矩陣統計
thresholds_array = np.arange(0.01, 1.00, 0.01)
tp_arr = np.zeros_like(thresholds_array)
fp_arr = np.zeros_like(thresholds_array)
fn_arr = np.zeros_like(thresholds_array)
tn_arr = np.zeros_like(thresholds_array)

for i, t in enumerate(thresholds_array):
    y_pred_t = (y_proba_all >= t).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    if cm_t.size == 4:
        tn_arr[i], fp_arr[i], fn_arr[i], tp_arr[i] = cm_t.ravel()
    else:
        tn_arr[i] = len(y_test) - sum(y_pred_t)
        tp_arr[i] = sum(y_pred_t)

CURVE_STATS = pd.DataFrame({
    "Threshold": thresholds_array,
    "TP": tp_arr, "FP": fp_arr,
    "FN": fn_arr, "TN": tn_arr,
})

# 預先繪製特徵重要性圖（固定，不在 callback 中重繪）
xgb_clf = model_pipeline.named_steps["classifier"]
importances = xgb_clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]
top_features = X_test.columns[indices]
top_importances = importances[indices]

FIG_IMPORTANCE = px.bar(
    x=top_importances, y=top_features, orientation="h",
    title="XGBoost 全局核心風控特徵 (Top 10)",
    labels={"x": "相對重要性得分", "y": "加密特徵欄位 (PCA)"},
    color=top_importances,
    color_continuous_scale=[[0, "#1a3a5c"], [0.5, "#1E9FFB"], [1, "#22D3EE"]],
)
FIG_IMPORTANCE.update_layout(
    yaxis={"categoryorder": "total ascending",
           "gridcolor": COLORS["border"], "gridwidth": 1},
    xaxis={"gridcolor": COLORS["border"], "gridwidth": 1},
    paper_bgcolor=COLORS["bg_card"],
    plot_bgcolor=COLORS["bg_card"],
    font={"color": COLORS["text"]},
    margin={"l": 10, "r": 10, "b": 30, "t": 40},
    coloraxis_showscale=False,
)

# 每日報表交易池（server-side 全域，15k 筆索引不適合放 dcc.Store）
WINDOW_SIZE = 15786
recent_tx_indices = list(
    np.random.choice(np.arange(len(y_test)), WINDOW_SIZE, replace=False)
)

# ── 版面 helpers ──────────────────────────────────────────

def make_kpi_card(title, value_id, color):
    return dbc.Card([
        dbc.CardBody([
            html.P(title, style={"color": COLORS["text_muted"], "fontSize": "11px",
                                  "marginBottom": "4px"}),
            html.H4(id=value_id, children="—",
                    style={"color": color, "fontWeight": "bold", "marginBottom": 0}),
        ], style={"padding": "12px"})
    ], style={"backgroundColor": COLORS["bg_card"],
              "border": f"1px solid {COLORS['border']}"})


EMPTY_TABLE_ROWS = [
    {"交易ID": "-", "風險評分": "-", "系統判定狀態": "-", "處置優先級": "-"}
] * 5

TABLE_KWARGS = dict(
    style_table={"overflowX": "auto"},
    style_header={
        "backgroundColor": COLORS["bg_sidebar"],
        "color": COLORS["cyan"], "fontWeight": "600",
        "border": f"1px solid {COLORS['border']}",
        "fontSize": "11px", "letterSpacing": "0.05em", "textTransform": "uppercase",
    },
    style_cell={
        "backgroundColor": COLORS["bg_input"],
        "color": COLORS["text_muted"],
        "textAlign": "center", "padding": "9px 8px",
        "border": f"1px solid {COLORS['border']}",
        "fontFamily": "monospace", "fontSize": "13px",
    },
    style_data_conditional=[
        {"if": {"filter_query": '{系統判定狀態} contains "詐騙"'},
         "color": COLORS["fraud"], "fontWeight": "bold"},
        {"if": {"filter_query": '{系統判定狀態} contains "安全"'},
         "color": COLORS["safe"]},
    ],
)


def page_monitor():
    return html.Div([
        dbc.Row([
            dbc.Col([
                html.P(id="status-display", children="—",
                       style={"fontSize": "22px", "fontWeight": "bold",
                               "color": COLORS["text"], "textAlign": "center",
                               "margin": "8px 0"}),
                html.Pre(id="sop-display", children="—",
                         style={"color": COLORS["text_muted"], "fontSize": "12px",
                                "whiteSpace": "pre-wrap", "marginBottom": 0}),
            ], width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([
                html.H6("銀行後台實時交易風險監控流",
                        style={"color": COLORS["text"], "marginBottom": "8px"}),
                dash_table.DataTable(
                    id="stream-table",
                    columns=[{"name": c, "id": c} for c in
                             ["交易ID", "風險評分", "系統判定狀態", "處置優先級"]],
                    data=EMPTY_TABLE_ROWS,
                    **TABLE_KWARGS,
                ),
            ], width=8),
            dbc.Col([
                html.Pre(id="profit-display", children="—",
                         style={"color": COLORS["text"], "fontSize": "12px",
                                "whiteSpace": "pre-wrap",
                                "backgroundColor": COLORS["bg_card"],
                                "padding": "12px", "borderRadius": "6px",
                                "border": f"1px solid {COLORS['border']}",
                                "height": "100%"}),
            ], width=4),
        ]),
    ])


def page_analysis():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id="importance-chart", figure=FIG_IMPORTANCE,
                              config={"displayModeBar": False})
                ), style={"backgroundColor": COLORS["bg_card"],
                           "border": f"1px solid {COLORS['border']}"}),
            ], width=6),
            dbc.Col([
                dbc.Card(dbc.CardBody(
                    dcc.Graph(id="profit-curve", config={"displayModeBar": False})
                ), style={"backgroundColor": COLORS["bg_card"],
                           "border": f"1px solid {COLORS['border']}"}),
            ], width=6),
        ]),
    ])


def page_report():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Button("產生管理報表", id="report-btn", color="secondary",
                           className="mb-3"),
                html.Pre(id="report-summary", children="（尚未產生報表）",
                         style={"color": COLORS["text"], "fontSize": "12px",
                                "whiteSpace": "pre-wrap",
                                "backgroundColor": COLORS["bg_card"],
                                "padding": "12px", "borderRadius": "6px",
                                "border": f"1px solid {COLORS['border']}",
                                "marginBottom": "12px"}),
                dcc.Download(id="report-download"),
                dbc.Button("下載報表 CSV", id="download-btn",
                           color="success", style={"display": "none"}),
            ], width=8),
        ]),
    ])


# ── Dash app ──────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.SLATE],
    suppress_callback_exceptions=True,
    title="銀行智慧型信用卡防詐系統",
)

_sidebar_style = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "260px", "padding": "16px",
    "backgroundColor": COLORS["bg_sidebar"],
    "borderRight": f"1px solid {COLORS['border']}",
    "overflowY": "auto",
    "zIndex": 100,
}
_content_style = {
    "marginLeft": "276px",
    "padding": "20px",
    "backgroundColor": COLORS["bg_page"],
    "minHeight": "100vh",
}

app.layout = html.Div([
    dcc.Location(id="url"),
    dcc.Store(id="tx-state",       data={"id": None, "score": 0.0}),
    dcc.Store(id="history-store",  data=[]),
    dcc.Store(id="settings-store", data={"threshold": 0.50, "cost_fn": 250, "cost_fp": 15}),
    dcc.Store(id="report-store"),
    dcc.Interval(id="init-interval", interval=100, max_intervals=1),

    # ── 側邊欄 ──────────────────────────────────────
    html.Div([
        html.H6("信用卡防詐系統",
                style={"color": COLORS["text"], "fontSize": "14px", "marginBottom": "2px"}),
        html.P("工資系 商管程式設計 期末專題",
               style={"color": COLORS["text_muted"], "fontSize": "11px", "marginBottom": "12px"}),
        html.Hr(style={"borderColor": COLORS["border"]}),

        dbc.Nav([
            dbc.NavLink("監控", href="/",         active="exact",
                        style={"color": COLORS["text"], "borderRadius": "6px",
                               "marginBottom": "4px", "fontSize": "13px"}),
            dbc.NavLink("分析", href="/analysis",  active="exact",
                        style={"color": COLORS["text"], "borderRadius": "6px",
                               "marginBottom": "4px", "fontSize": "13px"}),
            dbc.NavLink("報表", href="/report",    active="exact",
                        style={"color": COLORS["text"], "borderRadius": "6px",
                               "marginBottom": "4px", "fontSize": "13px"}),
        ], vertical=True, pills=True),

        html.Hr(style={"borderColor": COLORS["border"]}),

        html.Label("銀行營運方針", style={"color": COLORS["text"], "fontSize": "12px"}),
        dcc.Dropdown(
            id="scenario-dropdown",
            options=[
                {"label": "損益平衡最優化", "value": "balanced"},
                {"label": "嚴格資安防禦",   "value": "strict"},
                {"label": "客戶體驗優先",   "value": "ux"},
            ],
            value="balanced", clearable=False,
            style={"fontSize": "12px", "marginBottom": "12px"},
        ),
        html.Label("風險判定閥值", style={"color": COLORS["text"], "fontSize": "12px"}),
        dcc.Slider(id="threshold-slider", min=0.01, max=0.99, step=0.01, value=0.50,
                   tooltip={"placement": "bottom", "always_visible": True},
                   marks={0.01: "0", 0.5: "0.5", 0.99: "1"},
                   className="mb-3"),
        html.Label("漏抓成本 FN Cost (USD/筆)", style={"color": COLORS["text"], "fontSize": "12px"}),
        dcc.Slider(id="cost-fn-slider", min=50, max=1000, step=10, value=250,
                   tooltip={"placement": "bottom", "always_visible": True},
                   marks={50: "50", 500: "500", 1000: "1000"},
                   className="mb-3"),
        html.Label("誤判成本 FP Cost (USD/筆)", style={"color": COLORS["text"], "fontSize": "12px"}),
        dcc.Slider(id="cost-fp-slider", min=1, max=100, step=1, value=15,
                   tooltip={"placement": "bottom", "always_visible": True},
                   marks={1: "1", 50: "50", 100: "100"},
                   className="mb-3"),
        dbc.Button("接收即時交易", id="simulate-btn", color="primary",
                   size="sm", className="w-100 mt-2"),
    ], style=_sidebar_style),

    # ── 主內容區 ──────────────────────────────────────
    html.Div([
        # KPI 卡（所有頁面皆顯示）
        dbc.Row([
            dbc.Col(make_kpi_card("今日攔截筆數",    "kpi-intercepted", COLORS["fraud"]),   width=3),
            dbc.Col(make_kpi_card("防護金額 (USD)",   "kpi-saved",       COLORS["safe"]),    width=3),
            dbc.Col(make_kpi_card("模型 AUC",          "kpi-auc",         COLORS["cyan"]),    width=3),
            dbc.Col(make_kpi_card("當前閥值",           "kpi-threshold",   COLORS["warning"]), width=3),
        ], className="mb-3 g-2"),

        # 三個頁面（全在 DOM，CSS 切換顯示）
        html.Div(id="page-monitor",  children=page_monitor()),
        html.Div(id="page-analysis", children=page_analysis(), style={"display": "none"}),
        html.Div(id="page-report",   children=page_report(),   style={"display": "none"}),
    ], style=_content_style),
], style={"backgroundColor": COLORS["bg_page"]})

# ── Callbacks ─────────────────────────────────────────────

@app.callback(
    Output("page-monitor",  "style"),
    Output("page-analysis", "style"),
    Output("page-report",   "style"),
    Input("url", "pathname"),
)
def route_page(pathname):
    show = {"display": "block"}
    hide = {"display": "none"}
    if pathname == "/analysis":
        return hide, show, hide
    if pathname == "/report":
        return hide, hide, show
    return show, hide, hide


@app.callback(
    Output("threshold-slider", "value"),
    Input("scenario-dropdown", "value"),
)
def handle_scenario(scenario):
    return {"strict": 0.20, "ux": 0.80, "balanced": 0.50}.get(scenario, 0.50)


@app.callback(
    Output("kpi-intercepted",  "children"),
    Output("kpi-saved",        "children"),
    Output("kpi-auc",          "children"),
    Output("kpi-threshold",    "children"),
    Output("status-display",   "children"),
    Output("sop-display",      "children"),
    Output("profit-display",   "children"),
    Output("stream-table",     "data"),
    Output("profit-curve",     "figure"),
    Output("tx-state",         "data"),
    Output("history-store",    "data"),
    Output("settings-store",   "data"),
    Input("init-interval",     "n_intervals"),
    Input("simulate-btn",      "n_clicks"),
    Input("threshold-slider",  "value"),
    Input("cost-fn-slider",    "value"),
    Input("cost-fp-slider",    "value"),
    State("tx-state",          "data"),
    State("history-store",     "data"),
    prevent_initial_call=False,
)
def update_dashboard(n_intervals, n_clicks, threshold, cost_fn, cost_fp,
                     tx_state, history_data):
    triggered = ctx.triggered_id
    is_new_tx = triggered in ("simulate-btn", "init-interval") or tx_state["id"] is None

    # ── 抽樣一筆交易 ─────────────────────────────────
    if is_new_tx:
        if np.random.rand() < 0.30:
            idx = int(np.random.choice(np.where(y_all == 1)[0]))
        else:
            idx = int(np.random.choice(np.where(y_all == 0)[0]))
        risk  = float(model_pipeline.predict_proba(X_test.iloc[[idx]])[0][1])
        tx_id = f"#TX-{np.random.randint(10000, 99999)}"
        tx_state = {"id": tx_id, "score": risk}
    else:
        tx_id = tx_state["id"]
        risk  = tx_state["score"]

    # ── 判定結果與 SOP ───────────────────────────────
    is_fraud   = risk >= threshold
    status_txt = "[!] 詐騙警示 (FRAUD)" if is_fraud else "[OK] 安全交易 (SAFE)"
    if is_fraud:
        sop = (
            "[風控系統核心響應機制觸發]:\n"
            "1. 該筆交易已被即時攔截，暫停授權發放。\n"
            "2. 系統已自動透過 LINE/簡訊 發送消費確認通知給持卡人。\n"
            "3. 風控中心已自動將此案派發工單至『一線客服部』進行人工外撥核對。"
        )
    else:
        sop = (
            "[標準商務流程放行]:\n"
            "1. 交易通過 XGBoost 演算法風控比對，毫秒級授權成功。\n"
            "2. 計入當日持卡人正常消費信用額度，保障刷卡順暢體驗。"
        )

    priority = (
        "Level 1 (緊急阻斷)" if risk > 0.90 else
        "Level 2 (人工確認)" if is_fraud else
        "Level 3 (自動化驗證)"
    )

    # ── 更新交易歷史 ─────────────────────────────────
    if is_new_tx:
        history_data = [[tx_id, round(risk, 4), status_txt, priority]] + history_data
        history_data = history_data[:5]
    elif history_data:
        history_data[0][2] = status_txt
        history_data[0][3] = priority

    table_rows = history_data + [["-", "-", "-", "-"]] * (5 - len(history_data))
    df_table = pd.DataFrame(table_rows,
                             columns=["交易ID", "風險評分", "系統判定狀態", "處置優先級"])

    # ── 利潤計算 ─────────────────────────────────────
    y_pred = (y_proba_all >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(len(y_test) - sum(y_pred))
        fp = fn = 0
        tp = int(sum(y_pred))

    prevented = int(tp) * cost_fn
    friction  = int(fp) * cost_fp + int(fn) * cost_fn
    net       = prevented - friction

    profit_txt = (
        f"全局財務效益試算:\n"
        f"- 成功防堵呆帳: +${prevented:,} USD (攔截 {tp} 筆)\n"
        f"- 營收摩擦與漏抓成本: -${friction:,} USD\n"
        f"銀行風控純防護收益淨值: ${net:,} USD"
    )

    # ── 利潤曲線 ─────────────────────────────────────
    df_curve = CURVE_STATS.copy()
    df_curve["Profit"] = (df_curve["TP"] * cost_fn
                          - (df_curve["FP"] * cost_fp + df_curve["FN"] * cost_fn))
    fig_curve = px.line(
        df_curve, x="Threshold", y="Profit",
        title="動態利潤最佳化曲線圖",
        labels={"Threshold": "判定閥值", "Profit": "預期淨利潤 (USD)"},
        color_discrete_sequence=[COLORS["accent"]],
    )
    fig_curve.add_scatter(
        x=[threshold], y=[net], mode="markers",
        marker={"size": 10, "color": COLORS["warning"], "symbol": "diamond"},
        name="當前決策點",
    )
    min_p, max_p = df_curve["Profit"].min(), df_curve["Profit"].max()
    p_range = max(float(max_p - min_p), 1.0)
    fig_curve.update_layout(
        xaxis={"range": [0, 1], "gridcolor": COLORS["border"], "gridwidth": 1},
        yaxis={"range": [min_p - p_range * 0.05, max_p + p_range * 0.05],
               "gridcolor": COLORS["border"], "gridwidth": 1},
        paper_bgcolor=COLORS["bg_card"], plot_bgcolor=COLORS["bg_card"],
        font={"color": COLORS["text"]},
        margin={"l": 10, "r": 10, "b": 30, "t": 40},
        showlegend=False,
    )

    settings = {"threshold": threshold, "cost_fn": cost_fn, "cost_fp": cost_fp}

    return (
        str(tp), f"${prevented:,}", str(AUC_SCORE), str(round(threshold, 2)),
        status_txt, sop, profit_txt, df_table.to_dict("records"),
        fig_curve, tx_state, history_data, settings,
    )


@app.callback(
    Output("report-summary", "children"),
    Output("report-store",   "data"),
    Output("download-btn",   "style"),
    Input("report-btn",      "n_clicks"),
    State("settings-store",  "data"),
    prevent_initial_call=True,
)
def generate_report(n_clicks, settings):
    threshold = settings["threshold"]
    cost_fn   = settings["cost_fn"]
    cost_fp   = settings["cost_fp"]

    recent_y     = y_test.iloc[recent_tx_indices]
    recent_proba = y_proba_all[recent_tx_indices]
    y_pred       = (recent_proba >= threshold).astype(int)
    cm = confusion_matrix(recent_y, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = int(len(recent_y) - sum(y_pred))
        fp = fn = 0
        tp = int(sum(y_pred))

    saved_loss  = int(tp) * cost_fn
    profit_arr  = (CURVE_STATS["TP"] * cost_fn
                   - (CURVE_STATS["FP"] * cost_fp + CURVE_STATS["FN"] * cost_fn))
    best_thresh = float(CURVE_STATS["Threshold"].iloc[np.argmax(profit_arr)])

    summary = (
        f"=== 今日風控摘要 ===\n\n"
        f"總交易量：{len(recent_y)} 筆\n"
        f"攔截詐騙：{tp} 筆\n"
        f"誤攔交易：{fp} 筆\n"
        f"成功阻止損失：${saved_loss:,}\n"
        f"最佳 Threshold：{best_thresh:.2f}\n"
        f"模型 AUC：{AUC_SCORE}"
    )

    report_df = pd.DataFrame({
        "Transaction_ID": [f"TX-{np.random.randint(100000, 999999)}"
                           for _ in range(len(recent_y))],
        "Risk_Score":     np.round(recent_proba, 4),
        "Is_Fraud_Real":  recent_y.values,
        "System_Blocked": y_pred,
    })
    report_df["Status"] = report_df.apply(
        lambda x: (
            "True Positive (成功攔截)"  if x["Is_Fraud_Real"] == 1 and x["System_Blocked"] == 1 else
            "False Positive (客戶誤擋)" if x["Is_Fraud_Real"] == 0 and x["System_Blocked"] == 1 else
            "False Negative (漏抓損失)" if x["Is_Fraud_Real"] == 1 and x["System_Blocked"] == 0 else
            "True Negative (安全放行)"
        ), axis=1,
    )

    csv_content = report_df.to_csv(index=False, encoding="utf-8-sig")
    store = {"filename": "daily_report.csv", "content": csv_content}
    return summary, store, {"display": "inline-block"}


@app.callback(
    Output("report-download", "data"),
    Input("download-btn",     "n_clicks"),
    State("report-store",     "data"),
    prevent_initial_call=True,
)
def download_report(n_clicks, store_data):
    if not store_data:
        return no_update
    return dcc.send_string(store_data["content"], store_data["filename"])


if __name__ == "__main__":
    print("\n" + "="*50)
    print("請開啟瀏覽器前往：http://127.0.0.1:8050")
    print("="*50 + "\n")
    app.run(debug=False)
