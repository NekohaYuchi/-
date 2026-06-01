"""
下載預訓練模型與測試資料集
執行方式：python setup.py
"""
import urllib.request
import os
import sys

ASSETS = {
    "fraud_model_pipeline.pkl": "https://github.com/NekohaYuchi/-/releases/download/v1.1.0/fraud_model_pipeline.pkl",
    "test_data.pkl":            "https://github.com/NekohaYuchi/-/releases/download/v1.1.0/test_data.pkl",
    "scaler.pkl":               "https://github.com/NekohaYuchi/-/releases/download/v1.1.0/scaler.pkl",
}

def download(filename, url):
    if os.path.exists(filename):
        print(f"  已存在，略過：{filename}")
        return

    print(f"  下載中：{filename} ...")
    try:
        def progress(count, block_size, total_size):
            pct = min(int(count * block_size * 100 / total_size), 100)
            sys.stdout.write(f"\r  進度：{pct}%  ")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, filename, reporthook=progress)
        print(f"\r  完成：{filename}            ")
    except Exception as e:
        print(f"\r  失敗：{e}")
        sys.exit(1)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print("=== 下載預訓練模型 ===")
    for name, url in ASSETS.items():
        download(name, url)
    print("\n完成！現在可以執行：")
    print("  python system_dash.py   # Dash 版（推薦）")
    print("  python system.py        # Gradio 版")
