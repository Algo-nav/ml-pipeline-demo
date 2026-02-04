"""
Deploy Fraud Detector to HuggingFace Spaces
"""

import os
import shutil
import json
from huggingface_hub import HfApi, create_repo, upload_folder

SPACE_NAME = "fraud-detector-cicd"
USERNAME = "Nav772"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"


def create_space_files():
    os.makedirs("fraud_space_deploy", exist_ok=True)

    shutil.copy("fraud_artifacts/model.pkl", "fraud_space_deploy/model.pkl")
    shutil.copy("fraud_artifacts/metrics.json", "fraud_space_deploy/metrics.json")
    shutil.copy("fraud_artifacts/retrain_report.json", "fraud_space_deploy/retrain_report.json")

    with open("fraud_artifacts/metrics.json", "r") as f:
        metrics = json.load(f)
    m = metrics.get("metrics", {})

    with open("fraud_artifacts/retrain_report.json", "r") as f:
        retrain = json.load(f)

    app_lines = [
        "import gradio as gr",
        "import pickle",
        "import numpy as np",
        "import json",
        "",
        'with open("model.pkl", "rb") as f:',
        "    model = pickle.load(f)",
        "",
        'with open("metrics.json", "r") as f:',
        "    metrics = json.load(f)",
        "",
        'with open("retrain_report.json", "r") as f:',
        "    retrain_report = json.load(f)",
        "",
        "",
        "def predict(amount, hour, day_of_week, distance_from_home,",
        "            distance_from_last_txn, ratio_to_median, num_txn_last_24h,",
        "            is_foreign, merchant_risk_score, card_age_months):",
        "    features = np.array([[amount, hour, day_of_week, distance_from_home,",
        "                          distance_from_last_txn, ratio_to_median,",
        "                          num_txn_last_24h, int(is_foreign),",
        "                          merchant_risk_score, card_age_months]])",
        "    prob = model.predict_proba(features)[0]",
        '    return {"Legitimate": float(prob[0]), "Fraud": float(prob[1])}',
        "",
        "",
        "def get_model_info():",
        '    m = metrics.get("metrics", {})',
        '    r = retrain_report',
        '    info = f"Last updated: {metrics.get('timestamp', 'N/A')}\n"',
        '    info += f"F1 Score: {m.get('f1', 0):.4f}\n"',
        '    info += f"ROC AUC: {m.get('roc_auc', 0):.4f}\n"',
        '    info += f"Precision: {m.get('precision', 0):.4f}\n"',
        '    info += f"Recall: {m.get('recall', 0):.4f}\n"',
        '    info += f"\nRetrain Decision: {r.get('decision', 'N/A')}\n"',
        '    info += f"Reason: {r.get('reason', 'N/A')}\n"',
        '    drift = r.get("drift", {})',
        '    info += f"Drift Detected: {drift.get('dataset_drift', 'N/A')}\n"',
        '    info += f"Drifted Features: {drift.get('drifted_features', [])}\n"',
        "    return info",
        "",
        "",
        "with gr.Blocks() as demo:",
        '    gr.Markdown("# Fraud Detection System (Auto-Retrained)")',
        '    gr.Markdown("This model is automatically retrained when data drift is detected.")',
        "",
        "    with gr.Tab('Predict'):",
        "        with gr.Row():",
        "            with gr.Column():",
        '                amount = gr.Number(value=50.0, label="Transaction Amount ($)")',
        '                hour = gr.Slider(0, 23, value=14, step=1, label="Hour of Day")',
        '                day = gr.Slider(0, 6, value=3, step=1, label="Day of Week (0=Mon)")',
        '                dist_home = gr.Number(value=10.0, label="Distance from Home (km)")',
        '                dist_last = gr.Number(value=5.0, label="Distance from Last Transaction (km)")',
        "            with gr.Column():",
        '                ratio = gr.Number(value=1.0, label="Ratio to Median Spending")',
        '                n_txn = gr.Slider(0, 20, value=3, step=1, label="Transactions in Last 24h")',
        '                foreign = gr.Checkbox(value=False, label="Foreign Transaction")',
        '                merchant = gr.Slider(0, 1, value=0.2, step=0.05, label="Merchant Risk Score")',
        '                card_age = gr.Slider(1, 120, value=36, step=1, label="Card Age (months)")',
        "",
        '        predict_btn = gr.Button("Analyze Transaction", variant="primary")',
        '        output = gr.Label(num_top_classes=2, label="Result")',
        "",
        "        predict_btn.click(",
        "            fn=predict,",
        "            inputs=[amount, hour, day, dist_home, dist_last,",
        "                    ratio, n_txn, foreign, merchant, card_age],",
        "            outputs=output",
        "        )",
        "",
        '        gr.Examples(',
        '            examples=[',
        '                [25.0, 14, 2, 5.0, 3.0, 0.8, 2, False, 0.1, 48],',
        '                [500.0, 3, 5, 100.0, 80.0, 5.0, 10, True, 0.7, 6],',
        '                [1200.0, 2, 6, 200.0, 150.0, 8.0, 15, True, 0.9, 3],',
        '            ],',
        "            inputs=[amount, hour, day, dist_home, dist_last,",
        "                    ratio, n_txn, foreign, merchant, card_age],",
        "        )",
        "",
        "    with gr.Tab('Model Info'):",
        '        info_btn = gr.Button("Show Model Info")',
        '        info_output = gr.Textbox(label="Model Details", lines=12)',
        "        info_btn.click(fn=get_model_info, outputs=info_output)",
        "",
        'demo.launch()',
    ]

    with open("fraud_space_deploy/app.py", "w") as f:
        f.write("\n".join(app_lines))

    with open("fraud_space_deploy/requirements.txt", "w") as f:
        f.write("gradio\nnumpy\nscikit-learn\n")

    readme_lines = [
        "---",
        "title: Fraud Detection (Auto-Retrained)",
        "emoji: 🔍",
        "colorFrom: red",
        "colorTo: orange",
        "sdk: gradio",
        "sdk_version: 5.29.0",
        "app_file: app.py",
        "pinned: false",
        "license: mit",
        "---",
        "",
        "# Fraud Detection System (Auto-Retrained)",
        "",
        "Credit card fraud detector with automated retraining pipeline.",
        "",
        "## Pipeline",
        "",
        "1. New data batch generated (simulating production)",
        "2. Drift detection (Wasserstein distance)",
        "3. Challenger model trained on combined data",
        "4. Champion vs Challenger comparison",
        "5. Deploy only if challenger wins",
        "",
        "## Links",
        "",
        "- [GitHub Repo](https://github.com/Algo-nav/ml-pipeline-demo)",
        "- [Author](https://huggingface.co/Nav772)",
    ]

    with open("fraud_space_deploy/README.md", "w") as f:
        f.write("\n".join(readme_lines))

    print("Space files created")


def deploy():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN not set")

    api = HfApi()

    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print(f"Space ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo note: {e}")

    api.upload_folder(
        folder_path="fraud_space_deploy",
        repo_id=REPO_ID,
        repo_type="space",
        token=token
    )

    print(f"Deployed to: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    create_space_files()
    deploy()
