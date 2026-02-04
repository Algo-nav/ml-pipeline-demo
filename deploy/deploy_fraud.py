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

    # Write app.py
    app_code = (
        "import gradio as gr\n"
        "import pickle\n"
        "import numpy as np\n"
        "import json\n"
        "\n"
        "with open('model.pkl', 'rb') as f:\n"
        "    model = pickle.load(f)\n"
        "\n"
        "with open('metrics.json', 'r') as f:\n"
        "    metrics = json.load(f)\n"
        "\n"
        "with open('retrain_report.json', 'r') as f:\n"
        "    retrain_report = json.load(f)\n"
        "\n"
        "\n"
        "def predict(amount, hour, day_of_week, distance_from_home,\n"
        "            distance_from_last_txn, ratio_to_median, num_txn_last_24h,\n"
        "            is_foreign, merchant_risk_score, card_age_months):\n"
        "    features = np.array([[amount, hour, day_of_week, distance_from_home,\n"
        "                          distance_from_last_txn, ratio_to_median,\n"
        "                          num_txn_last_24h, int(is_foreign),\n"
        "                          merchant_risk_score, card_age_months]])\n"
        "    prob = model.predict_proba(features)[0]\n"
        "    return {'Legitimate': float(prob[0]), 'Fraud': float(prob[1])}\n"
        "\n"
        "\n"
        "def get_model_info():\n"
        "    m = metrics.get('metrics', {})\n"
        "    r = retrain_report\n"
        "    lines = []\n"
        "    lines.append('Last updated: ' + metrics.get('timestamp', 'N/A'))\n"
        "    lines.append('F1 Score: ' + str(round(m.get('f1', 0), 4)))\n"
        "    lines.append('ROC AUC: ' + str(round(m.get('roc_auc', 0), 4)))\n"
        "    lines.append('Precision: ' + str(round(m.get('precision', 0), 4)))\n"
        "    lines.append('Recall: ' + str(round(m.get('recall', 0), 4)))\n"
        "    lines.append('')\n"
        "    lines.append('Retrain Decision: ' + str(r.get('decision', 'N/A')))\n"
        "    lines.append('Reason: ' + str(r.get('reason', 'N/A')))\n"
        "    drift = r.get('drift', {})\n"
        "    lines.append('Drift Detected: ' + str(drift.get('dataset_drift', 'N/A')))\n"
        "    lines.append('Drifted Features: ' + str(drift.get('drifted_features', [])))\n"
        "    return chr(10).join(lines)\n"
        "\n"
        "\n"
        "with gr.Blocks() as demo:\n"
        "    gr.Markdown('# Fraud Detection System (Auto-Retrained)')\n"
        "    gr.Markdown('This model is automatically retrained when data drift is detected.')\n"
        "\n"
        "    with gr.Tab('Predict'):\n"
        "        with gr.Row():\n"
        "            with gr.Column():\n"
        "                amount = gr.Number(value=50.0, label='Transaction Amount ($)')\n"
        "                hour = gr.Slider(0, 23, value=14, step=1, label='Hour of Day')\n"
        "                day = gr.Slider(0, 6, value=3, step=1, label='Day of Week (0=Mon)')\n"
        "                dist_home = gr.Number(value=10.0, label='Distance from Home (km)')\n"
        "                dist_last = gr.Number(value=5.0, label='Distance from Last Txn (km)')\n"
        "            with gr.Column():\n"
        "                ratio = gr.Number(value=1.0, label='Ratio to Median Spending')\n"
        "                n_txn = gr.Slider(0, 20, value=3, step=1, label='Txns in Last 24h')\n"
        "                foreign = gr.Checkbox(value=False, label='Foreign Transaction')\n"
        "                merchant = gr.Slider(0, 1, value=0.2, step=0.05, label='Merchant Risk Score')\n"
        "                card_age = gr.Slider(1, 120, value=36, step=1, label='Card Age (months)')\n"
        "\n"
        "        predict_btn = gr.Button('Analyze Transaction', variant='primary')\n"
        "        output = gr.Label(num_top_classes=2, label='Result')\n"
        "\n"
        "        predict_btn.click(\n"
        "            fn=predict,\n"
        "            inputs=[amount, hour, day, dist_home, dist_last,\n"
        "                    ratio, n_txn, foreign, merchant, card_age],\n"
        "            outputs=output\n"
        "        )\n"
        "\n"
        "        gr.Examples(\n"
        "            examples=[\n"
        "                [25.0, 14, 2, 5.0, 3.0, 0.8, 2, False, 0.1, 48],\n"
        "                [500.0, 3, 5, 100.0, 80.0, 5.0, 10, True, 0.7, 6],\n"
        "                [1200.0, 2, 6, 200.0, 150.0, 8.0, 15, True, 0.9, 3],\n"
        "            ],\n"
        "            inputs=[amount, hour, day, dist_home, dist_last,\n"
        "                    ratio, n_txn, foreign, merchant, card_age],\n"
        "        )\n"
        "\n"
        "    with gr.Tab('Model Info'):\n"
        "        info_btn = gr.Button('Show Model Info')\n"
        "        info_output = gr.Textbox(label='Model Details', lines=12)\n"
        "        info_btn.click(fn=get_model_info, outputs=info_output)\n"
        "\n"
        "demo.launch()\n"
    )

    with open("fraud_space_deploy/app.py", "w") as f:
        f.write(app_code)

    with open("fraud_space_deploy/requirements.txt", "w") as f:
        f.write("gradio\nnumpy\nscikit-learn\n")

    readme = (
        "---\n"
        "title: Fraud Detection (Auto-Retrained)\n"
        "emoji: \xf0\x9f\x94\x8d\n"
        "colorFrom: red\n"
        "colorTo: orange\n"
        "sdk: gradio\n"
        "sdk_version: 5.29.0\n"
        "app_file: app.py\n"
        "pinned: false\n"
        "license: mit\n"
        "---\n"
        "\n"
        "# Fraud Detection System (Auto-Retrained)\n"
        "\n"
        "Credit card fraud detector with automated retraining pipeline.\n"
        "\n"
        "## Pipeline\n"
        "\n"
        "1. New data batch generated (simulating production)\n"
        "2. Drift detection (Wasserstein distance)\n"
        "3. Challenger model trained on combined data\n"
        "4. Champion vs Challenger comparison\n"
        "5. Deploy only if challenger wins\n"
        "\n"
        "## Links\n"
        "\n"
        "- [GitHub Repo](https://github.com/Algo-nav/ml-pipeline-demo)\n"
        "- [Author](https://huggingface.co/Nav772)\n"
    )

    with open("fraud_space_deploy/README.md", "w") as f:
        f.write(readme)

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
        print("Space ready: " + REPO_ID)
    except Exception as e:
        print("Repo note: " + str(e))

    api.upload_folder(
        folder_path="fraud_space_deploy",
        repo_id=REPO_ID,
        repo_type="space",
        token=token
    )

    print("Deployed to: https://huggingface.co/spaces/" + REPO_ID)


if __name__ == "__main__":
    create_space_files()
    deploy()
