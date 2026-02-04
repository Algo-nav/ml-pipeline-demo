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

    # --- app.py ---
    L = []
    L.append("import gradio as gr")
    L.append("import pickle")
    L.append("import numpy as np")
    L.append("import json")
    L.append("")
    L.append("with open('model.pkl', 'rb') as f:")
    L.append("    model = pickle.load(f)")
    L.append("")
    L.append("with open('metrics.json', 'r') as f:")
    L.append("    metrics = json.load(f)")
    L.append("")
    L.append("with open('retrain_report.json', 'r') as f:")
    L.append("    retrain_report = json.load(f)")
    L.append("")
    L.append("")
    L.append("def predict(amount, hour, day_of_week, distance_from_home,")
    L.append("            distance_from_last_txn, ratio_to_median, num_txn_last_24h,")
    L.append("            is_foreign, merchant_risk_score, card_age_months):")
    L.append("    features = np.array([[amount, hour, day_of_week, distance_from_home,")
    L.append("                          distance_from_last_txn, ratio_to_median,")
    L.append("                          num_txn_last_24h, int(is_foreign),")
    L.append("                          merchant_risk_score, card_age_months]])")
    L.append("    prob = model.predict_proba(features)[0]")
    L.append("    return {'Legitimate': float(prob[0]), 'Fraud': float(prob[1])}")
    L.append("")
    L.append("")
    L.append("def get_model_info():")
    L.append("    m = metrics.get('metrics', {})")
    L.append("    r = retrain_report")
    L.append("    lines = []")
    L.append("    lines.append('Last updated: ' + metrics.get('timestamp', 'N/A'))")
    L.append("    lines.append('F1 Score: ' + str(round(m.get('f1', 0), 4)))")
    L.append("    lines.append('ROC AUC: ' + str(round(m.get('roc_auc', 0), 4)))")
    L.append("    lines.append('Precision: ' + str(round(m.get('precision', 0), 4)))")
    L.append("    lines.append('Recall: ' + str(round(m.get('recall', 0), 4)))")
    L.append("    lines.append('')")
    L.append("    lines.append('Retrain Decision: ' + str(r.get('decision', 'N/A')))")
    L.append("    lines.append('Reason: ' + str(r.get('reason', 'N/A')))")
    L.append("    drift = r.get('drift', {})")
    L.append("    lines.append('Drift Detected: ' + str(drift.get('dataset_drift', 'N/A')))")
    L.append("    lines.append('Drifted Features: ' + str(drift.get('drifted_features', [])))")
    L.append("    return chr(10).join(lines)")
    L.append("")
    L.append("")
    L.append("with gr.Blocks() as demo:")
    L.append("    gr.Markdown('# Fraud Detection System (Auto-Retrained)')")
    L.append("    gr.Markdown('This model is automatically retrained when data drift is detected.')")
    L.append("")
    L.append("    with gr.Tab('Predict'):")
    L.append("        with gr.Row():")
    L.append("            with gr.Column():")
    L.append("                amount = gr.Number(value=50.0, label='Transaction Amount ($)')")
    L.append("                hour = gr.Slider(0, 23, value=14, step=1, label='Hour of Day')")
    L.append("                day = gr.Slider(0, 6, value=3, step=1, label='Day of Week (0=Mon)')")
    L.append("                dist_home = gr.Number(value=10.0, label='Distance from Home (km)')")
    L.append("                dist_last = gr.Number(value=5.0, label='Distance from Last Txn (km)')")
    L.append("            with gr.Column():")
    L.append("                ratio = gr.Number(value=1.0, label='Ratio to Median Spending')")
    L.append("                n_txn = gr.Slider(0, 20, value=3, step=1, label='Txns in Last 24h')")
    L.append("                foreign = gr.Checkbox(value=False, label='Foreign Transaction')")
    L.append("                merchant = gr.Slider(0, 1, value=0.2, step=0.05, label='Merchant Risk Score')")
    L.append("                card_age = gr.Slider(1, 120, value=36, step=1, label='Card Age (months)')")
    L.append("")
    L.append("        predict_btn = gr.Button('Analyze Transaction', variant='primary')")
    L.append("        output = gr.Label(num_top_classes=2, label='Result')")
    L.append("")
    L.append("        predict_btn.click(")
    L.append("            fn=predict,")
    L.append("            inputs=[amount, hour, day, dist_home, dist_last,")
    L.append("                    ratio, n_txn, foreign, merchant, card_age],")
    L.append("            outputs=output")
    L.append("        )")
    L.append("")
    L.append("        gr.Examples(")
    L.append("            examples=[")
    L.append("                [25.0, 14, 2, 5.0, 3.0, 0.8, 2, False, 0.1, 48],")
    L.append("                [500.0, 3, 5, 100.0, 80.0, 5.0, 10, True, 0.7, 6],")
    L.append("                [1200.0, 2, 6, 200.0, 150.0, 8.0, 15, True, 0.9, 3],")
    L.append("            ],")
    L.append("            inputs=[amount, hour, day, dist_home, dist_last,")
    L.append("                    ratio, n_txn, foreign, merchant, card_age],")
    L.append("        )")
    L.append("")
    L.append("    with gr.Tab('Model Info'):")
    L.append("        info_btn = gr.Button('Show Model Info')")
    L.append("        info_output = gr.Textbox(label='Model Details', lines=12)")
    L.append("        info_btn.click(fn=get_model_info, outputs=info_output)")
    L.append("")
    L.append("demo.launch()")

    NL = chr(10)
    with open("fraud_space_deploy/app.py", "w") as f:
        f.write(NL.join(L) + NL)

    # --- requirements.txt ---
    with open("fraud_space_deploy/requirements.txt", "w") as f:
        f.write(NL.join(["gradio", "numpy", "scikit-learn"]) + NL)

    # --- README.md ---
    R = []
    R.append("---")
    R.append("title: Fraud Detector CI/CD")
    R.append("emoji: \U0001f6e1")
    R.append("colorFrom: red")
    R.append("colorTo: red")
    R.append("sdk: gradio")
    R.append("sdk_version: 5.29.0")
    R.append("app_file: app.py")
    R.append("pinned: false")
    R.append("license: mit")
    R.append("---")
    R.append("")
    R.append("# Fraud Detection System (Auto-Retrained)")
    R.append("")
    R.append("Credit card fraud detector with automated retraining pipeline.")
    R.append("")
    R.append("## Pipeline")
    R.append("")
    R.append("1. New data batch generated (simulating production)")
    R.append("2. Drift detection (Wasserstein distance)")
    R.append("3. Challenger model trained on combined data")
    R.append("4. Champion vs Challenger comparison")
    R.append("5. Deploy only if challenger wins")
    R.append("")
    R.append("## Links")
    R.append("")
    R.append("- [GitHub Repo](https://github.com/Algo-nav/ml-pipeline-demo)")
    R.append("- [Author](https://huggingface.co/Nav772)")

    with open("fraud_space_deploy/README.md", "w") as f:
        f.write(NL.join(R) + NL)

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
