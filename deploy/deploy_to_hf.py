"""
Deploy to HuggingFace Spaces
"""

import os
import shutil
import json
from huggingface_hub import HfApi, create_repo, upload_folder

SPACE_NAME = "iris-classifier-cicd"
USERNAME = "Nav772"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"


def create_space_files():
    """Create the Gradio app files for the Space."""
    
    os.makedirs("space_deploy", exist_ok=True)
    
    # Copy model artifacts
    shutil.copy("artifacts/model.pkl", "space_deploy/model.pkl")
    shutil.copy("artifacts/metrics.json", "space_deploy/metrics.json")
    
    # Read metrics for display
    with open("artifacts/metrics.json", "r") as f:
        metrics = json.load(f)
    accuracy_pct = f"{metrics['accuracy']:.1%}"
    
    # Create app.py
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
        'CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]',
        "",
        "",
        "def predict(sepal_length, sepal_width, petal_length, petal_width):",
        "    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])",
        "    probabilities = model.predict_proba(features)[0]",
        "    result = {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}",
        "    return result",
        "",
        "",
        "demo = gr.Interface(",
        "    fn=predict,",
        "    inputs=[",
        '        gr.Slider(4.0, 8.0, value=5.8, label="Sepal Length (cm)"),',
        '        gr.Slider(2.0, 4.5, value=3.0, label="Sepal Width (cm)"),',
        '        gr.Slider(1.0, 7.0, value=4.0, label="Petal Length (cm)"),',
        '        gr.Slider(0.1, 2.5, value=1.2, label="Petal Width (cm)"),',
        "    ],",
        '    outputs=gr.Label(num_top_classes=3, label="Prediction"),',
        '    title="Iris Classifier (CI/CD Demo)",',
        "    description=(",
        '        "Classify Iris flowers based on sepal and petal measurements. "',
        '        "Model Performance: ' + accuracy_pct + ' accuracy. "',
        '        "This model was automatically trained and deployed via GitHub Actions CI/CD."',
        "    ),",
        "    examples=[",
        "        [5.1, 3.5, 1.4, 0.2],",
        "        [6.2, 2.9, 4.3, 1.3],",
        "        [7.7, 3.0, 6.1, 2.3],",
        "    ]",
        ")",
        "",
        'if __name__ == "__main__":',
        "    demo.launch()",
    ]
    
    with open("space_deploy/app.py", "w") as f:
        f.write("\n".join(app_lines))
    
    # Create requirements.txt
    with open("space_deploy/requirements.txt", "w") as f:
        f.write("gradio\nnumpy\nscikit-learn\n")
    
    # Create README.md
    readme_lines = [
        "---",
        "title: Iris Classifier (CI/CD Demo)",
        "emoji: 🌸",
        "colorFrom: pink",
        "colorTo: purple",
        "sdk: gradio",
        "sdk_version: 5.29.0",
        "app_file: app.py",
        "pinned: false",
        "license: mit",
        "---",
        "",
        "# Iris Classifier (CI/CD Demo)",
        "",
        "A simple Iris flower classifier demonstrating automated ML deployment via GitHub Actions.",
        "",
        "## CI/CD Pipeline",
        "",
        "This model is automatically:",
        "1. **Validated** - Data quality checks",
        "2. **Tested** - Unit tests for training code",
        "3. **Trained** - Model training with accuracy threshold",
        "4. **Deployed** - Pushed to this HuggingFace Space",
        "",
        "## Model Details",
        "",
        "- **Algorithm:** Random Forest (100 trees)",
        "- **Dataset:** Iris (150 samples, 3 classes)",
        f"- **Accuracy:** {accuracy_pct}",
        "",
        "## Links",
        "",
        "- [GitHub Repository](https://github.com/Algo-nav/ml-pipeline-demo)",
        "- [Author: Nav772](https://huggingface.co/Nav772)",
    ]
    
    with open("space_deploy/README.md", "w") as f:
        f.write("\n".join(readme_lines))
    
    print("Space files created")


def deploy():
    """Deploy to HuggingFace Space."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
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
        folder_path="space_deploy",
        repo_id=REPO_ID,
        repo_type="space",
        token=token
    )
    
    print(f"Deployed to: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    create_space_files()
    deploy()
