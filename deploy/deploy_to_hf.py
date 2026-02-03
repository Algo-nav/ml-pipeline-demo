"""
Deploy to HuggingFace Spaces
===========================
Uploads the trained model and creates a Gradio app.
"""

import os
import shutil
from huggingface_hub import HfApi, create_repo, upload_folder

# Configuration
SPACE_NAME = "iris-classifier-cicd"
USERNAME = "Nav772"
REPO_ID = f"{USERNAME}/{SPACE_NAME}"

def create_space_files():
    """Create the Gradio app files for the Space."""
    
    os.makedirs("space_deploy", exist_ok=True)
    
    # Copy model artifacts
    shutil.copy("artifacts/model.pkl", "space_deploy/model.pkl")
    shutil.copy("artifacts/metrics.json", "space_deploy/metrics.json")
    
    # Create app.py
    app_code = """
import gradio as gr
import pickle
import numpy as np
import json

# Load model and metrics
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("metrics.json", "r") as f:
    metrics = json.load(f)

# Iris class names
CLASS_NAMES = ["Setosa", "Versicolor", "Virginica"]
FEATURE_NAMES = ["Sepal Length (cm)", "Sepal Width (cm)", "Petal Length (cm)", "Petal Width (cm)"]

def predict(sepal_length, sepal_width, petal_length, petal_width):
    """Make prediction for given features."""
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Get prediction and probabilities
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Format results
    result = {CLASS_NAMES[i]: float(prob) for i, prob in enumerate(probabilities)}
    
    return result

# Build interface
demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(4.0, 8.0, value=5.8, label="Sepal Length (cm)"),
        gr.Slider(2.0, 4.5, value=3.0, label="Sepal Width (cm)"),
        gr.Slider(1.0, 7.0, value=4.0, label="Petal Length (cm)"),
        gr.Slider(0.1, 2.5, value=1.2, label="Petal Width (cm)"),
    ],
    outputs=gr.Label(num_top_classes=3, label="Prediction"),
    title="🌸 Iris Classifier (CI/CD Demo)",
    description=f"""
    Classify Iris flowers based on sepal and petal measurements.
    
    **Model Performance:** {metrics['accuracy']:.1%} accuracy
    
    **Pipeline:** This model was automatically trained and deployed via GitHub Actions CI/CD.
    """,
    examples=[
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [6.2, 2.9, 4.3, 1.3],  # Versicolor
        [7.7, 3.0, 6.1, 2.3],  # Virginica
    ]
)

if __name__ == "__main__":
    demo.launch()
"""
    
    with open("space_deploy/app.py", "w") as f:
        f.write(app_code)
    
    # Create requirements.txt
    requirements = """gradio
numpy
scikit-learn
"""
    
    with open("space_deploy/requirements.txt", "w") as f:
        f.write(requirements)
    
    # Create README.md
    readme = """---
title: Iris Classifier (CI/CD Demo)
emoji: 🌸
colorFrom: pink
colorTo: purple
sdk: gradio
sdk_version: 5.29.0
app_file: app.py
pinned: false
license: mit
---

# 🌸 Iris Classifier (CI/CD Demo)

A simple Iris flower classifier demonstrating **automated ML deployment via GitHub Actions**.

## 🔄 CI/CD Pipeline

This model is automatically:
1. **Validated** - Data quality checks
2. **Tested** - Unit tests for training code
3. **Trained** - Model training with accuracy threshold
4. **Deployed** - Pushed to this HuggingFace Space

Every push to the main branch triggers the full pipeline.

## 📊 Model Details

- **Algorithm:** Random Forest (100 trees)
- **Dataset:** Iris (150 samples, 3 classes)
- **Features:** Sepal length/width, Petal length/width

## 🔗 Repository

[GitHub Repository](https://github.com/Algo-nav/ml-pipeline-demo)

## 👤 Author

[Nav772](https://huggingface.co/Nav772)
"""
    
    with open("space_deploy/README.md", "w") as f:
        f.write(readme)
    
    print("✅ Space files created")


def deploy():
    """Deploy to HuggingFace Space."""
    
    token = os.environ.get("HF_TOK1EN")
    if not token:
        raise ValueError("HF_TOKEN environment variable not set")
    
    api = HfApi()
    
    # Create or get repo
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="gradio",
            token=token,
            exist_ok=True
        )
        print(f"✅ Space ready: {REPO_ID}")
    except Exception as e:
        print(f"⚠️ Repo note: {e}")
    
    # Upload files
    api.upload_folder(
        folder_path="space_deploy",
        repo_id=REPO_ID,
        repo_type="space",
        token=token
    )
    
    print(f"✅ Deployed to: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    create_space_files()
    deploy()
