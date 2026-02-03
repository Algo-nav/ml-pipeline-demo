# 🔄 ML Pipeline Demo

A complete CI/CD pipeline for Machine Learning using GitHub Actions.

## 🎯 What This Demonstrates

1. **Automated Data Validation** - Checks data quality before training
2. **Unit Testing** - Tests for data loading, training, and evaluation
3. **Model Training** - Trains with configurable parameters and accuracy threshold
4. **Auto-Deployment** - Deploys to HuggingFace Spaces on successful build

## 📁 Project Structure
```
ml-pipeline-demo/
├── .github/
│   └── workflows/
│       └── ml-pipeline.yml    # GitHub Actions workflow
├── src/
│   ├── train.py              # Model training script
│   └── validate_data.py      # Data validation script
├── tests/
│   └── test_model.py         # Unit tests
├── deploy/
│   └── deploy_to_hf.py       # HuggingFace deployment script
├── artifacts/                 # Generated model and metrics (gitignored)
├── requirements.txt
└── README.md
```

## 🔄 Pipeline Stages
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Validate   │────▶│    Test     │────▶│    Train    │────▶│   Deploy    │
│    Data     │     │   (pytest)  │     │   Model     │     │  to HF      │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

Each stage must pass before the next one runs.

## 🚀 Trigger Conditions

- **Push to main**: Runs full pipeline including deployment
- **Pull Request**: Runs validation, tests, and training (no deployment)
- **Manual**: Can be triggered manually from Actions tab

## 📊 Model Details

- **Task**: Iris flower classification (3 classes)
- **Algorithm**: Random Forest Classifier
- **Metrics**: Accuracy (threshold: 90%)

## 🔗 Links

- **Deployed Model**: [HuggingFace Space](https://huggingface.co/spaces/Nav772/iris-classifier-cicd)
- **Pipeline Runs**: [GitHub Actions](https://github.com/Algo-nav/ml-pipeline-demo/actions)

## 🛠️ Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run data validation
python src/validate_data.py

# Run tests
pytest tests/ -v

# Train model
python src/train.py --n_estimators 100

# Deploy (requires HF_TOKEN)
export HF_TOKEN="your-token"
python deploy/deploy_to_hf.py
```

## 👤 Author

[Nav772](https://huggingface.co/Nav772) / [Algo-nav](https://github.com/Algo-nav)
