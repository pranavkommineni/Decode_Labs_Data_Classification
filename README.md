#  Data Classification Using AI

A full end-to-end supervised classification pipeline using **scikit-learn**, supporting multiple datasets and 7 classifiers with cross-validation, optional hyperparameter tuning, and rich output artefacts.
---
---
## Quick Start

```bash
# path
The path for the project is: C:\Users\prana\Decode_Labs_Data_Classification> 

#install requirements
pip install -r requirements.txt

# Run on Iris (default)
python main.py

# Run on Wine dataset with only two models
python main.py --dataset wine --models random_forest svm

# Run on Breast Cancer with hyperparameter tuning + 10-fold CV
python main.py --dataset breast_cancer --tune --cv 10

# Custom CSV (last column = target)
python main.py --dataset data/my_data.csv

# Run tests
python -m pytest test_classification.py -v
```
---
---
## Architecture

```
project2_data_classification/
│
|_ data/
|   |- my_data.csv
├── main.py               ← CLI entry point & pipeline orchestrator
├── data_loader.py        ← Dataset loading, validation, train/test split, scaling
├── models.py             ← Registry of 7 classifiers with hyper-parameter grids
├── trainer.py            ← Fit, cross-validate, tune (GridSearchCV), evaluate
├── evaluator.py          ← Multi-model comparison table + charts
├── test_classification.py← 20+ unit & integration tests (pytest)
├── requirements.txt
└── results/              ← Auto-created at runtime
    ├── comparison.csv
    ├── results.json
    ├── model_comparison.png
    └── cm_<model>.png     (one per model)
```
---
---

#  Data Classification Pipeline

---

## Project Overview

The Data Classification Pipeline is a Machine Learning-based system designed to automate the classification of structured datasets using multiple supervised learning algorithms. The project performs data preprocessing, feature scaling, model training, cross-validation, performance evaluation, and visualization of classification results.

The objective of this project is to create a scalable and reusable classification framework capable of evaluating multiple machine learning models and identifying the best-performing classifier for a given dataset.

---

## Technologies Used

- Python 3.11
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- Logging Framework

---

## Project Workflow

1. Dataset Loading
2. Data Validation
3. Data Preprocessing
4. Train-Test Splitting
5. Feature Scaling
6. Model Training
7. Cross Validation
8. Model Evaluation
9. Confusion Matrix Generation
10. Results Visualization and Storage

---

##  [05/06/2026]
### Requirement Analysis & Project Planning

#### Tasks Completed
- Analyzed project requirements and objectives.
- Designed overall system architecture.
- Planned modular folder structure.
- Selected machine learning algorithms for implementation.
- Defined dataset handling strategy.

#### Outcome
- Project architecture finalized.
- Development roadmap established.

---

##  [06/06/2026]
### Dataset Loading & Processing Module

#### Tasks Completed
- Implemented dataset loading functionality.
- Added CSV dataset support.
- Developed automatic target column identification.
- Created dataset metadata extraction methods.
- Implemented feature-target separation logic.

#### Outcome
- Functional Data Loader module completed.
- Dataset preprocessing workflow initiated.

---

##  [07/06/2026]
### Data Preprocessing & Feature Engineering

#### Tasks Completed
- Implemented train-test splitting mechanism.
- Added feature scaling using StandardScaler.
- Developed preprocessing pipeline.
- Configured reproducibility using random states.
- Integrated target label encoding functionality.

#### Outcome
- Complete preprocessing module developed.
- Standardized feature transformation workflow established.

---

##  [08/06/2026]
### Machine Learning Model Integration

#### Tasks Completed
- Integrated Logistic Regression.
- Integrated Decision Tree Classifier.
- Integrated Random Forest Classifier.
- Integrated Gradient Boosting Classifier.
- Integrated Support Vector Machine (SVM).
- Integrated K-Nearest Neighbors (KNN).
- Integrated Naive Bayes Classifier.

#### Outcome
- Multi-model classification framework completed.
- Automated model evaluation pipeline prepared.

---

##  [09/06/2026]
### Evaluation System & Visualization

#### Tasks Completed
- Implemented model evaluation metrics.
- Added Accuracy Score calculations.
- Added Cross Validation support.
- Developed Confusion Matrix generation.
- Created visualization export functionality.
- Configured results storage mechanism.

#### Outcome
- Evaluation framework completed.
- Visualization module operational.

---

##  [10/06/2026]
### Debugging, Testing & Optimization

#### Tasks Completed
- Fixed LabelEncoder initialization issue.
- Corrected target encoding workflow.
- Resolved logging configuration errors.
- Fixed dataset loading exceptions.
- Addressed train-test split edge cases.
- Performed end-to-end pipeline testing.
- Generated confusion matrix reports.
- Verified successful execution across all classification models.

#### Outcome
- Stable and fully functional classification pipeline.
- Successful generation of evaluation reports and visualizations.

---

# Features Implemented

### Data Processing
- CSV Dataset Support
- Automatic Target Detection
- Label Encoding
- Feature Scaling

### Machine Learning Models
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Naive Bayes Classifier

### Evaluation & Analysis
- Accuracy Measurement
- Cross Validation
- Model Comparison
- Confusion Matrix Visualization

### Reporting
- Automated Results Storage
- Performance Charts
- Confusion Matrix Export
- Logging System

---

# Results Achieved

- Successfully loaded and processed datasets.
- Trained multiple machine learning classification models.
- Generated evaluation metrics for performance comparison.
- Created confusion matrix visualizations for each model.
- Automatically stored outputs in the results directory.
- Achieved successful end-to-end execution of the classification workflow.

---

# Future Enhancements

- XGBoost Integration
- LightGBM Integration
- Hyperparameter Optimization using Optuna
- SHAP Explainable AI
- Streamlit Dashboard
- FastAPI Deployment
- Docker Containerization
- MLflow Experiment Tracking
- Automated Model Selection
- PDF Report Generation

---

# Conclusion

The Data Classification Pipeline successfully demonstrates an end-to-end machine learning workflow, covering dataset preprocessing, model training, evaluation, and visualization. Through systematic development, debugging, and optimization during Week 2, the project evolved into a reliable classification framework capable of handling real-world machine learning tasks while maintaining scalability for future enhancements.