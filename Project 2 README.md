# 🤖 DecodeLabs AI — Project 2: Data Classification Using AI

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.21+-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-11557C?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2E7D32?style=for-the-badge)
![Batch](https://img.shields.io/badge/Batch-2026-E8553E?style=for-the-badge)

> **DecodeLabs Industrial Training Kit · Artificial Intelligence Track**
> Project 2 of the AI series — stepping from rule-based logic into the world of Supervised Machine Learning.

---

## 📌 Project Overview

This project implements a complete **supervised machine learning pipeline** for multi-class data classification. Building on the rule-based foundations of Project 1, this milestone moves into **predictive AI** — training a model on historical data so it can classify new, unseen inputs with intelligence and precision.

The model learns to identify three species of Iris flowers using four physical measurements, using the **K-Nearest Neighbors (KNN)** algorithm with proper preprocessing, hyperparameter tuning, and rigorous evaluation.

**Key insight:** We do not write the rules. We provide history, and the machine derives the logic.

---

## 🎯 Objectives

- Load and explore a structured tabular dataset (the Iris Benchmark)
- Apply **StandardScaler** to neutralise feature scale bias for fair distance calculations
- Split data into **80% training / 20% testing** with shuffle to remove ordering bias
- Use the **Elbow Method** to determine the optimal value of K
- Train a **KNeighborsClassifier** and generate predictions on unseen test data
- Evaluate performance using **Accuracy**, **F1 Score**, and **Confusion Matrix**
- Visualise results with Elbow curve, heatmap, and feature spread charts

---

## 🌸 Dataset: The Iris Benchmark

| Property | Value |
|---|---|
| Total Samples | 150 (perfectly balanced) |
| Classes | 3 — Setosa, Versicolor, Virginica |
| Features | 4 — Sepal Length, Sepal Width, Petal Length, Petal Width |
| Feature Ranges | 0.1 cm – 7.9 cm across dimensions |
| Train / Test Split | 120 samples / 30 samples (80/20) |

> ⭐ **Petal Length and Petal Width** are the strongest discriminators between species — they carry the most signal for classification.

---

## 🏗️ ML Pipeline (IPO Framework)

```
INPUT                    PROCESS                      OUTPUT
──────                   ───────                      ──────
Iris Domain Data    →    StandardScaler          →    Class Predictions
Feature Matrix           Train-Test Split              Confusion Matrix
(150 × 4)                KNN Algorithm (K=5)           F1 Score
                         Elbow Method                  Visualisations
```

### Step-by-Step Breakdown

| Step | Method | Purpose |
|------|--------|---------|
| 01 | `load_iris()` | Load 150-sample Iris dataset into X and y |
| 02 | `StandardScaler.fit_transform()` | Normalise to Mean=0, Variance=1 |
| 03 | `train_test_split()` | 80/20 shuffled split, `random_state=42` |
| 04 | Elbow loop K=1..20 | Find optimal K with minimum error rate |
| 05 | `KNeighborsClassifier.fit()` | Train KNN on X_train, y_train |
| 06 | `model.predict(X_test)` | Generate predictions on held-out test set |
| 07 | Matplotlib + Seaborn | Elbow plot, Confusion Matrix heatmap, Feature bars |
| 08 | `scaler.transform(new_sample)` | Predict on a brand-new unseen flower |

---

## 🤖 The Algorithm: K-Nearest Neighbors

KNN operates on the **Proximity Principle** — similar data points exist close together in feature space.

```
For each new test point:
  1. Calculate Euclidean distance to every training point
  2. Sort distances → select the K closest neighbours
  3. Take majority vote among those K labels
  4. Assign the winning class to the new point
```

```python
model = KNeighborsClassifier(n_neighbors=5)   # INSTANTIATE
model.fit(X_train, y_train)                   # FIT (memorise the map)
y_pred = model.predict(X_test)                # PREDICT (apply logic)
```

### Why K=5?

The **Elbow Method** evaluates error rate across K=1 to K=20:
- `K=1` → Overfits (memorises noise)
- `K=100` → Underfits (too much averaging)
- `K=5` → **Optimal: minimum error rate on the Iris test set**

---

## ⚖️ Feature Scaling: Why It Matters

KNN uses **Euclidean distance**. Without scaling, features with larger ranges (e.g., Petal Length: 1.0–6.9 cm) dominate over features with smaller ranges (e.g., Sepal Width: 2.0–4.4 cm), creating an unfair bias.

**StandardScaler** transforms each feature to `Mean = 0, Variance = 1`:

```
z = (x - μ) / σ
```

```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

After scaling, all four features contribute **equally** to the distance calculation.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Accuracy** | **100.00%** |
| **F1 Score (weighted)** | **1.0000** |
| **Optimal K** | **5** |
| Training Samples | 120 (80%) |
| Test Samples | 30 (20%) |

### Confusion Matrix

```
                Setosa    Versicolor    Virginica
  Setosa          10 ✓        0             0
  Versicolor       0         9 ✓            0
  Virginica        0          0           11 ✓
```

All 30 test samples correctly classified — **zero misclassifications** across all three species.

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Setosa | 1.00 | 1.00 | 1.00 | 10 |
| Versicolor | 1.00 | 1.00 | 1.00 | 9 |
| Virginica | 1.00 | 1.00 | 1.00 | 11 |
| **Weighted Avg** | **1.00** | **1.00** | **1.00** | **30** |

---

## 📈 Visualisations

The script automatically generates and saves a figure (`project2_knn_results.png`) containing three plots:

1. **Elbow Curve** — Error rate vs K value with optimal K highlighted in orange
2. **Confusion Matrix Heatmap** — Seaborn annotated grid showing per-class predictions
3. **Feature Spread** — Bar chart of standard deviation per feature after scaling

---

## 🧠 Key Concepts Explained

### Why Not Just Use Accuracy?

The **Accuracy Mirage**: in imbalanced datasets, a model predicting the majority class always gets high accuracy — but it is useless for detecting minority classes. The Iris dataset is balanced, so accuracy holds here, but F1 Score is always reported as best practice.

### Precision vs Recall vs F1

| Metric | Formula | Best Used When |
|--------|---------|---------------|
| Precision | TP / (TP + FP) | False alarms are costly (spam filters) |
| Recall | TP / (TP + FN) | Missing a case is costly (medical screening) |
| F1 Score | 2 × P × R / (P + R) | Need a single balanced metric |

### Confusion Matrix Quadrants

| | Predicted Positive | Predicted Negative |
|--|--|--|
| **Actually Positive** | TP ✓ (correct) | FN ✗ (missed — Type II) |
| **Actually Negative** | FP ✗ (false alarm — Type I) | TN ✓ (correct) |

---

## 🗂️ Project Structure

```
DecodeLabs-AI-Project2/
│
├── project2_knn_classifier.py      # Main Python script (8 steps)
├── project2_knn_results.png        # Auto-generated visualisation chart
├── DecodeLabs_P2_Presentation.pptx # 10-slide project presentation
├── DecodeLabs_P2_Report.docx       # Full technical report (11 sections)
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install scikit-learn numpy matplotlib seaborn
```

### Run the Project

```bash
python project2_knn_classifier.py
```

### Expected Output

```
============================================================
  DecodeLabs | Project 2: Data Classification Using AI
============================================================

📦 STEP 1 — Loading the Iris Dataset
   Total samples  : 150
   Features       : 4
   Classes        : 3  → ['setosa', 'versicolor', 'virginica']

⚖️  STEP 2 — Feature Scaling
   After StandardScaler: Mean ≈ 0, Variance ≈ 1  ✅

✂️  STEP 3 — Train-Test Split (80% / 20%)
   Training samples : 120
   Testing  samples : 30

🔍 STEP 4 — Finding Optimal K
   Optimal K found  : 5

🤖 STEP 5 — Training KNeighborsClassifier (K=5)
   Model trained successfully  ✅

📊 STEP 6 — Predictions & Evaluation
   Accuracy  : 100.00%
   F1 Score  : 1.0000

🌸 STEP 8 — Predicting a New Sample
   Predicted class  : SETOSA  🌷

✅ Project 2 Complete — KNN Classifier Trained & Validated
============================================================
```

---

## 📚 Libraries & Dependencies

| Library | Version | Role |
|---------|---------|------|
| `scikit-learn` | ≥ 1.0 | KNN classifier, StandardScaler, metrics |
| `numpy` | ≥ 1.21 | Array operations, elbow method |
| `matplotlib` | ≥ 3.4 | Elbow curve, figure layout |
| `seaborn` | ≥ 0.11 | Confusion matrix heatmap |
| `Python` | ≥ 3.8 | Core runtime |

---

## ⚠️ Limitations

- KNN stores the full training set in memory — prediction time scales with dataset size
- Performance is sensitive to K — the Elbow Method helps but isn't always definitive
- No native feature importance output (KNN is a black-box model)
- 100% accuracy reflects Iris's simplicity; real-world datasets will vary
- Struggles with highly non-linear decision boundaries

---

## 🔮 Future Enhancements

- [ ] Cross-validation (k-fold) for more robust accuracy estimates
- [ ] Algorithm comparison: Logistic Regression, Decision Tree, Random Forest, SVM
- [ ] Hyperparameter tuning with `GridSearchCV` (K, distance metric, weights)
- [ ] Feature selection via PCA or correlation analysis
- [ ] Apply to a real-world imbalanced dataset
- [ ] Flask / Streamlit web interface for live flower measurement predictions
- [ ] Progress to **Project 3: Deep Learning & Convolutional Neural Networks**

---

## 🏆 Qualification Criteria

This project is **mandatory** as part of the DecodeLabs AI Industrial Training Kit:

- ✅ Complete Project 2 to unlock subsequent projects
- ✅ All projects verified for quality before badge award
- ✅ Vital bridge to professional AI development for every intern

---

## 👨‍💻 About DecodeLabs

**DecodeLabs** is an industrial training platform empowering the next generation of AI and software engineers through hands-on, project-based learning.

> *"The absolute best way to master Artificial Intelligence is through hands-on practice, not just theory."*
> — DecodeLabs

🌐 [www.decodelabs.tech](https://www.decodelabs.tech)
📧 decodelabs.tech@gmail.com
📍 Greater Lucknow, India

---

## 📄 License

This project is part of the DecodeLabs Industrial Training Kit — Batch 2026.

---

*Built with 🧠 by a DecodeLabs AI Intern | Batch 2026*
