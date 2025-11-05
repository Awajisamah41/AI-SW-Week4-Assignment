"""
Predictive Analytics for Resource Allocation
Dataset: Kaggle Breast Cancer Dataset
Goal: Binary Classification (Malignant vs Benign)
"""

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, classification_report, 
                             confusion_matrix, roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("="*70)
print("PREDICTIVE ANALYTICS: BREAST CANCER CLASSIFICATION")
print("="*70)

# ============================================================================
# STEP 1: LOAD AND EXPLORE DATA
# ============================================================================
print("\n[STEP 1] Loading Dataset...")

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='diagnosis')

# Map targets: 0 = Malignant (High Priority), 1 = Benign (Low Priority)
y_labels = y.map({0: 'Malignant', 1: 'Benign'})

print(f"✓ Dataset loaded successfully!")
print(f"  - Total samples: {len(X)}")
print(f"  - Total features: {X.shape[1]}")
print(f"  - Classes: {data.target_names}")
print(f"\nClass Distribution:")
print(y_labels.value_counts())
print(f"\nClass Balance: {y_labels.value_counts(normalize=True).round(3)}")

# Display first few rows
print("\n" + "="*70)
print("Sample Data (First 5 Features):")
print("="*70)
print(X.iloc[:5, :5])

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2] Data Preprocessing...")

# Check for missing values
missing_vals = X.isnull().sum().sum()
print(f"✓ Missing values: {missing_vals}")

# Check for duplicates
duplicates = X.duplicated().sum()
print(f"✓ Duplicate rows: {duplicates}")

# Feature statistics
print(f"\nFeature Statistics:")
print(X.describe().iloc[:, :5].round(2))

# Split data into train and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Data split completed:")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Testing samples: {len(X_test)}")

# Feature scaling (standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Feature scaling applied (StandardScaler)")

# Convert back to DataFrame for feature importance analysis
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# ============================================================================
# STEP 3: MODEL TRAINING
# ============================================================================
print("\n[STEP 3] Training Random Forest Classifier...")

# Initialize Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# Train the model
rf_model.fit(X_train_scaled, y_train)
print(f"✓ Model training completed!")
print(f"  - Number of trees: {rf_model.n_estimators}")
print(f"  - Max depth: {rf_model.max_depth}")

# Cross-validation score
cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"  - Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============================================================================
# STEP 4: MODEL EVALUATION
# ============================================================================
print("\n[STEP 4] Model Evaluation...")

# Make predictions
y_train_pred = rf_model.predict(X_train_scaled)
y_test_pred = rf_model.predict(X_test_scaled)
y_test_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_test_proba)

print("\n" + "="*70)
print("PERFORMANCE METRICS")
print("="*70)
print(f"\nTraining Set:")
print(f"  - Accuracy:  {train_accuracy:.4f}")
print(f"  - F1-Score:  {train_f1:.4f}")

print(f"\nTesting Set:")
print(f"  - Accuracy:  {test_accuracy:.4f}")
print(f"  - F1-Score:  {test_f1:.4f}")
print(f"  - ROC-AUC:   {roc_auc:.4f}")

# Classification Report
print("\n" + "="*70)
print("DETAILED CLASSIFICATION REPORT")
print("="*70)
print(classification_report(y_test, y_test_pred, 
                          target_names=['Malignant', 'Benign']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# ============================================================================
# STEP 5: VISUALIZATIONS
# ============================================================================
print("\n[STEP 5] Generating Visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Malignant', 'Benign'],
            yticklabels=['Malignant', 'Benign'])
axes[0, 0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# 2. Feature Importance (Top 15)
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

axes[0, 1].barh(range(len(feature_importance)), feature_importance['importance'])
axes[0, 1].set_yticks(range(len(feature_importance)))
axes[0, 1].set_yticklabels(feature_importance['feature'], fontsize=8)
axes[0, 1].invert_yaxis()
axes[0, 1].set_xlabel('Importance Score')
axes[0, 1].set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# 3. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
axes[1, 0].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve', fontsize=14, fontweight='bold')
axes[1, 0].legend(loc='lower right')
axes[1, 0].grid(alpha=0.3)

# 4. Model Performance Comparison
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'F1-Score', 'ROC-AUC'],
    'Train': [train_accuracy, train_f1, roc_auc],
    'Test': [test_accuracy, test_f1, roc_auc]
})

x = np.arange(len(metrics_df['Metric']))
width = 0.35
axes[1, 1].bar(x - width/2, metrics_df['Train'], width, label='Train', alpha=0.8)
axes[1, 1].bar(x + width/2, metrics_df['Test'], width, label='Test', alpha=0.8)
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(metrics_df['Metric'])
axes[1, 1].legend()
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(metrics_df['Train']):
    axes[1, 1].text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
for i, v in enumerate(metrics_df['Test']):
    axes[1, 1].text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('model_evaluation_results.png', dpi=300, bbox_inches='tight')
print("✓ Visualizations saved as 'model_evaluation_results.png'")
plt.show()

# ============================================================================
# STEP 6: SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 6] Sample Predictions...")
print("="*70)

sample_indices = np.random.choice(len(X_test), 5, replace=False)
sample_predictions = pd.DataFrame({
    'Actual': y_test.iloc[sample_indices].map({0: 'Malignant', 1: 'Benign'}).values,
    'Predicted': [['Malignant', 'Benign'][p] for p in y_test_pred[sample_indices]],
    'Confidence': y_test_proba[sample_indices].round(3)
})

print("\nSample Predictions:")
print(sample_predictions.to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"✓ Dataset: Breast Cancer Wisconsin (Diagnostic)")
print(f"✓ Model: Random Forest Classifier")
print(f"✓ Test Accuracy: {test_accuracy:.2%}")
print(f"✓ Test F1-Score: {test_f1:.4f}")
print(f"✓ ROC-AUC Score: {roc_auc:.4f}")
print(f"✓ Training samples: {len(X_train)}")
print(f"✓ Testing samples: {len(X_test)}")

if test_accuracy > 0.95:
    print("\n🎯 Excellent model performance! Ready for deployment.")
elif test_accuracy > 0.90:
    print("\n✓ Good model performance. Consider hyperparameter tuning.")
else:
    print("\n⚠ Model may need improvement. Try feature engineering or different algorithms.")

print("="*70)