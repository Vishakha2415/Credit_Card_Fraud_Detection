import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score  # ADDED THESE
)
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("FRAUD DETECTION MODEL TRAINING")
print("="*60)

# Load data
print("\n📊 Loading dataset...")
df = pd.read_csv('creditcard.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Check class distribution
print("\n📈 Class Distribution:")
fraud_count = df['Class'].sum()
total_count = len(df)
print(f"Fraudulent: {fraud_count} ({fraud_count/total_count*100:.4f}%)")
print(f"Legitimate: {total_count-fraud_count} ({(total_count-fraud_count)/total_count*100:.4f}%)")

# ===== DATA PREPROCESSING =====
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

print(f"\n🔧 Original Features: {X.shape[1]}")

# Feature Engineering
print("\n🔧 Feature Engineering...")

# 1. Create time-based features
X['Hour'] = (X['Time'] // 3600) % 24
X['Is_Night'] = ((X['Hour'] >= 0) & (X['Hour'] <= 5)) | (X['Hour'] >= 22)
X['Is_Early_Morning'] = (X['Hour'] >= 5) & (X['Hour'] <= 9)

# 2. Amount features
X['Amount_Log'] = np.log1p(X['Amount'] + 1)
X['Amount_Scaled'] = (X['Amount'] - X['Amount'].mean()) / X['Amount'].std()

# 3. Interaction features
X['V1_V2'] = X['V1'] * X['V2']
X['V3_V4'] = X['V3'] * X['V4']
X['V14_V15'] = X['V14'] * X['V15']

# 4. PCA component ratios
X['V1_V2_Ratio'] = X['V1'] / (X['V2'] + 0.001)
X['V3_V4_Ratio'] = X['V3'] / (X['V4'] + 0.001)

# Drop original time column
X = X.drop(['Time'], axis=1)

print(f"✅ New Features: {X.shape[1]}")

# ===== HANDLE IMBALANCE =====
print("\n⚖️ Handling Class Imbalance...")
print(f"Before: Fraud cases = {y.sum()} ({y.mean()*100:.4f}%)")

# Separate fraud and non-fraud
fraud_df = X[y == 1]
non_fraud_df = X[y == 0]
fraud_labels = y[y == 1]
non_fraud_labels = y[y == 0]

# Undersample majority class
non_fraud_sampled = non_fraud_df.sample(n=len(fraud_df)*50, random_state=42)
non_fraud_labels_sampled = non_fraud_labels[non_fraud_sampled.index]

# Combine
X_balanced = pd.concat([fraud_df, non_fraud_sampled])
y_balanced = pd.concat([fraud_labels, non_fraud_labels_sampled])

print(f"After: Fraud cases = {y_balanced.sum()} ({y_balanced.mean()*100:.2f}%)")

# ===== SPLIT DATA =====
print("\n📊 Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, 
    test_size=0.2, 
    random_state=42,
    stratify=y_balanced
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ===== SCALING =====
print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===== MODEL TRAINING =====
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

print("\n🚀 Training Random Forest Classifier...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation - FIXED: Now imports are available
print("\n📈 Model Performance:")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\n🤖 Confusion Matrix:")
print(f"True Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Feature Importance
if hasattr(model, 'feature_importances_'):
    print("\n🔝 Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))

# ===== SAVE MODEL =====
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

# Save model and scaler
model_filename = 'fraud_model.pkl'
scaler_filename = 'scaler.pkl'

with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_filename, 'wb') as f:
    pickle.dump(scaler, f)

print(f"✅ Model saved as: {model_filename}")
print(f"✅ Scaler saved as: {scaler_filename}")

# ===== PREDICTION FUNCTION =====
print("\n" + "="*60)
print("PREDICTION FUNCTION")
print("="*60)

def predict_single_transaction(features_dict):
    """
    Predict fraud for a single transaction
    
    Args:
        features_dict: Dictionary with transaction features
    
    Returns:
        Dictionary with prediction results
    """
    # Convert to DataFrame
    df_input = pd.DataFrame([features_dict])
    
    # Apply same preprocessing
    df_input['Hour'] = (df_input['Time'] // 3600) % 24
    df_input['Is_Night'] = ((df_input['Hour'] >= 0) & (df_input['Hour'] <= 5)) | (df_input['Hour'] >= 22)
    df_input['Is_Early_Morning'] = (df_input['Hour'] >= 5) & (df_input['Hour'] <= 9)
    df_input['Amount_Log'] = np.log1p(df_input['Amount'] + 1)
    df_input['Amount_Scaled'] = (df_input['Amount'] - df_input['Amount'].mean()) / df_input['Amount'].std()
    df_input['V1_V2'] = df_input['V1'] * df_input['V2']
    df_input['V3_V4'] = df_input['V3'] * df_input['V4']
    df_input['V14_V15'] = df_input['V14'] * df_input['V15']
    df_input['V1_V2_Ratio'] = df_input['V1'] / (df_input['V2'] + 0.001)
    df_input['V3_V4_Ratio'] = df_input['V3'] / (df_input['V4'] + 0.001)
    
    # Drop time column
    df_input = df_input.drop(['Time'], axis=1)
    
    # Ensure all columns exist
    for col in X_train.columns:
        if col not in df_input.columns:
            df_input[col] = 0
    
    # Reorder columns
    df_input = df_input[X_train.columns]
    
    # Scale
    df_input_scaled = scaler.transform(df_input)
    
    # Predict
    fraud_prob = model.predict_proba(df_input_scaled)[0, 1]
    is_fraud = fraud_prob > 0.5
    
    # Risk factors
    risk_factors = []
    if features_dict.get('Amount', 0) > 1000:
        risk_factors.append("high_amount")
    hour = (features_dict.get('Time', 0) // 3600) % 24
    if hour <= 5 or hour >= 22:
        risk_factors.append("late_night")
    if abs(features_dict.get('V1', 0)) > 2:
        risk_factors.append("anomalous_v1")
    if abs(features_dict.get('V14', 0)) > 2:
        risk_factors.append("anomalous_v14")
    
    # Risk level
    if fraud_prob > 0.7:
        risk_level = "HIGH"
        recommendation = "Block and investigate"
    elif fraud_prob > 0.4:
        risk_level = "MEDIUM"
        recommendation = "Require additional verification"
    else:
        risk_level = "LOW"
        recommendation = "Approve transaction"
    
    return {
        'fraud_probability': float(fraud_prob),
        'is_fraud': bool(is_fraud),
        'confidence': float(fraud_prob) if is_fraud else float(1 - fraud_prob),
        'risk_level': risk_level,
        'risk_factors': risk_factors,
        'recommendation': recommendation
    }

# Test the function
print("\n🧪 Testing prediction function...")
test_transaction = {
    'Time': 50000,
    'V1': -1.359807, 'V2': -0.072781, 'V3': 2.536347, 'V4': 1.378155,
    'V5': -0.338321, 'V6': 0.462388, 'V7': 0.239599, 'V8': 0.098698,
    'V9': 0.363787, 'V10': 0.090794, 'V11': -0.551600, 'V12': -0.617801,
    'V13': -0.991390, 'V14': -0.311169, 'V15': 1.468177, 'V16': -0.470401,
    'V17': 0.207971, 'V18': 0.025791, 'V19': 0.403993, 'V20': 0.251412,
    'V21': -0.018307, 'V22': 0.277838, 'V23': -0.110474, 'V24': 0.066928,
    'V25': 0.128539, 'V26': -0.189115, 'V27': 0.133558, 'V28': -0.021053,
    'Amount': 149.62
}

result = predict_single_transaction(test_transaction)
print(f"Test Transaction Result:")
print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
print(f"  Is Fraud: {result['is_fraud']}")
print(f"  Risk Level: {result['risk_level']}")
print(f"  Risk Factors: {result['risk_factors']}")
print(f"  Recommendation: {result['recommendation']}")

print("\n" + "="*60)
print("🎯 TRAINING COMPLETE!")
print("="*60)
