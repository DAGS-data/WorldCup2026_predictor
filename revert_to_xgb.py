import re

for filepath in [
    '/opt/data/wc2026-predictor/frontend/index.html',
    '/opt/data/wc2026-predictor/backend/frontend/index.html',
]:
    with open(filepath) as f:
        c = f.read()

    # API endpoint version
    c = c.replace('/api/predict/v3', '/api/predict/v2')
    c = c.replace('/api/bracket/v3', '/api/bracket/v2')
    
    # Data field names
    c = c.replace('data.v3_prediction', 'data.v2_prediction')
    
    # Labels
    c = c.replace('Logistic Regression', 'XGBoost')
    c = c.replace('📊 Logistic', '🤖 XGBoost')
    
    # Model tag
    c = c.replace(
        'L2-regularized · 38 features · 86% accuracy · no overfitting',
        '38 features · 79.5% accuracy · 0.918 AUC'
    )
    
    # Comments
    c = c.replace('PRIMARY: Logistic (knockout)', 'PRIMARY: XGBoost (knockout)')
    c = c.replace('PRIMARY: Logistic advance probability', 'PRIMARY: XGBoost advance probability')
    
    # SHAP explanation revert (contribution → shap)
    c = c.replace('f.contribution > 0', 'f.shap > 0')
    c = c.replace("f.contribution.toFixed(3)", "f.shap.toFixed(3)")
    
    # Explanation labels
    c = c.replace('(feature contributions)', '')
    c = c.replace('Why? (feature contributions)', 'Why?')
    
    # Fix the row: from logistic format back to SHAP format
    # logistic: coef: '+f.coefficient.toFixed(3)  →  val: '+f.value.toFixed(2)
    c = c.replace("coef: '+f.coefficient.toFixed(3)", "val: '+f.value.toFixed(2)")
    
    # Remove featLabel variable (not needed for SHAP)
    c = c.replace(
        "const featLabel = f.feature.replace(/_/g, ' ');\n          ",
        ""
    )
    c = c.replace("featLabel", "f.feature")
    
    # Fix prediction summary text
    c = c.replace('📊 Logistic: 🏆', '🤖 XGBoost: 🏆')
    
    # Loading text
    c = c.replace('Predicting with Logistic Regression...', 'Predicting with XGBoost...')
    c = c.replace('Simulating tournament with Logistic Regression...', 'Simulating tournament with XGBoost...')
    
    with open(filepath, 'w') as f:
        f.write(c)

print('Both frontend files reverted to XGBoost')
