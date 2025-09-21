#!/usr/bin/env python3
"""
ML Binary Classifier for Malicious Prompt Detection
with AI Explainability using SHAP
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def extract_features(df):
    """Extract numerical features from AST and CFG data"""
    features = []
    
    for _, row in df.iterrows():
        ast_data = row['AST']
        cfg_data = row['CFG']
        
        # Basic features
        feat = {
            'length': row['Length'],
            'token_count': len(ast_data['tokens']),
            'sentence_count': ast_data['sentences'],
            'entity_count': len(ast_data['entities']),
            'cfg_rule_count': cfg_data['rule_count'],
            'cfg_depth': cfg_data['cfg_depth'],
        }
        
        # POS tag frequencies (top 10)
        pos_counter = Counter(ast_data['pos_tags'])
        top_pos = ['NOUN', 'VERB', 'ADJ', 'PROPN', 'PRON', 'ADP', 'DET', 'ADV', 'AUX', 'NUM']
        for pos in top_pos:
            feat[f'pos_{pos}'] = pos_counter.get(pos, 0)
        
        # Dependency relation frequencies (top 5)
        deps = [dep[1] for dep in ast_data['dependencies']]
        dep_counter = Counter(deps)
        top_deps = ['nsubj', 'dobj', 'prep', 'pobj', 'amod']
        for dep in top_deps:
            feat[f'dep_{dep}'] = dep_counter.get(dep, 0)
        
        features.append(feat)
    
    return pd.DataFrame(features)

class MaliciousPromptClassifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def train(self, X, y):
        """Train the classifier"""
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        self.model.fit(X_scaled, y)
        
    def predict(self, X):
        """Make predictions"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def explain_prediction(self, X, idx=0):
        """Simple feature importance explanation"""
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        prediction = self.predict(X.iloc[[idx]])[0]
        probability = self.predict_proba(X.iloc[[idx]])[0]
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {max(probability):.3f}")
        print(f"\nTop 10 Important Features:")
        for feature, importance in sorted_features[:10]:
            value = X.iloc[idx][feature]
            print(f"  {feature:<15}: {value:>8.2f} (importance: {importance:.3f})")

def main():
    parser = argparse.ArgumentParser(description='Malicious Prompt Classifier')
    parser.add_argument('--explain', '-e', action='store_true', 
                       help='Show explainability for predictions')
    parser.add_argument('--sample-idx', '-s', type=int, default=0,
                       help='Index of sample to explain (default: 0)')
    args = parser.parse_args()
    
    print("Loading balanced dataset with features...")
    with open('data/balanced_dataset_with_features.pkl', 'rb') as f:
        df = pickle.load(f)
    
    print("Extracting features...")
    X = extract_features(df)
    y = (df['Label'] == 'malicious').astype(int)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Label distribution: {Counter(y)}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train model
    classifier = MaliciousPromptClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate
    y_pred = classifier.predict(X_test)
    print(f"\nModel Performance:")
    print(classification_report(y_test, y_pred, target_names=['benign', 'malicious']))
    
    if args.explain:
        print(f"\n" + "="*50)
        print("EXPLAINABILITY ANALYSIS")
        print("="*50)
        
        # Explain prediction for specified sample
        sample_idx = min(args.sample_idx, len(X_test) - 1)
        print(f"\nExplaining prediction for test sample {sample_idx}:")
        print(f"Actual label: {'malicious' if y_test.iloc[sample_idx] else 'benign'}")
        classifier.explain_prediction(X_test, sample_idx)
        
        print(f"\nOriginal prompt preview:")
        test_prompts = df.iloc[X_test.index]
        print(f"'{test_prompts.iloc[sample_idx]['Prompt'][:100]}...'")

if __name__ == "__main__":
    main()