import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, learning_curve
)
from sklearn.ensemble import (
    RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# ─── Optional: XGBoost & LightGBM ────────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'kaggle_student_performance.csv')

NUMERIC_FEATURES = [
    'Age', 'StudyHoursPerWeek', 'AttendanceRate',
    'MathScore', 'ReadingScore', 'WritingScore'
]
CATEGORICAL_FEATURES = ['Gender', 'Department']


def load_dataset():
    df = pd.read_csv(DATA_FILE)
    df['Distinction'] = (df['CGPA'] >= 8.5).astype(int)
    return df


def _build_preprocessor():
    """Build the ColumnTransformer preprocessor shared across models."""
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer(transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])


def _get_preprocessed_data():
    """Return preprocessed X, y, and the fitted preprocessor."""
    df = load_dataset()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df['Distinction'].values

    preprocessor = _build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y, preprocessor, df


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  ENSEMBLE METHODS  — Bagging, Boosting (XGBoost, LightGBM)
# ═══════════════════════════════════════════════════════════════════════════════

def run_ensemble_comparison():
    """Compare Bagging (RF) and Boosting (XGB, LGBM, GradientBoosting) models."""
    X, y, preprocessor, df = _get_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {}

    # ── Bagging Ensemble ──────────────────────────────────────────────────────
    # 1a. Bagging with Decision Tree base
    models['bagging_dt'] = {
        'name': 'Bagging (Decision Tree)',
        'type': 'bagging',
        'model': BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=50, random_state=42, n_jobs=-1
        )
    }
    # 1b. Random Forest (Bagging + Feature Randomisation)
    models['random_forest'] = {
        'name': 'Random Forest',
        'type': 'bagging',
        'model': RandomForestClassifier(
            n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
        )
    }

    # ── Boosting Ensemble ─────────────────────────────────────────────────────
    # 2a. Sklearn Gradient Boosting (always available)
    models['gradient_boosting'] = {
        'name': 'Gradient Boosting (sklearn)',
        'type': 'boosting',
        'model': GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )
    }
    # 2b. XGBoost
    if HAS_XGB:
        models['xgboost'] = {
            'name': 'XGBoost',
            'type': 'boosting',
            'model': XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=42, eval_metric='logloss', use_label_encoder=False
            )
        }
    # 2c. LightGBM
    if HAS_LGBM:
        models['lightgbm'] = {
            'name': 'LightGBM',
            'type': 'boosting',
            'model': LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=3,
                random_state=42, verbose=-1
            )
        }

    results = {}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    for key, entry in models.items():
        mdl = entry['model']
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)

        # Probability scores for ROC-AUC
        if hasattr(mdl, 'predict_proba'):
            y_prob = mdl.predict_proba(X_test)[:, 1]
        else:
            y_prob = mdl.decision_function(X_test)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_prob))

        # Cross-Val
        cv_scores = cross_val_score(mdl, X, y, cv=cv, scoring='accuracy')

        # Feature importance (if supported)
        importance = None
        if hasattr(mdl, 'feature_importances_'):
            importance = [round(float(v), 4) for v in mdl.feature_importances_]

        results[key] = {
            'name': entry['name'],
            'type': entry['type'],
            'accuracy': round(acc * 100, 1),
            'precision': round(prec * 100, 1),
            'recall': round(rec * 100, 1),
            'f1_score': round(f1, 3),
            'roc_auc': round(auc, 3),
            'cv_mean': round(float(np.mean(cv_scores)) * 100, 1),
            'cv_std': round(float(np.std(cv_scores)) * 100, 1),
            'cv_folds': [round(float(s) * 100, 1) for s in cv_scores],
            'feature_importance': importance
        }

    return {
        'models': results,
        'xgb_available': HAS_XGB,
        'lgbm_available': HAS_LGBM,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  BIAS-VARIANCE TRADE-OFF
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_bias_variance():
    """
    Demonstrate bias-variance trade-off by training Decision Trees at
    increasing max_depth. Low depth → high bias (underfitting), high depth
    → high variance (overfitting).
    """
    X, y, _, _ = _get_preprocessed_data()

    depths = list(range(1, 12))
    results = []

    for d in depths:
        train_scores = []
        test_scores = []

        # Repeated holdout for stability
        for seed in range(5):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=seed * 10 + d
            )
            tree = DecisionTreeClassifier(max_depth=d, random_state=42)
            tree.fit(X_train, y_train)
            train_scores.append(accuracy_score(y_train, tree.predict(X_train)))
            test_scores.append(accuracy_score(y_test, tree.predict(X_test)))

        mean_train = round(float(np.mean(train_scores)) * 100, 1)
        mean_test = round(float(np.mean(test_scores)) * 100, 1)
        std_test = round(float(np.std(test_scores)) * 100, 1)

        results.append({
            'depth': d,
            'train_accuracy': mean_train,
            'test_accuracy': mean_test,
            'test_std': std_test,
            'gap': round(mean_train - mean_test, 1)
        })

    # Identify optimal zone
    best_idx = max(range(len(results)), key=lambda i: results[i]['test_accuracy'])
    optimal_depth = results[best_idx]['depth']

    return {
        'depth_analysis': results,
        'optimal_depth': optimal_depth,
        'interpretation': {
            'low_depth': 'High bias (underfitting) — model is too simple to capture patterns.',
            'optimal_depth': f'Best generalisation at max_depth={optimal_depth}.',
            'high_depth': 'High variance (overfitting) — model memorises training noise.'
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  OVERFITTING & UNDERFITTING  — Learning Curves
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_overfitting_underfitting():
    """
    Generate learning curves for three models representing underfitting,
    good fit, and overfitting regimes.
    """
    X, y, _, _ = _get_preprocessed_data()
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    scenarios = {
        'underfit': {
            'label': 'Underfitting (Shallow Tree, depth=1)',
            'model': DecisionTreeClassifier(max_depth=1, random_state=42),
        },
        'good_fit': {
            'label': 'Good Fit (Random Forest, depth=5)',
            'model': RandomForestClassifier(
                n_estimators=50, max_depth=5, random_state=42, n_jobs=-1
            ),
        },
        'overfit': {
            'label': 'Overfitting (Deep Tree, no limit)',
            'model': DecisionTreeClassifier(max_depth=None, random_state=42),
        }
    }

    results = {}

    for key, scenario in scenarios.items():
        train_sizes, train_scores, val_scores = learning_curve(
            scenario['model'], X, y,
            train_sizes=np.linspace(0.2, 1.0, 6),
            cv=cv, scoring='accuracy', n_jobs=-1
        )

        results[key] = {
            'label': scenario['label'],
            'train_sizes': [int(s) for s in train_sizes],
            'train_mean': [round(float(s) * 100, 1) for s in np.mean(train_scores, axis=1)],
            'train_std': [round(float(s) * 100, 1) for s in np.std(train_scores, axis=1)],
            'val_mean': [round(float(s) * 100, 1) for s in np.mean(val_scores, axis=1)],
            'val_std': [round(float(s) * 100, 1) for s in np.std(val_scores, axis=1)],
        }

    return {
        'scenarios': results,
        'causes': [
            'Underfitting: Model is too simple (few parameters, high bias). Training AND validation scores are low.',
            'Overfitting: Model is too complex (many parameters, high variance). Training score is high but validation score is low.',
            'Good Fit: Model complexity is balanced. Training and validation scores converge to a high value.'
        ],
        'prevention': [
            'Add more features / increase model complexity to fix underfitting.',
            'Use regularisation, pruning, dropout, or more data to prevent overfitting.',
            'Use cross-validation and early stopping to monitor generalisation.'
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  REGULARIZATION  (L1 / L2)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_regularization():
    """
    Compare Logistic Regression with No regularisation, L1 (Lasso),
    L2 (Ridge), and Elastic Net at various C values (inverse strength).
    """
    X, y, _, _ = _get_preprocessed_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # ── C-sweep for L1 vs L2 ─────────────────────────────────────────────────
    c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    sweep = {'l1': [], 'l2': []}

    for C in c_values:
        for penalty in ['l1', 'l2']:
            solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
            lr = LogisticRegression(
                C=C, penalty=penalty, solver=solver,
                random_state=42, max_iter=2000
            )
            lr.fit(X_train, y_train)
            train_acc = round(float(accuracy_score(y_train, lr.predict(X_train))) * 100, 1)
            test_acc = round(float(accuracy_score(y_test, lr.predict(X_test))) * 100, 1)
            cv_scores = cross_val_score(lr, X, y, cv=cv, scoring='accuracy')

            n_nonzero = int(np.sum(np.abs(lr.coef_[0]) > 1e-6))

            sweep[penalty].append({
                'C': C,
                'train_accuracy': train_acc,
                'test_accuracy': test_acc,
                'cv_mean': round(float(np.mean(cv_scores)) * 100, 1),
                'n_features_active': n_nonzero,
                'total_features': len(lr.coef_[0])
            })

    # ── Coefficient comparison at C=1.0 ───────────────────────────────────────
    coef_comparison = {}
    for penalty in ['l1', 'l2']:
        solver = 'liblinear' if penalty == 'l1' else 'lbfgs'
        lr = LogisticRegression(
            C=1.0, penalty=penalty, solver=solver,
            random_state=42, max_iter=2000
        )
        lr.fit(X_train, y_train)
        coef_comparison[penalty] = [round(float(c), 4) for c in lr.coef_[0]]

    # No-regularisation baseline
    lr_none = LogisticRegression(
        penalty=None, solver='lbfgs', random_state=42, max_iter=2000
    )
    lr_none.fit(X_train, y_train)
    coef_comparison['none'] = [round(float(c), 4) for c in lr_none.coef_[0]]

    return {
        'c_sweep': sweep,
        'c_values': c_values,
        'coefficient_comparison': coef_comparison,
        'explanation': {
            'l1': 'L1 (Lasso) adds |w| penalty → drives weak coefficients exactly to zero → built-in feature selection.',
            'l2': 'L2 (Ridge) adds w² penalty → shrinks all coefficients towards zero uniformly → prevents any single feature from dominating.',
            'C_meaning': 'C = 1/λ (inverse regularisation strength). Small C → strong regularisation. Large C → weak regularisation (closer to unregularised).',
            'purpose': 'Regularisation reduces overfitting by constraining model weights, improving generalisation on unseen data.'
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("[WEEK 13-14] Testing Ensemble Engine ...")
    print("=" * 60)

    ens = run_ensemble_comparison()
    for k, v in ens['models'].items():
        print(f"  {v['name']}: Acc={v['accuracy']}% | AUC={v['roc_auc']}")

    bv = analyze_bias_variance()
    print(f"\nOptimal tree depth: {bv['optimal_depth']}")

    of = analyze_overfitting_underfitting()
    for k, v in of['scenarios'].items():
        print(f"  {v['label']}: train final={v['train_mean'][-1]}% val final={v['val_mean'][-1]}%")

    reg = analyze_regularization()
    print(f"\nL1 C-sweep points: {len(reg['c_sweep']['l1'])}")
    print(f"L2 C-sweep points: {len(reg['c_sweep']['l2'])}")
