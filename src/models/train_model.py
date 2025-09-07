import pickle
from sklearn.metrics import classification_report, accuracy_score
from src.visualization.visualize import plot_roc_auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    f1_score, roc_auc_score, log_loss, brier_score_loss,
    average_precision_score
)

def logistic_regression_classification(X_train, X_test, y_train, y_test, threshold=0.3):
    """
    Train Logistic Regression on pre-split and scaled data and evaluate.

    Parameters
    ----------
    X_train, X_test : np.array or pd.DataFrame
        Scaled feature matrices
    y_train, y_test : pd.Series or np.array
        Target variables
    threshold : float, optional (default=0.35)
        Decision threshold for classification.

    Returns
    -------
    dict
        Dictionary containing trained model, predictions, and evaluation metrics
    """
    # Train Logistic Regression
    model = LogisticRegression(class_weight='balanced', max_iter=5000, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Evaluation metrics
    results = {
        'model': model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_true': y_test,
        'threshold': threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'log_loss': log_loss(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba)  # Precision-Recall AUC
    }

    # Print metrics
    print("LOGISTIC REGRESSION RESULTS ============================")
    print("Confusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['classification_report'])
    print("Accuracy:", results['accuracy'])
    print("F1 Score:", results['f1_score'])
    print("ROC-AUC:", results['roc_auc'])
    print("Log Loss:", results['log_loss'])
    print("Brier Score:", results['brier_score'])
    print("PR-AUC:", results['pr_auc'])

    #ROC/AUC CURVE
    plot_roc_auc(results["y_true"], results["y_proba"], model_name="Logistic Regression")

    return results

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, average_precision_score
)
import numpy as np

def xgb_classification_weighted(
    X_train, X_test, y_train, y_test,
    n_estimators=1000, max_depth=None, learning_rate=0.1,
    random_state=42, threshold=0.5, colsample_bytree=0.7, subsample=1.0):
    """
    Train a class-weighted XGBoost classifier and evaluate on test data.
    """

    # Handle class imbalance
    scale_pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))

    # Train model
    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        random_state=random_state,
        eval_metric='logloss'
        )
    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Initialize results
    results = {
        'model': model,
        'y_pred': y_pred,
        'y_true': y_test,
        'y_proba': y_proba,
        'threshold': threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'feature_importances': model.feature_importances_
    }

    # Metrics that need both classes
    if y_test.nunique() > 1:
        results.update({
            'roc_auc': roc_auc_score(y_test, y_proba),
            'log_loss': log_loss(y_test, y_proba, labels=[0,1]),
            'brier_score': brier_score_loss(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        })
    else:
        results.update({
            'roc_auc': None,
            'log_loss': None,
            'brier_score': None,
            'pr_auc': None
        })
        print("⚠️ Warning: Only one class in y_test, skipped ROC-AUC, log_loss, brier, PR-AUC.")

    # Print metrics
    print("XGBOOST MODEL RESULTS ============================")
    print("Confusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['classification_report'])
    print("Accuracy:", results['accuracy'])
    print("F1 Score:", results['f1_score'])
    if results['roc_auc'] is not None:
        print("ROC-AUC:", results['roc_auc'])
        print("Log Loss:", results['log_loss'])
        print("Brier Score:", results['brier_score'])
        print("PR-AUC:", results['pr_auc'])

    # ROC/AUC curve (only if both classes exist)
    if results['roc_auc'] is not None:
        plot_roc_auc(results["y_true"], results["y_proba"], model_name="XGBOOST")

    return results

from lightgbm import LGBMClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score, f1_score,
    roc_auc_score, log_loss, brier_score_loss, average_precision_score
)
import numpy as np

def lgbm_classification_weighted(
    X_train, X_test, y_train, y_test,
    n_estimators=1000, max_depth=-1, learning_rate=0.1,
    random_state=42, threshold=0.05, colsample_bytree=0.7, subsample=1.0, num_leaves=31):
    """
    Train a class-weighted LightGBM classifier and evaluate on test data.
    """

    # Handle class imbalance
    scale_pos_weight = np.sum(y_train == 0) / max(1, np.sum(y_train == 1))

    # Train model
    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        class_weight={0: 1, 1: scale_pos_weight},  # balancing classes
        colsample_bytree=colsample_bytree,
        subsample=subsample,
        num_leaves=num_leaves,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Predict probabilities
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Initialize results
    results = {
        'model': model,
        'y_pred': y_pred,
        'y_true': y_test,
        'y_proba': y_proba,
        'threshold': threshold,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'feature_importances': model.feature_importances_
    }

    # Metrics that need both classes
    if y_test.nunique() > 1:
        results.update({
            'roc_auc': roc_auc_score(y_test, y_proba),
            'log_loss': log_loss(y_test, y_proba, labels=[0, 1]),
            'brier_score': brier_score_loss(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba)
        })
    else:
        results.update({
            'roc_auc': None,
            'log_loss': None,
            'brier_score': None,
            'pr_auc': None
        })
        print("⚠️ Warning: Only one class in y_test, skipped ROC-AUC, log_loss, brier, PR-AUC.")

    # Print metrics
    print("LIGHTGBM MODEL RESULTS ============================")
    print("Confusion Matrix:\n", results['confusion_matrix'])
    print("\nClassification Report:\n", results['classification_report'])
    print("Accuracy:", results['accuracy'])
    print("F1 Score:", results['f1_score'])
    if results['roc_auc'] is not None:
        print("ROC-AUC:", results['roc_auc'])
        print("Log Loss:", results['log_loss'])
        print("Brier Score:", results['brier_score'])
        print("PR-AUC:", results['pr_auc'])

    # ROC/AUC curve (only if both classes exist)
    if results['roc_auc'] is not None:
        plot_roc_auc(results["y_true"], results["y_proba"], model_name="LightGBM")

    return results
