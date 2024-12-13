# modelTraining.py
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import joblib
import json
from sklearn.preprocessing import MinMaxScaler

def train_and_evaluate_model(X, y, create_model_func, class_weight=None, n_splits=5, features=None, output_dir='artifacts'):
    """
    Trains and evaluates a classification model using TimeSeriesSplit cross-validation.

    Parameters:
    - X (np.ndarray): Feature sequences with shape (samples, window_size, n_features).
    - y (np.ndarray): Labels with shape (samples,).
    - create_model_func (function): Function to create and compile the Keras model.
    - class_weight (dict, optional): Dictionary mapping class indices to weights.
    - n_splits (int): Number of cross-validation splits.
    - features (list, optional): List of feature names.
    - output_dir (str): Directory to save models and artifacts.

    Returns:
    - results (list of tuples): List containing (accuracy, precision, recall, f1_score) for each fold.
    - best_model (tf.keras.Model): The best-performing model based on F1 score.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []
    best_f1 = -1
    best_model = None
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    fold = 1
    for train_idx, test_idx in tscv.split(X):
        print(f"Training Fold {fold}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Initialize and fit the scaler on the training data
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        # Create and compile the model
        model = create_model_func(input_shape=X_train_scaled.shape[1:])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
        # Define EarlyStopping callback
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        # Train the model
        model.fit(
            X_train_scaled, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test_scaled, y_test),
            class_weight=class_weight,  # Utilize class weights
            callbacks=[early_stop],
            verbose=1
        )
        
        # Predictions
        y_pred_prob = model.predict(X_test_scaled)
        y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"Fold {fold} Results - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
        results.append((acc, prec, rec, f1))
        
        # Save the model for the current fold
        model_filename = os.path.join(output_dir, f'model_fold{fold}.keras')
        model.save(model_filename)
        print(f"Model for Fold {fold} saved to {model_filename}")
        
        # Save the scaler for the current fold
        scaler_filename = os.path.join(output_dir, f'scaler_fold{fold}.joblib')
        joblib.dump(scaler, scaler_filename)
        print(f"Scaler for Fold {fold} saved to {scaler_filename}")
        
        # Update best model based on F1 score
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
        
        fold += 1
    
    return results, best_model

def save_metrics(results, output_dir='artifacts'):
    """
    Saves the cross-validation metrics to a JSON file.

    Parameters:
    - results (list of tuples): List containing (accuracy, precision, recall, f1_score) for each fold.
    - output_dir (str): Directory to save the metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        'folds': [],
        'average': {}
    }

    acc_list = []
    prec_list = []
    rec_list = []
    f1_list = []

    for idx, (acc, prec, rec, f1) in enumerate(results, start=1):
        metrics['folds'].append({
            'fold': idx,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)

    metrics['average'] = {
        'accuracy': np.mean(acc_list),
        'precision': np.mean(prec_list),
        'recall': np.mean(rec_list),
        'f1_score': np.mean(f1_list)
    }

    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {os.path.join(output_dir, 'metrics.json')}")
