"""
Basic usage example for Dis-AE 2

This example demonstrates the basic API usage with synthetic data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from disae2 import DisAE

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
n_features = 32

# Features
X = np.random.randn(n_samples, n_features)

# Two classification tasks (binary and 3-class)
y_task_1 = np.random.randint(0, 2, size=n_samples)  # Binary task
y_task_2 = np.random.randint(0, 3, size=n_samples)  # 3-class task
y_tasks = np.column_stack([y_task_1, y_task_2])

# Three domain factors
y_domain_1 = np.random.randint(0, 5, size=n_samples)  # 5 machines
y_domain_2 = np.random.randint(0, 10, size=n_samples)  # 10 time bins
y_domain_3 = np.random.randint(0, 10, size=n_samples)  # 10 delay bins
y_domains = np.column_stack([y_domain_1, y_domain_2, y_domain_3])

# Split into train and validation
train_size = int(0.8 * n_samples)
X_train = X[:train_size]
y_tasks_train = y_tasks[:train_size]
y_domains_train = y_domains[:train_size]

X_val = X[train_size:]
y_tasks_val = y_tasks[train_size:]
y_domains_val = y_domains[train_size:]

print("Data shapes:")
print(f"  X_train: {X_train.shape}")
print(f"  y_tasks_train: {y_tasks_train.shape}")
print(f"  y_domains_train: {y_domains_train.shape}")
print()

# Initialize Dis-AE 2 model
model = DisAE(
    input_dim=n_features,
    latent_dim=16,
    num_tasks=[2, 3],  # Binary and 3-class tasks
    num_domains=[5, 10, 10],  # Three domain factors
    hidden_dims=[64, 32],
    reconstruction_weight=0.1,
    adversarial_weight=1.0,
    orthogonality_weight=0.1,
    learning_rate=0.01,
    batch_size=128,
    device='cuda',
    random_state=42,
)

print("Model initialized with parameters:")
print(f"  Input dim: {model.input_dim}")
print(f"  Latent dim: {model.latent_dim} (shared: {model.shared_dim}, private: {model.private_dim})")
print(f"  Num tasks: {model.num_tasks}")
print(f"  Num domains: {model.num_domains}")
print()

# Fit the model
print("Training model...")
model.fit(
    X_train,
    y_tasks_train,
    y_domains_train,
    X_val=X_val,
    y_tasks_val=y_tasks_val,
    y_domains_val=y_domains_val,
    max_epochs=100,
    early_stopping_patience=16,
    verbose=True,
)
print()

# Make predictions on test data
print("Making predictions...")

# Task predictions
y_pred = model.predict_tasks(X_val)
print(f"Task predictions: {len(y_pred)} tasks")
for i, preds in enumerate(y_pred):
    accuracy = (preds == y_tasks_val[:, i]).mean()
    print(f"  Task {i+1} accuracy: {accuracy:.4f}")

# Task probabilities
y_proba = model.predict_tasks_proba(X_val)
print(f"Task probabilities: {len(y_proba)} tasks")
for i, proba in enumerate(y_proba):
    print(f"  Task {i+1} probability shape: {proba.shape}")

# Domain predictions (should be poor for good domain generalization)
d_pred = model.predict_domains(X_val)
print(f"\nDomain predictions: {len(d_pred)} domain factors")
for i, preds in enumerate(d_pred):
    accuracy = (preds == y_domains_val[:, i]).mean()
    print(f"  Domain {i+1} accuracy: {accuracy:.4f} (lower is better for domain generalization)")

# Domain probabilities
d_proba = model.predict_domains_proba(X_val)
print(f"Domain probabilities: {len(d_proba)} domain factors")
for i, proba in enumerate(d_proba):
    print(f"  Domain {i+1} probability shape: {proba.shape}")

# Get embeddings
print("\nGenerating embeddings...")
shared, private = model.embed(X_val, private=True)
print(f"Shared features shape: {shared.shape}")
print(f"Private features shape: {private.shape}")

# Get only shared features
shared_only = model.embed(X_val, private=False)
print(f"Shared features only shape: {shared_only.shape}")

# Reconstruct data
print("\nReconstructing data...")
X_reconstructed = model.reconstruct(X_val)
reconstruction_error = np.mean((X_val - X_reconstructed) ** 2)
print(f"Reconstruction MSE: {reconstruction_error:.6f}")

# Save and load model
print("\nSaving model...")
model.save('model.pkl')
print("Model saved to model.pkl")

print("\nLoading model...")
loaded_model = DisAE.load('model.pkl')
print("Model loaded successfully")

# Verify loaded model works
y_pred_loaded = loaded_model.predict_tasks(X_val)
for i in range(len(y_pred)):
    assert np.array_equal(y_pred[i], y_pred_loaded[i]), f"Task {i} predictions don't match!"
print("Loaded model produces identical predictions")

print("\nExample completed successfully!")
