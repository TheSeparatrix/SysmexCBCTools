"""
Data_B example for Dis-AE 2

This example demonstrates using Dis-AE 2 on the Data_B dataset from the DomainBed research.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from disae2 import DisAE


def encode_labels(series):
    """Encode categorical labels as integers."""
    unique_values = sorted(series.unique())
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return series.map(mapping).values, mapping


def load_data_B(data_path):
    """Load and preprocess Data_B dataset."""
    print(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    # Feature columns (32 features)
    feature_columns = [str(i) for i in range(32)]
    X = data[feature_columns].values

    # Task column
    task_column = "ClassCategory_0"
    y_task_raw = data[task_column]

    # Domain columns
    domain_columns = ["Machine", "studytime_binned", "vendelay_binned"]
    y_domains_raw = data[domain_columns]

    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels
    y_task, task_mapping = encode_labels(y_task_raw)
    y_task = y_task.reshape(-1, 1)  # Single task

    # Encode each domain factor
    y_domains = []
    domain_mappings = []
    for col in domain_columns:
        encoded, mapping = encode_labels(data[col])
        y_domains.append(encoded)
        domain_mappings.append(mapping)
    y_domains = np.column_stack(y_domains)

    print(f"Data loaded: {X.shape[0]} samples")
    print(f"Task classes: {len(task_mapping)}")
    print(f"Domain cardinalities: {[len(m) for m in domain_mappings]}")

    return X, y_task, y_domains, scaler, task_mapping, domain_mappings


def main():
    # Load Data_B
    data_path = "/home/dk659/MDANN/DomainBed/domainbed/data/dataset_B/data_B.csv"
    X, y_task, y_domains, scaler, task_mapping, domain_mappings = load_data_B(data_path)

    # Split into train and validation
    # Using stratified split on first domain factor (Machine)
    X_train, X_val, y_task_train, y_task_val, y_domains_train, y_domains_val = train_test_split(
        X, y_task, y_domains,
        test_size=0.2,
        stratify=y_domains[:, 0],  # Stratify by Machine
        random_state=42
    )

    print(f"\nTrain samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print()

    # Initialize Dis-AE 2 model
    model = DisAE(
        input_dim=32,
        latent_dim=16,
        shared_dim=8,
        private_dim=8,
        num_tasks=[len(task_mapping)],  # Single classification task
        num_domains=[len(m) for m in domain_mappings],  # Three domain factors
        hidden_dims=[64, 32],
        reconstruction_weight=0.1,
        adversarial_weight=1.0,
        orthogonality_weight=0.1,
        learning_rate=0.01,
        batch_size=128,
        device='cuda',
        random_state=42,
    )

    print("Model configuration:")
    print(f"  Input dim: {model.input_dim}")
    print(f"  Shared dim: {model.shared_dim}")
    print(f"  Private dim: {model.private_dim}")
    print(f"  Num tasks: {model.num_tasks}")
    print(f"  Num domains: {model.num_domains}")
    print()

    # Train the model
    print("Training model...")
    model.fit(
        X_train,
        y_task_train,
        y_domains_train,
        X_val=X_val,
        y_tasks_val=y_task_val,
        y_domains_val=y_domains_val,
        max_epochs=100,
        early_stopping_patience=16,
        verbose=True,
    )
    print()

    # Evaluate on validation set
    print("Evaluation on validation set:")
    print("=" * 60)

    # Task prediction
    y_pred = model.predict_tasks(X_val)
    task_accuracy = (y_pred[0] == y_task_val[:, 0]).mean()
    print(f"\nTask accuracy: {task_accuracy:.4f}")

    # Task probabilities
    y_proba = model.predict_tasks_proba(X_val)
    print(f"Task prediction confidence (mean max probability): {y_proba[0].max(axis=1).mean():.4f}")

    # Domain predictions (should be poor)
    print("\nDomain prediction accuracies (lower indicates better domain invariance):")
    d_pred = model.predict_domains(X_val)
    for i, (preds, mapping) in enumerate(zip(d_pred, domain_mappings)):
        accuracy = (preds == y_domains_val[:, i]).mean()
        random_chance = 1.0 / len(mapping)
        print(f"  Domain {i+1}: {accuracy:.4f} (random chance: {random_chance:.4f})")

    # Get embeddings
    print("\nEmbedding analysis:")
    shared, private = model.embed(X_val, private=True)
    print(f"  Shared features shape: {shared.shape}")
    print(f"  Private features shape: {private.shape}")
    print(f"  Shared features mean magnitude: {np.abs(shared).mean():.4f}")
    print(f"  Private features mean magnitude: {np.abs(private).mean():.4f}")

    # Reconstruction quality
    X_reconstructed = model.reconstruct(X_val)
    reconstruction_mse = np.mean((X_val - X_reconstructed) ** 2)
    print(f"\nReconstruction MSE: {reconstruction_mse:.6f}")

    # Feature correlation analysis
    shared_corr = np.corrcoef(shared.T)
    private_corr = np.corrcoef(private.T)
    shared_private_corr = np.corrcoef(
        np.concatenate([shared, private], axis=1).T
    )[:shared.shape[1], shared.shape[1]:]

    print(f"\nFeature space analysis:")
    print(f"  Shared-Shared correlation (mean abs): {np.abs(shared_corr).mean():.4f}")
    print(f"  Private-Private correlation (mean abs): {np.abs(private_corr).mean():.4f}")
    print(f"  Shared-Private correlation (mean abs): {np.abs(shared_private_corr).mean():.4f}")
    print(f"  (Lower Shared-Private correlation indicates better feature separation)")

    # Save model
    model_path = 'disae2_data_B.pkl'
    print(f"\nSaving model to {model_path}...")
    model.save(model_path)
    print("Model saved successfully!")

    print("\n" + "=" * 60)
    print("Data_B example completed successfully!")


if __name__ == "__main__":
    main()
