"""
Dis-AE 2: Domain Separation Network Multi-Domain Adversarial Autoencoder

A scikit-learn style interface for multi-task, multi-domain learning with
domain generalization capabilities through adversarial training and feature separation.
"""

import pickle
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .networks import Decoder, DSNFeaturizer, MultiTaskClassifier
from .training import EarlyStopping


class DisAE:
    """
    Dis-AE 2: Domain Separation Network Multi-Domain Adversarial Autoencoder

    A multi-task, multi-domain learning model with domain generalization capabilities.
    Uses Domain Separation Networks (DSN) to split features into:
    - Shared features: Domain-invariant, used for task prediction
    - Private features: Domain-specific, used for reconstruction

    Parameters
    ----------
    input_dim : int
        Dimensionality of input features
    latent_dim : int, default=16
        Total latent dimension (shared_dim + private_dim)
    shared_dim : int, optional
        Dimension of shared (domain-invariant) features. If None, defaults to latent_dim // 2.
        **Must equal private_dim** due to orthogonality loss implementation.
    private_dim : int, optional
        Dimension of private (domain-specific) features. If None, defaults to latent_dim // 2.
        **Must equal shared_dim** due to orthogonality loss implementation.
    num_tasks : List[int]
        List of number of classes for each task (e.g., [2, 3] for binary and 3-class tasks)
    num_domains : List[int]
        List of cardinalities for each domain factor (e.g., [5, 10, 10])
    hidden_dims : List[int], default=[64, 32]
        Hidden layer dimensions for encoder/decoder
    reconstruction_weight : float, default=0.1
        Weight for reconstruction loss
    adversarial_weight : float, default=1.0
        Weight for adversarial domain loss
    orthogonality_weight : float, default=0.1
        Weight for orthogonality loss between shared and private features
    learning_rate : float, default=0.01
        Learning rate for both generator and discriminator
    learning_rate_g : float, optional
        Learning rate for generator (overrides learning_rate if provided)
    learning_rate_d : float, optional
        Learning rate for discriminator (overrides learning_rate if provided)
    weight_decay : float, default=0.0001
        Weight decay for both generator and discriminator
    weight_decay_g : float, optional
        Weight decay for generator (overrides weight_decay if provided)
    weight_decay_d : float, optional
        Weight decay for discriminator (overrides weight_decay if provided)
    batch_size : int, default=128
        Batch size for training
    d_steps_per_g_step : int, default=1
        Number of discriminator updates per generator update
    beta1 : float, default=0.5
        Beta1 parameter for Adam optimizer
    device : str, default='cuda'
        Device to use for training ('cuda' or 'cpu')
    random_state : int, optional
        Random seed for reproducibility

    Attributes
    ----------
    dsn_featurizer_ : DSNFeaturizer
        The trained encoder network
    classifiers_ : MultiTaskClassifier
        The trained task classifiers
    decoder_ : Decoder
        The trained decoder network
    discriminators_ : nn.ModuleList
        The trained domain discriminators
    is_fitted_ : bool
        Whether the model has been fitted

    Examples
    --------
    >>> from disae2 import DisAE
    >>> import numpy as np
    >>>
    >>> # Create synthetic data
    >>> X = np.random.randn(1000, 32)
    >>> y_tasks = np.random.randint(0, 2, size=(1000, 2))  # 2 binary tasks
    >>> y_domains = np.random.randint(0, 5, size=(1000, 3))  # 3 domain factors
    >>>
    >>> # Initialize and train model
    >>> model = DisAE(
    ...     input_dim=32,
    ...     num_tasks=[2, 2],
    ...     num_domains=[5, 5, 5]
    ... )
    >>> model.fit(X, y_tasks, y_domains, max_epochs=100)
    >>>
    >>> # Make predictions
    >>> y_pred = model.predict_tasks(X)
    >>> shared, private = model.embed(X)
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        shared_dim: Optional[int] = None,
        private_dim: Optional[int] = None,
        num_tasks: List[int] = [2],
        num_domains: List[int] = [2],
        hidden_dims: List[int] = [64, 32],
        reconstruction_weight: float = 0.1,
        adversarial_weight: float = 1.0,
        orthogonality_weight: float = 0.1,
        learning_rate: float = 0.01,
        learning_rate_g: Optional[float] = None,
        learning_rate_d: Optional[float] = None,
        weight_decay: float = 0.0001,
        weight_decay_g: Optional[float] = None,
        weight_decay_d: Optional[float] = None,
        batch_size: int = 128,
        d_steps_per_g_step: int = 1,
        beta1: float = 0.5,
        device: str = "cuda",
        random_state: Optional[int] = None,
    ):
        # Store parameters
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.shared_dim = shared_dim if shared_dim is not None else latent_dim // 2
        self.private_dim = private_dim if private_dim is not None else latent_dim // 2

        # Enforce equal dimensions for orthogonality loss
        if self.shared_dim != self.private_dim:
            raise ValueError(
                f"shared_dim ({self.shared_dim}) must equal private_dim ({self.private_dim}). "
                f"The orthogonality loss requires equal-dimensional feature spaces for the "
                f"element-wise dot product computation. Use latent_dim={self.shared_dim + self.private_dim} "
                f"without specifying shared_dim/private_dim for equal split."
            )

        self.num_tasks = num_tasks
        self.num_domains = num_domains
        self.hidden_dims = hidden_dims
        self.reconstruction_weight = reconstruction_weight
        self.adversarial_weight = adversarial_weight
        self.orthogonality_weight = orthogonality_weight
        self.learning_rate = learning_rate
        self.learning_rate_g = (
            learning_rate_g if learning_rate_g is not None else learning_rate
        )
        self.learning_rate_d = (
            learning_rate_d if learning_rate_d is not None else learning_rate
        )
        self.weight_decay = weight_decay
        self.weight_decay_g = (
            weight_decay_g if weight_decay_g is not None else weight_decay
        )
        self.weight_decay_d = (
            weight_decay_d if weight_decay_d is not None else weight_decay
        )
        self.batch_size = batch_size
        self.d_steps_per_g_step = d_steps_per_g_step
        self.beta1 = beta1
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.random_state = random_state

        # Set random seed if provided
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Initialize networks (will be created during fit)
        self.dsn_featurizer_ = None
        self.classifiers_ = None
        self.decoder_ = None
        self.discriminators_ = None
        self.gen_opt_ = None
        self.disc_opt_ = None
        self.is_fitted_ = False
        self.update_count_ = 0

    def _build_networks(self):
        """Build neural network components."""
        # DSN Featurizer (encoder)
        self.dsn_featurizer_ = DSNFeaturizer(
            input_dim=self.input_dim,
            shared_dim=self.shared_dim,
            private_dim=self.private_dim,
            hidden_dims=self.hidden_dims,
        ).to(self.device)

        # Multi-task classifiers (operate on shared features only)
        self.classifiers_ = MultiTaskClassifier(
            input_dim=self.shared_dim,
            num_tasks=self.num_tasks,
        ).to(self.device)

        # Decoder (uses both shared and private features)
        self.decoder_ = Decoder(
            input_dim=self.shared_dim + self.private_dim,
            output_dim=self.input_dim,
            hidden_dims=list(reversed(self.hidden_dims)),
        ).to(self.device)

        # Multi-domain discriminators (operate on shared features only)
        self.discriminators_ = nn.ModuleList(
            [self._build_discriminator(num_classes) for num_classes in self.num_domains]
        ).to(self.device)

        # Optimizers
        self.gen_opt_ = torch.optim.Adam(
            list(self.dsn_featurizer_.parameters())
            + list(self.classifiers_.parameters())
            + list(self.decoder_.parameters()),
            lr=self.learning_rate_g,
            weight_decay=self.weight_decay_g,
            betas=(self.beta1, 0.9),
        )

        self.disc_opt_ = torch.optim.Adam(
            self.discriminators_.parameters(),
            lr=self.learning_rate_d,
            weight_decay=self.weight_decay_d,
            betas=(self.beta1, 0.9),
        )

    def _build_discriminator(self, num_classes: int) -> nn.Module:
        """Build a single discriminator network."""
        layers = []
        dims = [self.shared_dim] + self.hidden_dims + [num_classes]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output layer
                layers.append(nn.ReLU())

        return nn.Sequential(*layers)

    def _orthogonality_loss(
        self, shared_features: torch.Tensor, private_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute orthogonality loss between shared and private features."""
        # Normalize features
        shared_norm = F.normalize(shared_features, p=2, dim=1)
        private_norm = F.normalize(private_features, p=2, dim=1)

        # Compute cosine similarity and penalize non-zero values
        cosine_sim = torch.sum(shared_norm * private_norm, dim=1)
        return torch.mean(cosine_sim**2)

    def _train_step(
        self, x: torch.Tensor, y_tasks: torch.Tensor, y_domains: torch.Tensor
    ) -> dict:
        """Perform one training step (either discriminator or generator)."""
        self.update_count_ += 1

        # Forward pass through DSN
        shared_features, private_features = self.dsn_featurizer_(x)

        # Compute discriminator losses
        disc_losses = []
        for i, discriminator in enumerate(self.discriminators_):
            disc_out = discriminator(shared_features)
            disc_labels = y_domains[:, i]
            disc_loss = F.cross_entropy(disc_out, disc_labels)
            disc_losses.append(disc_loss)

        disc_loss = sum(disc_losses)

        # Alternate between discriminator and generator updates
        if self.update_count_ % (1 + self.d_steps_per_g_step) < self.d_steps_per_g_step:
            # Discriminator step
            self.disc_opt_.zero_grad()
            disc_loss.backward()
            self.disc_opt_.step()

            return {"disc_loss": disc_loss.item(), "step_type": "discriminator"}
        else:
            # Generator step
            # Task classification losses
            task_preds = self.classifiers_(shared_features)
            task_losses = []
            for i, preds in enumerate(task_preds):
                task_loss = F.cross_entropy(preds, y_tasks[:, i])
                task_losses.append(task_loss)
            classifier_loss = sum(task_losses)

            # Reconstruction loss
            combined_features = torch.cat([shared_features, private_features], dim=1)
            reconstructed = self.decoder_(combined_features)
            reconstruction_loss = F.mse_loss(reconstructed, x)

            # Orthogonality loss
            orthogonal_loss = self._orthogonality_loss(
                shared_features, private_features
            )

            # Combined generator loss
            gen_loss = (
                classifier_loss
                + self.adversarial_weight * (-disc_loss)
                + self.reconstruction_weight * reconstruction_loss
                + self.orthogonality_weight * orthogonal_loss
            )

            self.disc_opt_.zero_grad()
            self.gen_opt_.zero_grad()
            gen_loss.backward()
            self.gen_opt_.step()

            return {
                "gen_loss": gen_loss.item(),
                "classifier_loss": classifier_loss.item(),
                "reconstruction_loss": reconstruction_loss.item(),
                "orthogonal_loss": orthogonal_loss.item(),
                "disc_loss": disc_loss.item(),
                "step_type": "generator",
            }

    def _validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Compute validation loss and accuracy."""
        self.dsn_featurizer_.eval()
        self.classifiers_.eval()
        self.decoder_.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y_tasks, y_domains in val_loader:
                x = x.to(self.device)
                y_tasks = y_tasks.to(self.device)

                # Forward pass
                shared_features, private_features = self.dsn_featurizer_(x)
                task_preds = self.classifiers_(shared_features)

                # Compute loss (only for first task for simplicity)
                loss = F.cross_entropy(task_preds[0], y_tasks[:, 0])
                total_loss += loss.item() * x.size(0)

                # Compute accuracy (average across all tasks)
                for i, preds in enumerate(task_preds):
                    correct = (preds.argmax(dim=1) == y_tasks[:, i]).sum().item()
                    total_correct += correct

                total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / (total_samples * len(self.num_tasks))

        self.dsn_featurizer_.train()
        self.classifiers_.train()
        self.decoder_.train()

        return avg_loss, avg_accuracy

    def fit(
        self,
        X: np.ndarray,
        y_tasks: np.ndarray,
        y_domains: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_tasks_val: Optional[np.ndarray] = None,
        y_domains_val: Optional[np.ndarray] = None,
        max_epochs: int = 100,
        early_stopping_patience: int = 16,
        verbose: bool = True,
    ) -> "DisAE":
        """
        Fit the Dis-AE 2 model.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training data
        y_tasks : np.ndarray, shape (n_samples, n_tasks)
            Task labels (pre-encoded as integers)
        y_domains : np.ndarray, shape (n_samples, n_domains)
            Domain labels (pre-encoded as integers)
        X_val : np.ndarray, optional
            Validation data. If provided, enables early stopping.
        y_tasks_val : np.ndarray, optional
            Validation task labels
        y_domains_val : np.ndarray, optional
            Validation domain labels
        max_epochs : int, default=100
            Maximum number of training epochs
        early_stopping_patience : int, default=16
            Patience for early stopping (only used if validation data provided)
        verbose : bool, default=True
            Whether to print training progress

        Returns
        -------
        self : DisAE
            The fitted model
        """
        # Input validation
        assert X.shape[0] == y_tasks.shape[0] == y_domains.shape[0], (
            "X, y_tasks, and y_domains must have same number of samples"
        )
        assert y_tasks.shape[1] == len(self.num_tasks), (
            f"y_tasks must have {len(self.num_tasks)} columns"
        )
        assert y_domains.shape[1] == len(self.num_domains), (
            f"y_domains must have {len(self.num_domains)} columns"
        )

        # Build networks
        if not self.is_fitted_:
            self._build_networks()

        # Create data loaders
        X_tensor = torch.FloatTensor(X)
        y_tasks_tensor = torch.LongTensor(y_tasks)
        y_domains_tensor = torch.LongTensor(y_domains)
        train_dataset = TensorDataset(X_tensor, y_tasks_tensor, y_domains_tensor)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )

        # Validation data
        val_loader = None
        early_stopping = None
        if X_val is not None:
            assert y_tasks_val is not None and y_domains_val is not None, (
                "Must provide y_tasks_val and y_domains_val with X_val"
            )
            X_val_tensor = torch.FloatTensor(X_val)
            y_tasks_val_tensor = torch.LongTensor(y_tasks_val)
            y_domains_val_tensor = torch.LongTensor(y_domains_val)
            val_dataset = TensorDataset(
                X_val_tensor, y_tasks_val_tensor, y_domains_val_tensor
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )
            early_stopping = EarlyStopping(patience=early_stopping_patience, mode="min")

        # Training loop
        for epoch in range(max_epochs):
            epoch_losses = []

            for x, y_tasks, y_domains in train_loader:
                x = x.to(self.device)
                y_tasks = y_tasks.to(self.device)
                y_domains = y_domains.to(self.device)

                loss_dict = self._train_step(x, y_tasks, y_domains)

                if loss_dict["step_type"] == "generator":
                    epoch_losses.append(loss_dict["gen_loss"])

            # Validation
            if val_loader is not None:
                val_loss, val_accuracy = self._validate(val_loader)

                if verbose and epoch % 10 == 0:
                    avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
                    print(
                        f"Epoch {epoch}/{max_epochs} - "
                        f"Train Loss: {avg_train_loss:.4f}, "
                        f"Val Loss: {val_loss:.4f}, "
                        f"Val Acc: {val_accuracy:.4f}"
                    )

                # Early stopping check
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break
            else:
                if verbose and epoch % 10 == 0:
                    avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
                    print(
                        f"Epoch {epoch}/{max_epochs} - Train Loss: {avg_train_loss:.4f}"
                    )

        self.is_fitted_ = True
        return self

    def predict_tasks(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict task labels for all tasks.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data

        Returns
        -------
        predictions : List[np.ndarray]
            List of predicted labels, one array per task
        """
        assert self.is_fitted_, "Model must be fitted before prediction"

        self.dsn_featurizer_.eval()
        self.classifiers_.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            shared_features, _ = self.dsn_featurizer_(X_tensor)
            task_preds = self.classifiers_(shared_features)
            predictions = [preds.argmax(dim=1).cpu().numpy() for preds in task_preds]

        self.dsn_featurizer_.train()
        self.classifiers_.train()

        return predictions

    def predict_tasks_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict task label probabilities for all tasks.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data

        Returns
        -------
        probabilities : List[np.ndarray]
            List of probability arrays, one per task
        """
        assert self.is_fitted_, "Model must be fitted before prediction"

        self.dsn_featurizer_.eval()
        self.classifiers_.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            shared_features, _ = self.dsn_featurizer_(X_tensor)
            task_preds = self.classifiers_(shared_features)
            probabilities = [
                F.softmax(preds, dim=1).cpu().numpy() for preds in task_preds
            ]

        self.dsn_featurizer_.train()
        self.classifiers_.train()

        return probabilities

    def predict_domains(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict domain labels for all domain factors.

        Note: For a well-trained domain generalization model, these predictions
        should have poor performance, indicating domain-invariant features.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data

        Returns
        -------
        predictions : List[np.ndarray]
            List of predicted domain labels, one array per domain factor
        """
        assert self.is_fitted_, "Model must be fitted before prediction"

        self.dsn_featurizer_.eval()
        self.discriminators_.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            shared_features, _ = self.dsn_featurizer_(X_tensor)
            predictions = [
                discriminator(shared_features).argmax(dim=1).cpu().numpy()
                for discriminator in self.discriminators_
            ]

        self.dsn_featurizer_.train()
        self.discriminators_.train()

        return predictions

    def predict_domains_proba(self, X: np.ndarray) -> List[np.ndarray]:
        """
        Predict domain label probabilities for all domain factors.

        Note: For a well-trained domain generalization model, these probabilities
        should be close to uniform, indicating domain-invariant features.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data

        Returns
        -------
        probabilities : List[np.ndarray]
            List of probability arrays, one per domain factor
        """
        assert self.is_fitted_, "Model must be fitted before prediction"

        self.dsn_featurizer_.eval()
        self.discriminators_.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            shared_features, _ = self.dsn_featurizer_(X_tensor)
            probabilities = [
                F.softmax(discriminator(shared_features), dim=1).cpu().numpy()
                for discriminator in self.discriminators_
            ]

        self.dsn_featurizer_.train()
        self.discriminators_.train()

        return probabilities

    def embed(
        self, X: np.ndarray, private: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate domain-agnostic data representations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data
        private : bool, default=True
            If True, returns both shared and private features separately.
            If False, returns only shared features.

        Returns
        -------
        embeddings : np.ndarray or Tuple[np.ndarray, np.ndarray]
            If private=True: (shared_features, private_features)
            If private=False: shared_features only
        """
        assert self.is_fitted_, "Model must be fitted before embedding"

        self.dsn_featurizer_.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            shared_features, private_features = self.dsn_featurizer_(X_tensor)
            shared_np = shared_features.cpu().numpy()
            private_np = private_features.cpu().numpy()

        self.dsn_featurizer_.train()

        if private:
            return shared_np, private_np
        else:
            return shared_np

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct input data from learned representations.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input data

        Returns
        -------
        X_reconstructed : np.ndarray, shape (n_samples, n_features)
            Reconstructed data
        """
        assert self.is_fitted_, "Model must be fitted before reconstruction"

        self.dsn_featurizer_.eval()
        self.decoder_.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            shared_features, private_features = self.dsn_featurizer_(X_tensor)
            combined_features = torch.cat([shared_features, private_features], dim=1)
            reconstructed = self.decoder_(combined_features)
            X_reconstructed = reconstructed.cpu().numpy()

        self.dsn_featurizer_.train()
        self.decoder_.train()

        return X_reconstructed

    def save(self, path: str):
        """
        Save the model to disk.

        Parameters
        ----------
        path : str
            Path to save the model
        """
        assert self.is_fitted_, "Model must be fitted before saving"

        state = {
            "dsn_featurizer": self.dsn_featurizer_.state_dict(),
            "classifiers": self.classifiers_.state_dict(),
            "decoder": self.decoder_.state_dict(),
            "discriminators": self.discriminators_.state_dict(),
            "params": {
                "input_dim": self.input_dim,
                "latent_dim": self.latent_dim,
                "shared_dim": self.shared_dim,
                "private_dim": self.private_dim,
                "num_tasks": self.num_tasks,
                "num_domains": self.num_domains,
                "hidden_dims": self.hidden_dims,
                "reconstruction_weight": self.reconstruction_weight,
                "adversarial_weight": self.adversarial_weight,
                "orthogonality_weight": self.orthogonality_weight,
                "learning_rate": self.learning_rate,
                "learning_rate_g": self.learning_rate_g,
                "learning_rate_d": self.learning_rate_d,
                "weight_decay": self.weight_decay,
                "weight_decay_g": self.weight_decay_g,
                "weight_decay_d": self.weight_decay_d,
                "batch_size": self.batch_size,
                "d_steps_per_g_step": self.d_steps_per_g_step,
                "beta1": self.beta1,
                "device": self.device,
                "random_state": self.random_state,
            },
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: str) -> "DisAE":
        """
        Load a saved model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model

        Returns
        -------
        model : DisAE
            The loaded model
        """
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Create new instance
        model = cls(**state["params"])

        # Build networks
        model._build_networks()

        # Load state dicts
        model.dsn_featurizer_.load_state_dict(state["dsn_featurizer"])
        model.classifiers_.load_state_dict(state["classifiers"])
        model.decoder_.load_state_dict(state["decoder"])
        model.discriminators_.load_state_dict(state["discriminators"])

        model.is_fitted_ = True

        return model
