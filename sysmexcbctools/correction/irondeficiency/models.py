import copy
import json

import numpy as np
import torch
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    TransformerMixin,
    is_classifier,
)
from sklearn.metrics import accuracy_score
from sklearn.utils import check_array
from torch import nn
from torch.utils.data import WeightedRandomSampler


def weights_init(layer_in):
    """'He' initialisation of weights for ReLU activation function."""
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(tensor=layer_in.weight, nonlinearity="relu")
        layer_in.bias.data.fill_(0.0)


def heteroscedastic_loss(y_pred, var_pred, y_true, eps=1e-8):
    """Heteroscedastic loss function for regression tasks.
    Args:
        y_pred (torch.Tensor): predicted mean values
        var_pred (torch.Tensor): predicted variance values
        y_true (torch.Tensor): true values
        eps (float, optional): small value to avoid division by zero.
    Returns:
        torch.Tensor: heteroscedastic loss
    """
    return -1 * torch.mean(
        # # higher weight to Ferritin < 15
        # (15 / (y_true + eps))
        (
            torch.log(1 / (torch.sqrt(2 * np.pi * var_pred**2 + eps)))
            - (y_true - y_pred) ** 2 / (2 * var_pred**2 + eps)
        )
    )


def heteroscedastic_lognormal_loss(y_pred, log_var_pred, y_true, eps=1e-8):
    """Heteroscedastic lognormal loss function for regression tasks.

    Args:
        y_pred (torch.Tensor): Predicted log-mean values (log(mu))
        log_var_pred (torch.Tensor): Predicted log-variance values (log(sigma^2))
        y_true (torch.Tensor): True values
        eps (float, optional): Small value to avoid numerical issues.

    Returns:
        torch.Tensor: Heteroscedastic loss
    """
    # Ensure y_true is strictly positive
    y_true = torch.clamp(y_true, min=eps)

    # Convert log variance to variance
    var_pred = torch.exp(log_var_pred)

    # Calculate log(y_true)
    log_y_true = torch.log(y_true + eps)

    # Calculate the loss
    loss = (
        log_y_true
        + torch.log(2 * np.pi * var_pred)
        + (log_y_true - y_pred) ** 2 / (2 * var_pred + eps)
    )

    # # higher weight to Ferritin < 30
    # weight = 50 - 50 / 60 * (y_true - 50)
    # weight = 0.5 * y_true
    # loss = loss * weight

    return torch.mean(loss)


class DenseNetwork(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list, dropout: float = 0.0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout

        self.layers = nn.Sequential()
        self.layers.add_module(
            "layer_0", nn.Linear(self.input_dim, self.hidden_dims[0])
        )
        self.layers.add_module("act_0", nn.ReLU())
        self.layers.add_module("norm_0", nn.BatchNorm1d(self.hidden_dims[0]))
        self.layers.add_module("drop_0", nn.Dropout(self.dropout))
        for i in range(len(self.hidden_dims) - 1):
            self.layers.add_module(
                f"layer_{i+1}",
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i + 1]),
            )
            self.layers.add_module(f"act_{i+1}", nn.ReLU())
            self.layers.add_module(
                f"norm_{i+1}", nn.BatchNorm1d(self.hidden_dims[i + 1])
            )
            self.layers.add_module(f"drop_{i+1}", nn.Dropout(self.dropout))

        self.layers.add_module(
            f"layer_{len(self.hidden_dims)}",
            nn.Linear(self.hidden_dims[-1], self.output_dim),
        )

    def initialise_weights(self):
        self.apply(weights_init)

    def forward(self, x):
        return self.layers(x)


class DenseClassifier(DenseNetwork):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list, dropout: float = 0.0
    ):
        super().__init__(input_dim, output_dim, hidden_dims, dropout)
        self.output_dim = output_dim
        self.initialise_weights()

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.softmax(self.forward(x), dim=1)

    def fit(
        self,
        x,
        y,
        n_epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        verbose: bool = False,
        balance_classes: bool = False,
        weight_decay: float = 0.0,
        x_val=None,
        y_val=None,
    ):
        if verbose:
            print(
                f"Training model for {n_epochs} epochs with batch size {batch_size} and learning rate {lr}"
            )
            print("Model architecture:")
            print(self)
        dataset = torch.utils.data.TensorDataset(x, y)

        _has_val = False
        if x_val is not None or y_val is not None:
            assert (
                x_val is not None and y_val is not None
            ), "Both x_val and y_val must be provided"
            _has_val = True

        if balance_classes:
            class_counts = np.bincount(y)
            class_weights = 1 / class_counts
            sample_weights = class_weights[y]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_history = []
        for epoch in range(n_epochs):
            losses = {}
            for i, (x_batch, y_batch) in enumerate(dataloader):
                self.train()
                optimiser.zero_grad()
                y_pred = self.forward(x_batch)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimiser.step()
            losses["Epoch"] = epoch
            losses["Loss"] = loss.item()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
            if _has_val:
                self.eval()
                with torch.no_grad():
                    y_pred_val = self.forward(x_val)
                    val_loss = criterion(y_pred_val, y_val)
                    losses["Val Loss"] = val_loss.item()
                    if verbose and epoch % 10 == 0:
                        print(f"Validation Loss: {val_loss.item()}")
            loss_history.append(losses)
        return self, loss_history


class PyTorchClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        dropout: float = 0.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.model = DenseClassifier(input_dim, output_dim, hidden_dims, dropout)

    def fit(
        self,
        X,
        y,
        verbose: bool = False,
        balance_classes: bool = False,
        weight_decay: float = 0.01,
        n_iter: int = 100,
        batch_size: int = 128,
        lr: float = 1e-3,
        X_val=None,
        y_val=None,
        return_loss_history: bool = False,
        sample_weight=None,
    ):
        self.n_features_in_ = X.shape[1]
        # if y is shape (n_samples,), make it torch.long, otherwise torch.float32
        if len(y.shape) == 1:
            _y_dtype = torch.long
        else:
            _y_dtype = torch.float32

        if sample_weight is not None:
            balance_classes = True
            print(
                "Warning: sample_weight is not supported, using balance_classes instead"
            )

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=_y_dtype)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=_y_dtype)
        _, loss_history = self.model.fit(
            X,
            y,
            n_iter,
            batch_size,
            lr,
            verbose,
            balance_classes,
            weight_decay=weight_decay,
            x_val=X_val,
            y_val=y_val,
        )
        self.classes_ = np.unique(y)
        if return_loss_history:
            return self, loss_history
        return self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.model.predict(X).detach().numpy()

    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        return self.model.predict_proba(X).detach().numpy()

    def decision_function(self, X):
        self.model.eval()
        X = torch.tensor(X, dtype=torch.float32)
        return self.model.forward(X).detach().numpy()

    def save_model(self, path):
        model_architecture = {
            "class_name": self.model.__class__.__name__,
            "model_params": {
                "input_dim": self.model.input_dim,
                "output_dim": self.model.output_dim,
                "hidden_dims": self.model.hidden_dims,
                "dropout": self.model.dropout,
            },
            "state_dict": {
                k: v.numpy().tolist()
                for k, v in self.model.state_dict().items()
                if "num_batches_tracked" not in k
            },
        }
        with open(path, "w") as f:
            json.dump(model_architecture, f)

    def load_model(self, path):
        with open(path, "r") as f:
            model_architecture = json.load(f)
        self.model = DenseClassifier(
            model_architecture["model_params"]["input_dim"],
            model_architecture["model_params"]["output_dim"],
            model_architecture["model_params"]["hidden_dims"],
            model_architecture["model_params"]["dropout"],
        )
        self.model.load_state_dict(
            {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in model_architecture["state_dict"].items()
            }
        )
        return self

    # calling the model should be equivalent to predict
    def __call__(self, X):
        return self.predict(X)


class DenseBinaryClassifier(DenseNetwork):
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0.0):
        super().__init__(input_dim, 1, hidden_dims, dropout)
        self.output_dim = 1

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            return torch.round(torch.sigmoid(self.forward(x))).detach().numpy()

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).detach().numpy()

    def fit(
        self,
        x,
        y,
        n_epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        verbose: bool = False,
    ):
        if verbose:
            print(
                f"Training model for {n_epochs} epochs with batch size {batch_size} and learning rate {lr}"
            )
            print("Model architecture:")
            print(self)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        self.train()
        criterion = nn.BCEWithLogitsLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr=lr)
        for epoch in range(n_epochs):
            for x_batch, y_batch in enumerate(dataloader):
                optimiser.zero_grad()
                y_pred = torch.flatten(self.forward(x_batch))
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimiser.step()
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
        return self


class DenseRegressor(DenseNetwork):
    def __init__(
        self, input_dim: int, output_dim: int, hidden_dims: list, dropout: float = 0.0
    ):
        super().__init__(input_dim, output_dim, hidden_dims, dropout)
        self.output_dim = output_dim

    def forward(self, x):
        return self.layers(x)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)

    def fit(
        self,
        x,
        y,
        n_epochs: int = 200,
        batch_size: int = 128,
        lr: float = 1e-3,
        verbose: bool = False,
        heteroscedastic: bool = False,
        x_val=None,
        y_val=None,
        weight_decay: float = 0.0,
        weighting=None,
    ):
        if verbose:
            print(
                f"Training model for {n_epochs} epochs with batch size {batch_size} and learning rate {lr}"
            )
            print("Model architecture:")
            print(self)
            print(
                "Number of trainable parameters:",
                sum(p.numel() for p in self.parameters()),
            )

        _has_val = False
        if x_val is not None or y_val is not None:
            assert (
                x_val is not None and y_val is not None
            ), "Both x_val and y_val must be provided"
            _has_val = True

        dataset = torch.utils.data.TensorDataset(x, y)

        # do weightings
        if weighting not in [None, "balanced"]:
            raise NotImplementedError(
                "Only None and 'balanced' weighting schemes are implemented"
            )
        if weighting == "balanced":
            # divide y space into 10 bins
            y_bins = np.linspace(y.min(), y.max(), 10)
            # make a weighted sampler that gives higher weight to bins with less samples
            sample_weights = np.zeros(len(y))
            for i in range(len(y_bins) - 1):
                mask = (y.cpu().numpy() >= y_bins[i]) & (
                    y.cpu().numpy() < y_bins[i + 1]
                )
                sample_weights[mask] = 1 / np.sum(mask)
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, sampler=sampler
            )
        else:
            dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=True
            )

        # do training
        self.train()
        optimiser = torch.optim.Adam(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        loss_history = []
        if heteroscedastic:
            criterion = heteroscedastic_lognormal_loss
            for epoch in range(n_epochs):
                losses = {}
                epoch_loss = []
                self.train()
                for x_batch, y_batch in dataloader:
                    optimiser.zero_grad()
                    model_out = self.forward(x_batch)
                    y_pred = model_out[:, 0]  # predicted log-mean
                    log_var_pred = model_out[:, 1]  # predicted log-variance

                    loss = criterion(y_pred, log_var_pred, y_batch)
                    # print("DEBUG", y_pred, var_pred, y_batch)
                    # print("DEBUG", loss)
                    # loss = torch.mean(loss)
                    loss.backward()
                    optimiser.step()
                    epoch_loss.append(loss.item())
                losses["Epoch"] = epoch
                losses["Loss"] = np.mean(epoch_loss)
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {losses['Loss']}")
                if _has_val:
                    self.eval()
                    with torch.no_grad():
                        model_out = self.forward(x_val)
                        val_loss = criterion(
                            model_out[:, 0],
                            model_out[:, 1],
                            y_val,
                        )
                        losses["Val Loss"] = val_loss.item()
                        if verbose and epoch % 10 == 0:
                            print(f"Validation Loss: {val_loss.item()}")
                loss_history.append(losses)
        else:
            criterion = nn.MSELoss()
            for epoch in range(n_epochs):
                losses = {}
                epoch_loss = []
                for x_batch, y_batch in dataloader:
                    optimiser.zero_grad()
                    y_pred = self.forward(x_batch)
                    loss = criterion(y_pred, y_batch.view(y_pred.shape))
                    loss.backward()
                    optimiser.step()
                    epoch_loss.append(loss.item())
                losses["Epoch"] = epoch
                losses["Loss"] = np.mean(epoch_loss)
                if verbose and epoch % 10 == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss.item()}")
                if _has_val:
                    self.eval()
                    with torch.no_grad():
                        val_loss = criterion(self.forward(x_val), y_val.view(-1, 1))
                        losses["Val Loss"] = val_loss.item()
                        if verbose and epoch % 10 == 0:
                            print(f"Validation Loss: {val_loss.item()}")
                loss_history.append(losses)
        return self, loss_history


class PyTorchRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list,
        dropout: float = 0.0,
        heteroscedastic: bool = False,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.model = DenseRegressor(input_dim, output_dim, hidden_dims, dropout)
        self.heteroscedastic = heteroscedastic

    def fit(
        self,
        X,
        y,
        verbose: bool = False,
        batch_size: int = 128,
        lr: float = 1e-3,
        n_iter: int = 200,
        return_loss_history: bool = False,
        X_val=None,
        y_val=None,
        weight_decay: float = 0.0,
        weighting=None,
    ):
        self.n_features_in_ = X.shape[1]
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
        self.model, loss_history = self.model.fit(
            X,
            y,
            n_iter,
            batch_size,
            lr,
            verbose,
            heteroscedastic=self.heteroscedastic,
            x_val=X_val,
            y_val=y_val,
            weight_decay=weight_decay,
            weighting=weighting,
        )
        return self, loss_history if return_loss_history else self

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        if self.heteroscedastic:
            outs = self.model.predict(X)
            return (torch.exp(outs[:, 0] - torch.exp(outs[:, 1]))).detach().numpy()
        else:
            return self.model.predict(X).detach().numpy()

    def save_model(self, path):
        model_architecture = {
            "class_name": self.model.__class__.__name__,
            "model_params": {
                "input_dim": self.model.input_dim,
                "output_dim": self.model.output_dim,
                "hidden_dims": self.model.hidden_dims,
                "dropout": self.model.dropout,
                "heteroscedastic": self.heteroscedastic,
            },
            "state_dict": {
                k: v.numpy().tolist()
                for k, v in self.model.state_dict().items()
                if "num_batches_tracked" not in k
            },
        }
        with open(path, "w") as f:
            json.dump(model_architecture, f)

    def load_model(self, path):
        with open(path, "r") as f:
            model_architecture = json.load(f)
        self.model = DenseRegressor(
            model_architecture["model_params"]["input_dim"],
            model_architecture["model_params"]["output_dim"],
            model_architecture["model_params"]["hidden_dims"],
            model_architecture["model_params"]["dropout"],
            # model_architecture["model_params"]["heteroscedastic"],
        )
        self.heteroscedastic = model_architecture["model_params"]["heteroscedastic"]
        self.model.load_state_dict(
            {
                k: torch.tensor(v, dtype=torch.float32)
                for k, v in model_architecture["state_dict"].items()
            }
        )
        return self


class StackingEstimator(BaseEstimator, TransformerMixin):
    """Taken from TPOT's source code as TPOT is not compatible with current Python versions:
    Meta-transformer for adding predictions and/or class probabilities as synthetic feature(s).

    Parameters
    ----------
    estimator : object
        The base estimator from which the transformer is built.
    """

    def __init__(self, estimator):
        """Create a StackingEstimator object.

        Parameters
        ----------
        estimator: object with fit, predict, and predict_proba methods.
            The estimator to generate synthetic features from.
        """
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        """Fit the StackingEstimator meta-transformer.

        Parameters
        ----------
        X: array-like of shape (n_samples, n_features)
            The training input samples.
        y: array-like, shape (n_samples,)
            The target values (integers that correspond to classes in classification, real numbers in regression).
        fit_params:
            Other estimator-specific parameters.

        Returns
        -------
        self: object
            Returns a copy of the estimator
        """
        self.estimator.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """Transform data by adding two synthetic feature(s).

        Parameters
        ----------
        X: numpy ndarray, {n_samples, n_components}
            New data, where n_samples is the number of samples and n_components is the number of components.

        Returns
        -------
        X_transformed: array-like, shape (n_samples, n_features + 1) or (n_samples, n_features + 1 + n_classes) for classifier with predict_proba attribute
            The transformed feature set.
        """
        X = check_array(X)
        X_transformed = np.copy(X)
        # add class probabilities as a synthetic feature
        if is_classifier(self.estimator) and hasattr(self.estimator, "predict_proba"):
            y_pred_proba = self.estimator.predict_proba(X)
            # check all values that should be not infinity or not NAN
            if np.all(np.isfinite(y_pred_proba)):
                X_transformed = np.hstack((y_pred_proba, X))

        # add class prediction as a synthetic feature
        X_transformed = np.hstack(
            (np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed)
        )
        return X_transformed


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Ordinal classifier based on splitting k-class
    classification into k - 1 binary classification problems."""

    def __init__(self, clf):
        self.clf = clf
        self.clfs = {}
        self.unique_class = np.NaN

    def fit(self, X, y):
        self.unique_class = np.sort(np.unique(y))
        if self.unique_class.shape[0] > 2:
            for i in range(self.unique_class.shape[0] - 1):
                # for each k - 1 ordinal value we fit a binary classification problem
                binary_y = (y > self.unique_class[i]).astype(np.uint8)
                clf = copy.deepcopy(self.clf)
                clf.fit(X, binary_y)
                self.clfs[i] = clf
        return self

    def predict_proba(self, X):
        clfs_predict = {i: self.clfs[i].predict_proba(X) for i in self.clfs}
        predicted = []
        k = len(self.unique_class) - 1
        for i, y in enumerate(self.unique_class):
            if i == 0:
                # V1 = 1 - Pr(y > V1)
                predicted.append(1 - clfs_predict[0][:, 1])
            elif i < k:
                # Vi = Pr(y <= Vi) * Pr(y > Vi-1)
                predicted.append(
                    (1 - clfs_predict[i][:, 1]) * clfs_predict[i - 1][:, 1]
                )
            else:
                # Vk = Pr(y > Vk-1)
                predicted.append(clfs_predict[k - 1][:, 1])
        return np.vstack(predicted).T

    def predict(self, X):
        return self.unique_class[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)
