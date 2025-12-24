"""
Neural network components for Dis-AE 2

Includes DSN featurizer, decoder, and multi-task classifier implementations.
"""

from typing import List

import torch.nn as nn


class DSNFeaturizer(nn.Module):
    """
    Domain Separation Network Featurizer

    Splits input into shared (domain-invariant) and private (domain-specific) features.

    Parameters
    ----------
    input_dim : int
        Input dimensionality
    shared_dim : int
        Shared feature dimensionality
    private_dim : int
        Private feature dimensionality
    hidden_dims : List[int]
        Hidden layer dimensions for the base encoder
    dropout : float, default=0.0
        Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        private_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
    ):
        super(DSNFeaturizer, self).__init__()

        self.input_dim = input_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim

        # Build base encoder
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.base_encoder = nn.Sequential(*layers)
        base_output_dim = dims[-1]

        # Projection layers for shared and private features
        self.shared_projection = nn.Linear(base_output_dim, shared_dim)
        self.private_projection = nn.Linear(base_output_dim, private_dim)

        self.n_shared = shared_dim
        self.n_private = private_dim

    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input features

        Returns
        -------
        shared_features : torch.Tensor, shape (batch_size, shared_dim)
            Domain-invariant features
        private_features : torch.Tensor, shape (batch_size, private_dim)
            Domain-specific features
        """
        base_features = self.base_encoder(x)
        shared_features = self.shared_projection(base_features)
        private_features = self.private_projection(base_features)
        return shared_features, private_features


class Decoder(nn.Module):
    """
    MLP Decoder for reconstructing inputs from features

    Parameters
    ----------
    input_dim : int
        Feature dimensionality (typically shared_dim + private_dim)
    output_dim : int
        Output dimensionality (should match original input)
    hidden_dims : List[int]
        Hidden layer dimensions
    dropout : float, default=0.0
        Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float = 0.0,
    ):
        super(Decoder, self).__init__()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation on output layer
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Feature representation

        Returns
        -------
        reconstruction : torch.Tensor, shape (batch_size, output_dim)
            Reconstructed input
        """
        return self.decoder(x)


class MultiTaskClassifier(nn.Module):
    """
    Multi-task classifier

    Creates separate classifier heads for each task.

    Parameters
    ----------
    input_dim : int
        Input feature dimensionality
    num_tasks : List[int]
        List of number of classes for each task
    """

    def __init__(
        self,
        input_dim: int,
        num_tasks: List[int],
    ):
        super(MultiTaskClassifier, self).__init__()

        self.num_tasks = num_tasks
        self.classifiers = nn.ModuleList([
            nn.Linear(input_dim, num_classes)
            for num_classes in num_tasks
        ])

    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input features

        Returns
        -------
        predictions : List[torch.Tensor]
            List of predictions, one per task
            Each has shape (batch_size, num_classes_i)
        """
        return [classifier(x) for classifier in self.classifiers]
