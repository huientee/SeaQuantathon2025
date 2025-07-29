#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 10:37:25 2025

@author: teehuien
"""

import torch
import torch.nn as nn
import pennylane as qml
import numpy as np
import math
from typing import Optional, Tuple

class QuantumLayer:
    """
    Quantum computing layer for marine heatwave prediction
    """
    
    def __init__(self, n_qubits: int = 8, n_layers: int = 2):
        """
        Initialize quantum layer
        
        Args:
            n_qubits: Number of qubits
            n_layers: Number of quantum layers
        """
            
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Create quantum device
        self.dev = qml.device("default.qubit", wires=n_qubits)
        
        # Define quantum circuit
        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def quantum_circuit(inputs, weights):
            """
            Quantum circuit for feature processing
            
            Args:
                inputs: Input features (normalized to [-π, π])
                weights: Quantum gate parameters
            """
            # Input encoding - angle embedding
            # Normalize inputs to avoid overflow
            inputs_normalized = self._normalize_inputs(inputs)
            qml.AngleEmbedding(inputs_normalized, wires=range(self.n_qubits), rotation='Y')
            
            # Variational layers
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
            # Measurement - expectation values of Pauli-Z
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.quantum_circuit = quantum_circuit
        
        # Weight shape for the quantum circuit
        self.weight_shape = {"weights": (n_layers, n_qubits, 3)}
    
    def _normalize_inputs(self, inputs):
        """Normalize inputs to prevent quantum circuit overflow"""
        # Clamp to reasonable range and scale to [-π, π]
        inputs_clamped = torch.clamp(inputs, -10, 10)
        max_val = torch.max(torch.abs(inputs_clamped))
        if max_val > 0:
            inputs_normalized = (inputs_clamped / max_val) * math.pi
        else:
            inputs_normalized = inputs_clamped
        return inputs_normalized
    
    def create_torch_layer(self):
        """Create PyTorch-compatible quantum layer"""
        return qml.qnn.TorchLayer(self.quantum_circuit, self.weight_shape)

class PhysicsInformedQNN(nn.Module):
    """
    Physics-informed Quantum Neural Network for marine heatwave prediction
    Combines quantum feature processing with classical neural networks
    """
    
    def __init__(self, input_dim: int, n_qubits: int = 8, hidden_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.hidden_dim = hidden_dim
        
        # Feature preprocessing - reduce dimensionality to quantum layer size
        self.feature_reducer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, n_qubits),
            nn.Tanh()  # Output in [-1, 1] range for quantum processing
        )
        
        # Quantum layer
        self.quantum_layer = QuantumLayer(n_qubits).create_torch_layer()
        
        # Post-quantum classical processing
        self.post_quantum = nn.Sequential(
            nn.Linear(n_qubits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Physics-informed constraints
        self.temperature_constraint = nn.Sigmoid()  # Constrain reasonable temperature range
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize neural network weights"""
        for module in [self.feature_reducer, self.post_quantum]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """Forward pass through the quantum-enhanced model"""
        # Feature preprocessing
        x_reduced = self.feature_reducer(x)
        
        # Quantum processing
        x_quantum = self.quantum_layer(x_reduced)
        
        # Classical post-processing
        output = self.post_quantum(x_quantum)
        
        return output

class EnsembleQNN(nn.Module):
    """
    Ensemble of quantum neural networks for improved prediction accuracy
    """
    
    def __init__(self, input_dim: int, n_models: int = 3, n_qubits: int = 8):
        super().__init__()
        
        self.n_models = n_models
        
        # Create ensemble of QNNs with different architectures
        self.models = nn.ModuleList([
            PhysicsInformedQNN(input_dim, n_qubits + i, 64 + i*16) 
            for i in range(n_models)
        ])
        
        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(torch.ones(n_models) / n_models)
        
    def forward(self, x):
        """Forward pass through ensemble"""
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Weighted ensemble
        predictions = torch.stack(predictions, dim=-1)
        weights = torch.softmax(self.ensemble_weights, dim=0)
        
        ensemble_pred = torch.sum(predictions * weights, dim=-1, keepdim=True)
        
        return ensemble_pred

class QuantumModelTrainer:
    """
    Training utilities for quantum neural networks
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        
        # Optimizer with lower learning rate for quantum layers
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            patience=10, 
            factor=0.5,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def train_epoch(self, dataloader, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output.squeeze(), target)
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def validate(self, dataloader) -> Tuple[float, np.ndarray, np.ndarray]:
        """Validate model performance"""
        self.model.eval()
        total_loss = 0.0
        predictions = []
        targets = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output.squeeze(), target)
                
                total_loss += loss.item()
                predictions.extend(output.squeeze().cpu().numpy())
                targets.extend(target.cpu().numpy())
        
        return total_loss / len(dataloader), np.array(predictions), np.array(targets)