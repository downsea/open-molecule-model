# PanGu Drug Model Architecture (v1)

This document outlines the architecture of the PanGu Drug Model, as described in the paper "PanGu Drug Model: Learn a Molecule Like a Human".

## Overview

The PanGu Drug Model is a conditional variational autoencoder (cVAE) that translates between 2D molecular graphs and their corresponding SELFIES string representations. This graph-to-sequence approach allows the model to learn a deep understanding of molecular structures and properties.

## Encoder

The encoder is a graph transformer model that maps a 2D molecular graph to a latent vector representation.

*   **Input**: A 2D molecular graph, where nodes are atoms and edges are bonds.
*   **Architecture**:
    *   10 transformer convolution layers.
    *   512-dimensional hidden units.
    *   6 attention heads.
*   **Output**: A latent matrix of size 8x256, which is formed by concatenating the graph representations from layers 1, 2, 3, 4, 5, 6, 8, and 10.

## Decoder

The decoder is a transformer-based sequence model that generates a SELFIES string from the latent representation and a condition vector.

*   **Input**:
    *   The latent matrix from the encoder.
    *   A condition vector representing desired molecular properties (e.g., molecular weight, logP).
*   **Architecture**:
    *   6 decoding layers.
    *   The first layer is an absolute positional encoding layer.
    *   The following 5 layers are relative positional encoding layers.
    *   512-dimensional hidden units.
    *   8 attention heads.
*   **Output**: A SELFIES string representing the generated molecule.

## Training

The model is trained as a cVAE using the following setup:

*   **Objective Function**: The evidence lower bound (ELBO), which consists of a reconstruction loss and a KL divergence term.
*   **Regularization**: The KL divergence term is weighted by a regularization coefficient β = 0.001.
*   **Optimizer**: Adam optimizer.
*   **Learning Rate Scheduler**: Polynomial learning rate scheduler.
*   **Pre-training Data**: 1.7 billion small molecules from the ZINC20, DrugSpaceX, and UniChem databases.