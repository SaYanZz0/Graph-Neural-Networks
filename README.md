# Graph Neural Networks: GCN and GAT

This repository contains a Jupyter Notebook demonstrating the implementation of Graph Convolutional Networks (GCN) and Graph Attention Networks (GAT) for node/edge/graph classification tasks. The notebook provides a step-by-step guide on how to build, train, and evaluate these models using the Cora dataset, a popular benchmark for node classification.

## Notebook Overview

The notebook, [GCN-GAT-nodeLevelClassification.ipynb](GCN-GAT-nodeLevelClassification.ipynb), is structured as follows:

1.  **Introduction and Setup:**
    *   Briefly introduces Graph Neural Networks (GNNs) and their application to node classification.
    *   Imports necessary libraries, including PyTorch, PyTorch Geometric, and NetworkX.
    *   Checks for GPU availability and sets the device accordingly.

2.  **Dataset Loading and Exploration:**
    *   Loads the Cora dataset using PyTorch Geometric's `Planetoid` class.
    *   Explores the dataset's characteristics, such as the number of nodes, edges, features, and classes.
    *   Visualizes a subgraph of the Cora citation network to provide a visual understanding of the graph structure.

3.  **Graph Convolutional Network (GCN):**
    *   **Model Definition:** Defines the GCN model using PyTorch Geometric's `GCNConv` layer. The model architecture consists of two GCN layers followed by a log softmax activation function for classification.
    *   **Training:**
        *   Initializes the GCN model, optimizer (Adam), and loss function (Negative Log Likelihood Loss).
        *   Defines training and testing functions to train the model and evaluate its performance.
        *   Trains the GCN model for a specified number of epochs, monitoring the training loss and accuracy.
    *   **Evaluation:** Evaluates the trained GCN model on the test set, reporting the test accuracy.

4.  **Graph Attention Network (GAT):**
    *   **Model Definition:** Defines the GAT model using PyTorch Geometric's `GATConv` layer. Similar to the GCN model, it has two GAT layers and a log softmax output layer. The GAT model utilizes attention mechanisms to weigh the importance of neighboring nodes.
    *   **Training:**
        *   Initializes the GAT model, optimizer (Adam), and loss function (Negative Log Likelihood Loss).
        *   Trains the GAT model for a specified number of epochs, monitoring the training loss and accuracy.
    *   **Evaluation:** Evaluates the trained GAT model on the test set, reporting the test accuracy.

5.  **Results Visualization:**
    *   Creates plots to visualize the training process for both GCN and GAT models. This includes plots of:
        *   Training loss over epochs.
        *   Training accuracy over epochs.
        *   Test accuracy over epochs.
    *   Compares the performance of GCN and GAT models based on the visualization and numerical results.

## Key Concepts and Libraries

*   **Graph Neural Networks (GNNs):** Neural networks designed to operate on graph-structured data.
*   **Graph Convolutional Networks (GCNs):** A type of GNN that applies convolutional operations on the graph, aggregating information from neighboring nodes.
*   **Graph Attention Networks (GATs):** A type of GNN that uses attention mechanisms to weigh the importance of neighboring nodes during aggregation.
*   **PyTorch:** A popular deep learning framework used for building and training neural networks.
*   **PyTorch Geometric:** A library built on top of PyTorch for deep learning on graphs.
*   **NetworkX:** A Python library for creating, manipulating, and studying graphs.
*   **Cora Dataset:** A citation network dataset where nodes represent scientific papers and edges represent citations. The task is to classify papers into different research areas based on their content and citation links.

## Running the Notebook

To run the notebook, you will need to have the following libraries installed:

*   PyTorch
*   PyTorch Geometric
*   NetworkX
*   Matplotlib
*   NumPy

You can install these libraries using pip:

```bash
pip install torch torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.13.0+cu117.html
pip install networkx
pip install matplotlib
pip install numpy
```

## References

*   Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907.
*   Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. arXiv preprint arXiv:1710.10903.
*   PyTorch Geometric Documentation: https://pytorch-geometric.readthedocs.io/en/latest/
