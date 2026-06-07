# Spectral Clustering

## Overview

This project demonstrates the principles of Spectral Clustering and compares its performance with the classical K-Means algorithm.

The program constructs a nearest-neighbor graph, computes the graph Laplacian, analyzes its eigenvalues and eigenvectors, and performs clustering in the spectral embedding space.

Two synthetic datasets are included to illustrate different clustering scenarios.

---

## Features

* Construction of a k-nearest neighbors graph
* Visualization of graph structure
* K-Means clustering
* Graph Laplacian computation
* Eigenvalue spectrum analysis
* Spectral clustering implementation
* Comparison of K-Means and Spectral Clustering results

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* NetworkX
* Scikit-learn

---

## Datasets

### Moons Dataset

Two interleaving crescent-shaped clusters.

This dataset demonstrates a situation where Spectral Clustering can outperform traditional K-Means due to the non-linear structure of the data.

### Friends Dataset

Three manually defined groups representing connected communities.

The goal is to identify the three underlying clusters using graph-based analysis.

---

## Algorithm Workflow

1. Load the selected dataset.
2. Construct a k-nearest neighbors graph.
3. Perform K-Means clustering.
4. Compute the Degree Matrix and Graph Laplacian.
5. Calculate eigenvalues and eigenvectors of the Laplacian matrix.
6. Visualize the eigenvalue spectrum.
7. Build a spectral embedding using the smallest non-zero eigenvectors.
8. Apply K-Means in the spectral space.
9. Compare the results with standard K-Means clustering.

---

## Installation

Install the required dependencies:

pip install numpy matplotlib networkx scikit-learn
---

## Running the Project

Execute:

python demonstrait.py
Choose one of the available modes:

friends
or

```text
moons
`

---

## Output

The program generates the following visualizations:

* Nearest-neighbor graph
* K-Means clustering result
* Graph Laplacian matrix
* Eigenvalue spectrum
* Comparison between K-Means and Spectral Clustering

Additionally, all Laplacian eigenvalues are printed to the console.

---

## Project Purpose

The purpose of this project is to provide an educational demonstration of graph-based clustering techniques and to illustrate how spectral methods can identify clusters that are difficult to separate using conventional distance-based algorithms.

---

## Author

Artem Miroshnyk

Educational project on Spectral Clustering and Graph Analysis.
