# MNIST Dimensionality Reduction with SVD and Logistic Regression

## Overview
This project applies **Singular Value Decomposition (SVD)** as a preprocessing step on the MNIST dataset and explores how dimensionality reduction impacts the performance and efficiency of a logistic regression classifier. By experimenting with different numbers of SVD components, we analyze trade-offs between training time and model accuracy.

---

## Project Goals
1. **Dimensionality Reduction**: Implement SVD from scratch to preprocess the MNIST dataset.
2. **Model Comparison**: Train logistic regression models on the original and reduced datasets and evaluate their accuracy and training time.
3. **Exploration and Visualization**: 
   - Examine the impact of varying SVD components on model performance.
   - Visualize the top 5 singular vectors as 28x28 images.
   - Identify the optimal number of SVD components.

---

## Key Features

### **1. Singular Value Decomposition (SVD)**
The SVD algorithm is implemented from scratch using only NumPy functions. It decomposes the data into three matrices: \( U, \Sigma, V^T \), which are then used to reduce the dimensionality of the dataset. 

### **2. Logistic Regression Classifier**
A logistic regression model is trained on:
- **Original Data**: Full-dimensional MNIST dataset.
- **SVD-Reduced Data**: MNIST data reduced to varying numbers of components.

### **3. Performance Metrics**
- **Accuracy**: How well the model predicts test labels.
- **Training Time**: Time taken to train the logistic regression model.

### **4. Visualizations**
- **Accuracy vs. Number of Components**: Line plot showing how dimensionality impacts model accuracy.
- **Training Time vs. Number of Components**: Line plot showing how dimensionality affects training efficiency.
- **Top 5 Singular Vectors**: Visualized as 28x28 grayscale images to interpret the SVD-reduced features.

## Setup and Usage

### **1. Clone the Repository**
```bash
git clone <repository_url>
cd MNIST-SVD
```

### **2. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **3. Run the Project**
Use the `Makefile` to execute key steps:
- **Run the Analysis**:
  ```bash
  make run
  ```
- **Visualize Results**:
  ```bash
  make visualize
  ```

### **4. Key Scripts**
- **`src/svd.py`**: Implements the SVD algorithm and dimensionality reduction.
- **`src/logistic_regression.py`**: Trains the logistic regression model on both full and reduced datasets.
- **`src/visualization.py`**: Generates plots and visualizes singular vectors.

---

## Conclusion
This project highlights the power of SVD in reducing the dimensionality of large datasets like MNIST. By balancing accuracy and efficiency, dimensionality reduction becomes an essential tool for scalable machine learning applications. The project provides a solid foundation for exploring further improvements in preprocessing and model optimization.

--- 

**Next Steps:**
- Experiment with other dimensionality reduction techniques like PCA.
- Extend the project to include deep learning-based classifiers for comparison.
- Integrate hyperparameter tuning for logistic regression to optimize performance.
