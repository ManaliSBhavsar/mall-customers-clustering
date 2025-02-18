# Mall Customers Clustering

This project uses K-Means clustering to analyze mall customers based on two features: **Annual Income** and **Spending Score (1-100)**. The goal is to cluster the customers into different groups to gain insights into their purchasing behavior.

## Project Structure

- `data/`: Contains the dataset (`Mall_Customers.csv`) used for clustering.
- `src/`: Contains the Python script for performing K-Means clustering (`kmeans_clustering.py`).
- `requirements.txt`: Lists the Python dependencies required to run the project.

## Elbow Method
The Elbow Method is used to determine the optimal number of clusters (K). The plot of error vs. the number of clusters helps identify the point where the curve begins to flatten, indicating the optimal K.

## Visualizations
Before Clustering: The initial scatter plot of data points based on Annual Income and Spending Score.

After Clustering: A scatter plot with data points color-coded by cluster and the centroids marked.
