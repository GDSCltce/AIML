## What is unsupervised learning?
Unsupervised learning, also known as unsupervised machine learning, uses machine learning algorithms to analyze and cluster unlabeled datasets. These algorithms discover hidden patterns or data groupings without the need for human intervention.
Its ability to discover similarities and differences in information make it the ideal solution for exploratory data analysis, cross-selling strategies, customer segmentation, and image recognition.

## Common Unsupervised Learning Approaches

Unsupervised learning models serve three primary tasks: clustering, association, and dimensionality reduction. The following outlines each learning method and highlights common algorithms and approaches for their effective implementation.

## Clustering

Clustering is a data mining technique that groups unlabeled data based on their similarities or differences. Clustering algorithms process raw, unclassified data objects into groups represented by structures or patterns in the information. Clustering algorithms can be categorized into exclusive, overlapping, hierarchical, and probabilistic types.

### Exclusive and Overlapping Clustering

**Exclusive clustering** involves grouping data points into only one cluster, also known as "hard" clustering. The K-means clustering algorithm is an example of exclusive clustering.

* K-means clustering: Data points are assigned to K groups, where K represents the number of clusters based on the distance from each group’s centroid. The data points closest to a given centroid will be clustered under the same category. K-means clustering is commonly used in market segmentation, document clustering, image segmentation, and image compression.

**Overlapping clusters** differ from exclusive clustering as they allow data points to belong to multiple clusters with separate degrees of membership. "Soft" or fuzzy k-means clustering is an example of overlapping clustering.

### Hierarchical Clustering

**Hierarchical clustering**, also known as hierarchical cluster analysis (HCA), is an unsupervised clustering algorithm categorized as agglomerative or divisive.

#### Agglomerative Clustering

Agglomerative clustering is considered a "bottoms-up approach." Data points are initially isolated as separate groupings and are then iteratively merged based on similarity until one cluster is achieved. Four commonly used methods to measure similarity are:

* Ward’s linkage: Defines the distance between two clusters by the increase in the sum of squared after the clusters are merged.
* Average linkage: Defined by the mean distance between two points in each cluster.
* Complete (or maximum) linkage: Defined by the maximum distance between two points in each cluster.
* Single (or minimum) linkage: Defined by the minimum distance between two points in each cluster.

Euclidean distance is the most common metric used to calculate these distances, although other metrics, such as Manhattan distance, are also cited in clustering literature.

#### Divisive Clustering

Divisive clustering, the opposite of agglomerative clustering, takes a "top-down" approach. In this case, a single data cluster is divided based on the differences between data points. While divisive clustering is not commonly used, it is worth noting in the context of hierarchical clustering. These clustering processes are usually visualized using a dendrogram, a tree-like diagram documenting the merging or splitting of data points at each iteration.




