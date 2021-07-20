# 18EC30010: Debjoy Saha
# Project Number: DC4
# Project Title: Coronavirus Data Clustering using Complete Linkage Hierarchical Clustering Technique

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D

# K-Means Model Class
class K_means:
    """ Class for K-means clustering """
    def __init__(self, k = 2, max_iter = 300):
        self.k = k
        self.max_iter = max_iter
    
    def fit(self, data, norm = True, kpp = True):
        """
        Train K-means clustering model using data
        Args:
            data(Pandas DataFrame): DataFrame with all data
            norm(Bool): If True, Perform Z-score normalisation for each attribute
            kpp(Bool): If True, Get the initial centroids using K++ algorithm
        """
        # Perform Z-score Normalisation 
        if norm:
            self.data = self.z_score_norm(data)    
        else:
            self.data = data

        # Initialise centroids using K++ algorithm 
        if not kpp:
            randc = random.sample(range(1, len(self.data)), self.k)
        else:
            randc = self.get_kpp_centroids()
        self.centroids = np.array([self.data.iloc[randc[i]] for i in range(self.k)])
        
        # Fit on Data for 20 iterations
        for iter_ in range(self.max_iter):
            self.classification = {cluster_idx: [] for cluster_idx in range(self.k)}
            for index in range(len(self.data)):
                sample = np.array(self.data.iloc[index])
                dist = [np.linalg.norm(sample - centroid) for centroid in self.centroids]
                self.classification[np.argmin(dist)].append(index)
            
            for cluster_idx in range(self.k):
                clf_data = np.array([np.array(self.data.iloc[id_]) for id_ in self.classification[cluster_idx]])
                self.centroids[cluster_idx] = np.mean(clf_data, axis = 0)
        return self.centroids
    
    def z_score_norm(self, data):
        """
        Performs Z-score normalisation of the data
        Args:
            data(Pandas DataFrame): DataFrame with all data
        """
        # Calculated normalised attribute values
        cols = list(data.columns)
        norm_dict = {}
        for col in cols:
            col_zscore = col + '_zscore'
            norm_dict[col_zscore] = (data[col] - data[col].mean())/data[col].std()
        
        # Create Normalised DataFrame 
        norm_data = pd.DataFrame(norm_dict)
        return norm_data
    
    def get_silhouette_coeff(self):
        """
        Evaluation of the clustering performance using the Silhouette Coefficient
        Args:
            None (Uses self.data)
        """
        data_np = np.array(self.data)
        neighbour_cluster = np.zeros(len(self.data))
        cluster = np.zeros(len(self.data))
        
        # Determining Cluster and Neighbour Cluster for each Data-Point
        for index in range(len(self.data)):
            sample = data_np[index, :]
            dist = [np.linalg.norm(sample - centroid) for centroid in self.centroids]
            cluster[index] = np.argmin(dist)
            dist[np.argmin(dist)] = max(dist)
            neighbour_cluster[index] = np.argmin(dist)
            
        sil_list = []
        for index in range(len(self.data)):
            sample = data_np[index, :]
            c = cluster[index]
            nc = neighbour_cluster[index]
            
            # a calculation
            a_dist = np.array([np.linalg.norm(sample - data_np[id_, :]) for id_ in self.classification[c]])
            a = np.sum(a_dist, axis = 0)/(a_dist.shape[0]-1)
            
            # b calculation
            b_dist = np.array([np.linalg.norm(sample - data_np[id_, :]) for id_ in self.classification[nc]])
            b = np.mean(b_dist, axis = 0)
                
            # Silhouette Score Calculation
            s = (b-a)/max(a, b)
            sil_list.append(s)
        
        return np.mean(sil_list)
    
    def get_kpp_centroids(self):
        """
        Get initial centroids with maximum inter-connecting distance(ref: http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) 
        Args:
            None (Uses self.data)
        """
        # Initialise the first centroid randomly
        randc = random.sample(range(1, len(self.data)), 1)

        # Initialise subsequent centroids with probability proportional to distance from closest centroid
        for kn in range(1, self.k):
            all_dist = []
            for index in range(len(self.data)):
                sample = np.array(self.data.iloc[index])
                dist = np.min([np.linalg.norm(sample - np.array(self.data.iloc[id_])) for id_ in randc])
                all_dist.append(dist)
            probabilities = all_dist/np.sum(all_dist)
            newrandc = np.random.choice(range(len(probabilities)), p=probabilities)
            randc.append(newrandc)
        return randc
            
    def plot(self, fname = None, clustered = True):
        """
        Visualize clustered Data
        Args:
            fname(str): Save in fname
            clustered(bool): If false, show unclustered data. Error if clustered = True and K-means untrained.
        """

        # Plot Data and store in png file
        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, projection='3d')
        
        if clustered:
            classification_val = self.classification
            for cluster in range(self.k):
                data_cluster = classification_val[cluster]
                xs = [np.array(self.data.iloc[i])[0] for i in data_cluster]
                ys = [np.array(self.data.iloc[i])[1] for i in data_cluster]
                zs = [np.array(self.data.iloc[i])[2] for i in data_cluster]
                ax.scatter(xs, ys, zs)
        else:
            xs = [np.array(self.data.iloc[i])[0] for i in range(len(self.data))]
            ys = [np.array(self.data.iloc[i])[1] for i in range(len(self.data))]
            zs = [np.array(self.data.iloc[i])[2] for i in range(len(self.data))]
            ax.scatter(xs, ys, zs)
        
        ax.set_xlabel(list(self.data)[0])
        ax.set_ylabel(list(self.data)[1])
        ax.set_zlabel(list(self.data)[2])
        fig.suptitle(fname.strip('.png'), fontsize=20)

        plt.savefig(fname)
        
    def savetxt(self, fname = None):
        """
        Store Cluster information for evaluation
        Args:
            fname(str): Save in fname
        """
        # Save cluster information in txt file
        final_clusters = {}
        for c in self.classification.values():
            sorted_c = np.sort(c)
            key_c = min(sorted_c)
            final_clusters[key_c] = sorted_c

        with open (fname, 'w') as fo:
            for k in np.sort(list(final_clusters.keys())):
                fo.write(','.join(str(i) for i in final_clusters[k]))
                fo.write('\n')

# Hierarchical clustering Class
class complete_linkage_clustering:
    """ Class for Hierarchical clustering using Complete Linkage Strategy"""
    def __init__(self, k = 2):
        self.k = k
        
    def get_distance_matrix(self, data):
        """
        Returns the initial distance matrix
        Args:
            data(Pandas DataFrame): DataFrame with all data
        """
        N = len(data)
        org_distance_matrix = np.zeros([N, N])
        data_arr = np.array(data)
        
        # Populate the Distance Matrix
        for i in range(len(data)):
            for j in range(i, len(data)):
                isample = np.array(data_arr[i])
                jsample = np.array(data_arr[j])
                dist = np.linalg.norm(isample - jsample)
                org_distance_matrix[i, j] = dist
                org_distance_matrix[j, i] = dist
        
        return org_distance_matrix
    
    def z_score_norm(self, data):
        """
        Performs Z-score normalisation of the data
        Args:
            data(Pandas DataFrame): DataFrame with all data
        """
        # Calculated normalised attribute values
        cols = list(data.columns)
        norm_dict = {}
        for col in cols:
            col_zscore = col + '_zscore'
            norm_dict[col_zscore] = (data[col] - data[col].mean())/data[col].std()
        
        # Create Normalised DataFrame 
        norm_data = pd.DataFrame(norm_dict)
        return norm_data
    
    def get_min(self, arr):
        """
        Returns the indices of the clusters having minimum distance
        Args:
            arr(Numpy 2D array): Distance Matrix
        """
        # Get cluster pair having minimum distance
        argmin = np.array([0, 1])
        minv = arr[0, 1]
        for i in range(arr.shape[0]):
            for j in range(i+1, arr.shape[1]):
                if minv > arr[i, j]:
                    minv = arr[i, j]
                    argmin = np.array([i, j])
        return argmin

    def fit(self, data, norm = True):
        """
        Train Complete Linkage clustering model using data
        Args:
            data(Pandas DataFrame): DataFrame with all data
            norm(Bool): If True, Perform Z-score normalisation for each attribute
        """
        if norm:
            self.data = self.z_score_norm(data)    
        else:
            self.data = data
        
        # Get Distance Matrix
        distance_matrix = self.get_distance_matrix(self.data)
        
        # Initialise List of Clusters
        clusters = [[i] for i in range(len(self.data))]
        
        # Iterate and combine clusters at each iteration until no. of clusters = k
        while len(clusters) > self.k:
            [i, j] = self.get_min(distance_matrix)
            
            # Extract the distance values for the aggregated cluster according to complete linkage algorithm
            dist1 = distance_matrix[i, :]
            dist2 = distance_matrix[j, :]
            distmax = np.maximum(dist1, dist2)
            distmax = np.delete(distmax, [i, j], 0)
            distmax = np.append(distmax, 0)
            
            # Replace individual cluster with aggregated cluster in Distance Matrix
            distance_matrix = np.delete(distance_matrix, [i, j], 0)
            distance_matrix = np.delete(distance_matrix, [i, j], 1)
            distance_matrix = np.r_[distance_matrix, np.reshape(distmax[:-1], [1, -1])]
            distance_matrix = np.c_[distance_matrix, np.reshape(distmax, [-1, 1])]
            
            # Add cluster to cluster list
            ci = clusters[i]
            cj = clusters[j]
            cf = ci + cj
            clusters.remove(ci)
            clusters.remove(cj)
            clusters.append(cf)
        
        self.distance_matrix = distance_matrix
        self.clusters = clusters
        return clusters

    def get_silhouette_coeff(self):
        """
        Evaluation of the clustering performance using the Silhouette Coefficient
        Args:
            None (Uses self.data)
        """
        cluster = np.zeros(len(self.data))
        neighbour_cluster = np.zeros(len(self.data))
        
        # Determining Cluster and Neighbour Cluster for each Data-Point
        for i in range(len(self.clusters)):
            for data_pt in self.clusters[i]:
                cluster[data_pt] = i  
                dist = np.copy(self.distance_matrix[i])
                dist[i] = max(dist)
                neighbour_cluster[data_pt] = np.argmin(dist)
        
        sil_list = []
        for index in range(len(self.data)):
            sample = np.array(self.data.iloc[index])
            c = cluster[index]
            nc = neighbour_cluster[index]
            
            # a calculation
            a_dist = np.array([np.linalg.norm(sample - np.array(self.data.iloc[int(id_)])) for id_ in self.clusters[int(c)]])
            a = np.sum(a_dist, axis = 0)/(a_dist.shape[0]-1)
            
            # b calculation
            b_dist = np.array([np.linalg.norm(sample - np.array(self.data.iloc[int(id_)])) for id_ in self.clusters[int(nc)]])
            b = np.mean(b_dist, axis = 0)
                
            # Silhouette Calculation
            s = (b-a)/max(a, b)
            sil_list.append(s)
        
        return np.mean(sil_list)
    
    def plot(self, fname = None, clustered = True):
        """
        Visualize clustered Data
        Args:
            fname(str): Save in fname
            clustered(bool): If false, show unclustered data. Error if clustered = True and untrained.
        """
        # Plot Data and store in png file
        fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        ax = fig.add_subplot(111, projection='3d')
        
        if clustered:
            for data_cluster in self.clusters:
                xs = [np.array(self.data.iloc[i])[0] for i in data_cluster]
                ys = [np.array(self.data.iloc[i])[1] for i in data_cluster]
                zs = [np.array(self.data.iloc[i])[2] for i in data_cluster]
                ax.scatter(xs, ys, zs)
        else:
            xs = [np.array(self.data.iloc[i])[0] for i in range(len(self.data))]
            ys = [np.array(self.data.iloc[i])[1] for i in range(len(self.data))]
            zs = [np.array(self.data.iloc[i])[2] for i in range(len(self.data))]
            ax.scatter(xs, ys, zs)
        
        ax.set_xlabel(list(self.data)[0])
        ax.set_ylabel(list(self.data)[1])
        ax.set_zlabel(list(self.data)[2])
        fig.suptitle(fname.strip('.png'), fontsize=20)

        plt.savefig(fname)
    
    def savetxt(self, fname = None):
        """
        Store Cluster information for evaluation
        Args:
            fname(str): Save in fname
        """
        # Save cluster data in txt file
        final_clusters = {}
        for c in self.clusters:
            sorted_c = np.sort(c)
            key_c = min(sorted_c)
            final_clusters[key_c] = sorted_c

        with open (fname, 'w') as fo:
            for k in np.sort(list(final_clusters.keys())):
                fo.write(','.join(str(i) for i in final_clusters[k]))
                fo.write('\n')


# Jaccard Similarity Score Comparison
def get_mapping(centroids1, centroids2, k):
    """
    Find Mapping between Cluster centroids
    Args:
        centroids1 (lists of points list): K-means centroids
        centroids2 (lists of points list): hierarchicalarchical clustering centroids
        k: No. of clusters
    """
    h_done = {i: False for i in range(k)}
    mapping = {}

    # Greedy Mapping - For each i in K-means centroid, determine closest Hierachical cluster j 
    for i in range(k):
        min_dist = np.inf
        argmin_dist = 0

        for j in range(k):
            if not h_done[j]:
                dist = np.linalg.norm(centroids1[i] - centroids2[j])
                if dist < min_dist:
                    min_dist = dist
                    argmin_dist = j

        mapping[i] = argmin_dist
        h_done[argmin_dist] = True
    return mapping

def jaccard_similarity(cluster1, cluster2):
    """
    Find Jaccard Similarity metric of two cluster
    Args:
        clusters1 (lists of indices): K-means clusters
        clusters2 (lists of indices): Hierarchical clusters
        k: No. of clusters
    """
    intersection = len(list(set(cluster1).intersection(cluster2)))
    union = (len(cluster1) + len(cluster2)) - intersection
    return float(intersection) / union

def get_similarity_clusters(kmeans_clf, hierarchical_clf):
    """
    Find Jaccard Similarity metric for mapping of trained K-means and Hierarchical Classifier
    Args:
        kmeans_clf: K-means trained classifier
        hierarchical_clf: Hierarchical Trained classifier
    """
    # Get each cluster
    h_clusters = hierarchical_clf.clusters
    k_clusters = [kmeans_clf.classification[i] for i in kmeans_clf.classification.keys()]
    assert len(h_clusters) == len(k_clusters)
    k = len(k_clusters)

    # Get Cluster Centroid for mapping
    h_centroids = np.array([np.mean(np.array([hierarchical_clf.data.iloc[id_] for id_ in h]), axis = 0) for h in h_clusters])
    k_centroids = kmeans_clf.centroids
    
    # Get an 1-to-1 mapping between clusters
    mapping = get_mapping(k_centroids, h_centroids, k)
    
    # For each mapping pair, get the jaccard similarity
    for (k, h) in mapping.items():
        score = jaccard_similarity(k_clusters[k], h_clusters[h])
        print("For Mapping {} --> {} : Jaccard Similariy = {}".format(k, h, round(score,3)))

def main():
    # Reading Data
    csv_data_file = 'COVID_4_unlabelled.csv'
    data = pd.read_csv(csv_data_file, index_col=0)
    
    """
        K-Means Clustering
    """
    znorm = True              # If True, using Z-Score Normalisation
    use_kpp = True            # If True, Initialise centroids using K-means++ algorithm
    n_iters = 20              # No. of iterations to run K-means algorithm
    k_list = [3, 4, 5, 6]     # Run for k no. of Clusters
    sil_score_list = []       # For storing silhouette scores for different k-values
    if znorm:
        print("Z-score Normalisation of Data Enabled")
        print("-----------------------------------------------")
    
    print("K-Means Clustering Part-")
    for k in k_list:
        print("K = {}".format(k))

        # Initialise K-Means instance
        kmeans_clf = K_means(k = k, max_iter = n_iters)

        # Fit K-means algorithm on data
        print("Training K-means algorithm on Data ...")
        centroids = kmeans_clf.fit(data, norm = znorm, kpp = use_kpp)

        # Compute te overall Silhouette score and store
        print("Computing the overall Silhouette Score ...")
        sil_score = kmeans_clf.get_silhouette_coeff()
        print("Silhouette Coefficitent for the cluster = {}".format(sil_score))
        sil_score_list.append(sil_score)

        # Plotting the Clustered Data
        plotfig = "Kmeans_k_{}.png".format(k)
        print("Generated clusters plot saved in {}".format(plotfig))
        kmeans_clf.plot(fname = plotfig)
        print("-----------------------------------------------")

    # Get Best Performance
    k = k_list[np.argmax(sil_score_list)]
    print("Best Performing K = {}".format(k))
    print("-----------------------------------------------")
    
    # Retrain on Best K value obtained
    print("Best K = {}".format(k))
    kmeans_clf = K_means(k = k, max_iter = n_iters)

    print("Training K-means algorithm on Data for best k ...")
    centroids = kmeans_clf.fit(data, norm = znorm, kpp = use_kpp)
    
    # Compute te overall Silhouette score
    print("Computing the overall Silhouette Score ...")
    sil_score = kmeans_clf.get_silhouette_coeff()
    print("Silhouette Coefficitent for the clustering = {}".format(sil_score))

    # Plotting the Clustered Data
    plotfig = "Kmeans_best_k_{}.png".format(k)
    fsave = "kmeans.txt"
    print("Generated clusters plot saved in {}".format(plotfig))
    print("Clustered data-points saved in {}".format(fsave))
    kmeans_clf.plot(fname = plotfig)
    kmeans_clf.savetxt(fname = fsave)
    print("-----------------------------------------------")
    
    """
        Hierarchical Clustering with complete linkage strategy
    """
    print("Hierarchical Clustering Part-")
    # Best K calculated from K-means Algo
    print("K = {}".format(k))
    hierarchical_clf = complete_linkage_clustering(k = k)
    
    # Fit hierarchical classifier on data
    print("Training Hierarchical Classifier algorithm on Data ...")
    clusters = hierarchical_clf.fit(data, norm = znorm)
    
    # Compute te overall Silhouette score
    print("Computing the overall Silhouette Score ...")
    score = hierarchical_clf.get_silhouette_coeff()
    print("Silhouette Coefficitent for the clustering = {}".format(sil_score))

    # Plotting the Clustered Data
    plotfig = "Hierarchical_best_k_{}.png".format(k)
    fsave = "agglomerative.txt"
    print("Generated clusters plot saved in {}".format(plotfig))
    print("Clustered data-points saved in {}".format(fsave))
    hierarchical_clf.plot(fname = plotfig)
    hierarchical_clf.savetxt(fname = fsave)
    print("-----------------------------------------------")

    """
        Jaccard Similarity Score Comparison
    """
    print("Jaccard Similarity for different clusters - ")
    get_similarity_clusters(kmeans_clf, hierarchical_clf)

# Run main() function
main()