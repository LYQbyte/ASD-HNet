import numpy as np
import torch.nn as nn
import torch
from brainspace.gradient import GradientMaps
import math
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def gradient_calcu(fc):
    """

    :param fc:  input the functional connection
    :return:  return the functional gradient matrices
    """
    # and fit to the data
    gradient = []
    for i in range(len(fc)):
        gm1 = GradientMaps(n_components=2, approach='dm', kernel='cosine', random_state=1)
        gm1.fit(fc[i])
        gradient.append(gm1.gradients_)

    gradient = np.array(gradient)
    return gradient


'''
app : {'dm', 'le', 'pca'} or object
        Embedding approach. If object. it can be an instance of PCAMaps,
        LaplacianEigenmaps or DiffusionMaps.
    kernel : {'pearson', 'spearman', 'cosine', 'normalized_angle', 'gaussian'}
        or None, optional.
        Kernel function to build the affinity matrix.'''


def data_norm(data):
    max_value = data.max()
    min_value = data.min()
    ranges = max_value - min_value
    data_normed = (data - min_value) / ranges
    return data_normed


def gradient_eucli(gra_array, gra_dim=2, dim=116):
    gra_like_mask = []
    for i in range(len(gra_array)):
        gra_like_mask_i = np.zeros((dim, dim))
        for j in range(dim):
            for z in range(dim):
                gra_like_mask_i[j][z] = np.sqrt(
                    sum(np.power((gra_array[i, j, 0:gra_dim] - gra_array[i, z, 0:gra_dim]), 2)))
        gra_like_mask.append(gra_like_mask_i)
    gra_like_mask = np.array(gra_like_mask)  # Calculate the distance between ROI i and ROI j
    # reversing and normalizing
    for i in range(len(gra_like_mask)):
        max_i = gra_like_mask[i].max()
        min_i = gra_like_mask[i].min()
        gra_like_mask[i] = data_norm(max_i - gra_like_mask[i])
    # now, the value represent the corr between ROI i and j, value bigger the corr bigger.
    return gra_like_mask


def kmeans(gra_data, k=7, mask=False):
    # gra_data shape: 507, 116, 2
    gra_clu = []
    if mask == False:
        for i in range(len(gra_data)):
            cluster_i = np.zeros((k, 116))
            # gmm = GMM(n_components=4).fit(X)  # 指定聚类中心个数为4
            # labels = gmm.predict(X)
            y_pred = KMeans(n_clusters=k, init="k-means++", random_state=1).fit_predict(gra_data[i])
            for j in range(len(y_pred)):
                cluster_i[y_pred[j]][j] = 1.0
            gra_clu.append(cluster_i)
        # plt.scatter(gra_data[i, :, 0], gra_data[i, :, 1], c=y_pred)
        # plt.show()
    else:
        for i in range(len(gra_data)):
            cluster_i = np.zeros((116, 116))
            y_pred = KMeans(n_clusters=k, init="k-means++", random_state=1).fit_predict(gra_data[i])
            for j in range(len(y_pred)):
                for k in range(len(y_pred)):
                    if y_pred[j]==y_pred[k]:
                        cluster_i[j][k]=1.0
            gra_clu.append(cluster_i)
    gra_clu = np.array(gra_clu)
    return gra_clu


def consensus_clustering(X, n_clusters, n_kmeans=3):
    n_samples = X.shape[0]
    consensus_matrix = np.zeros((n_samples, n_samples))

    for _ in range(n_kmeans):
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=15)
        labels = kmeans.fit_predict(X)
        for i in range(n_samples):
            for j in range(n_samples):
                if labels[i] == labels[j]:
                    consensus_matrix[i, j] += 1

    # Normalize the consensus matrix
    consensus_matrix /= n_kmeans

    # Perform hierarchical clustering on the consensus matrix
    condensed_dist = 1 - consensus_matrix  # Convert to distance matrix
    condensed_dist = squareform(condensed_dist)  # Convert to condensed form
    linkage_matrix = linkage(condensed_dist, method='average')

    # Extract final labels
    final_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')-1

    return final_labels


def setup_seed(seed):
    """ 该方法用于固定随机数

    Args:
        seed: 随机种子数

    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def gradient_avg_mask(Fc_all, k, mask=False):
    # Fc_all shape: num 116 116
    # 聚类为k个簇
    avg_Fc = np.sum(Fc_all, axis=0)/len(Fc_all)
    gm1 = GradientMaps(n_components=2,  approach='dm', kernel='cosine', random_state=42)
    gm1.fit(avg_Fc)
    gradient = gm1.gradients_
    y_pred = KMeans(n_clusters=k, init="k-means++", random_state=1).fit_predict(gradient)
    # y_pred = consensus_clustering(gradient, n_clusters=k, n_kmeans=5)
    # gmm = GMM(n_components=k, random_state=1).fit(gradient)  # 指定聚类中心个数为4
    # y_pred = gmm.predict(gradient)
    if mask==False:
        cluster = np.zeros((k, 116))
        for j in range(len(y_pred)):
            cluster[y_pred[j]][j] = 1.0
    else:
        cluster = np.zeros((116, 116))
        for j in range(len(y_pred)):
            for k in range(len(y_pred)):
                if y_pred[j]==y_pred[k]:
                    cluster[j][k]=1.0

    return cluster


if __name__ == "__main__":
    setup_seed(1)
    white_fc = np.load("./Data/ABIDE/FC_aal.npy")
    white_gra = gradient_calcu(white_fc)
    gra_cluster_m = kmeans(gra_data=white_gra, k=15)

    np.save("./Data/ABIDE/aal_gra_cluster.npy", gra_cluster_m)

    # np.save("./Data/class_4/aal_fgra.npy", white_gra)
    # white_gra_mask = gradient_eucli(white_gra, gra_dim=2, dim=48)
    # np.save("./Data/class_4/white_gra_mask.npy", white_gra_mask)
