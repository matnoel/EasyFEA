# Copyright (C) 2021-2025 Université Gustave Eiffel.
# This file is part of the EasyFEA project.
# EasyFEA is distributed under the terms of the GNU General Public License v3, see LICENSE.txt and CREDITS.md for more information.

"""
k_means
=======

KMeans using scikit-learn according to https://www.youtube.com/watch?v=i-gxm_ofjBo.
"""

from EasyFEA import Display, np, plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

if __name__ == "__main__":

    # https://www.youtube.com/watch?v=i-gxm_ofjBo

    Display.Clear()

    N = 1000

    K = 5

    data = make_blobs(N, 2, centers=5, cluster_std=1, random_state=101)[0]

    kmeans = KMeans(K, n_init="auto")

    clusters = kmeans.fit_predict(data)

    ax = Display.Init_Axes()
    ax.plot(*data.T, "bo")
    ax.set_title("data")

    ax_c = Display.Init_Axes()
    for k in range(K):
        idx = np.where(clusters == k)
        ax_c.plot(*data[idx].T, "o")
        ax_c.plot(*np.mean(data[idx], 0), ls="", marker="+", c="black", zorder=10)

    sil_score = []
    sse = []

    array_k = np.arange(2, 20)

    for k in array_k:

        kmeans = KMeans(k, n_init="auto")

        clusters = kmeans.fit_predict(data)

        sse.append(kmeans.inertia_)
        sil_score.append(silhouette_score(data, clusters))

    a_sse = Display.Init_Axes()
    a_sse.plot(array_k, sse)
    a_sse.set_title("sse")

    a_sil = Display.Init_Axes()
    a_sil.plot(array_k, sil_score)
    a_sil.set_title("silhouette score")

    plt.show()
