from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import pandas as pd

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['is_legendary'], axis=1)
    y = df['is_legendary']

    inertias = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(x)
        inertias.append(kmeans.inertia_)

    elbowv = KElbowVisualizer(kmeans, k=(2, 10))
    elbowv.fit(x)

    plt.plot(range(2, 10), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.savefig('elbow.png')



