from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

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

    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(2, 10), inertias, '-o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.xticks(range(2, 10))
    fig.savefig("kmeans.png")