from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

import pandas as pd

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['attack', 'defense', 'sp_attack', 'sp_defense', 'is_legendary', 'name'], axis=1)

    kmeans = KMeans(n_init=10)
    elbowv = KElbowVisualizer(kmeans, k=(2, 11))
    elbowv.fit(x)
    elbowv.show(outpath="elbow.png")

    best_k = elbowv.elbow_value_
    bestk_kmeans = KMeans(n_clusters=best_k, n_init=10)
    bestk_kmeans.fit(x)

    df['cluster'] = bestk_kmeans.labels_
    df.to_csv('pokemon_clustered.csv', index=False)
