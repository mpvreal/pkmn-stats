from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

import pandas as pd

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['is_legendary'], axis=1)
    y = df['is_legendary']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.6, random_state=42)
    normalize(x, axis=0, norm='max')

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('Accuracy: {}%'.format(accuracy_score(y_test, y_pred) * 100))

    fig = plt.figure(figsize=(25,20))
    _ = tree.plot_tree(model,
                          feature_names=x.columns,
                            class_names=['Not Legendary', 'Legendary'],
                            filled=True)
    fig.savefig("decision_tree.png")
