from matplotlib import pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import pandas as pd
from sklearn.preprocessing import StandardScaler

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['attack', 'defense', 'sp_attack', 'sp_defense', 'is_legendary', 'name'], axis=1)
    y = df['is_legendary']

    s_scaler = StandardScaler()
    x = pd.DataFrame(s_scaler.fit_transform(x), index=x.index, columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    clf = MLPClassifier(hidden_layer_sizes=(10, 10), random_state=1, max_iter=10000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print(clf.score(x_test, y_test))
    print('Accuracy: {}%'.format(accuracy_score(y_test, y_pred) * 100))
    print('Recall: {}%'.format(recall_score(y_test, y_pred) * 100))
    print('Precision: {}%'.format(precision_score(y_test, y_pred) * 100))
    print('F1: {}%'.format(f1_score(y_test, y_pred) * 100))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    for t in ax.texts:
        t.set_color('red')
    plt.savefig('confusion_matrix.png')
    plt.clf()
