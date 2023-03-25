from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt

import pandas as pd

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['is_legendary', 'name'], axis=1)
    y = df['is_legendary']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    normalize(x, axis=0, norm='max')

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('Accuracy: {}%'.format(accuracy_score(y_test, y_pred) * 100))

    fig = plt.figure(figsize=(25, 20))
    tree.plot_tree(model,
                   feature_names=x.columns,
                   class_names=['Not Legendary', 'Legendary'],
                   filled=True)
    fig.savefig("decision_tree.png")

    leg_pred_as_leg = []
    leg_pred_as_not_leg = []
    not_leg_pred_as_leg = []

    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test.iloc[i] == 1:
            leg_pred_as_leg.append(df.loc[x_test.index[i], 'name'])
        elif y_pred[i] == 0 and y_test.iloc[i] == 1:
            leg_pred_as_not_leg.append(df.loc[x_test.index[i], 'name'])
        elif y_pred[i] == 1 and y_test.iloc[i] == 0:
            not_leg_pred_as_leg.append(df.loc[x_test.index[i], 'name'])

    leg_df = pd.DataFrame({'Pokemon': leg_pred_as_leg, 'Predicted': 'Legendary', 'Actual': 'Legendary'})
    not_leg_df = pd.DataFrame({'Pokemon': leg_pred_as_not_leg, 'Predicted': 'Not Legendary', 'Actual': 'Legendary'})
    not_leg_pred_as_leg_df = pd.DataFrame(
        {'Pokemon': not_leg_pred_as_leg, 'Predicted': 'Legendary', 'Actual': 'Not Legendary'})
    df = pd.concat([leg_df, not_leg_df, not_leg_pred_as_leg_df])
    df.to_csv('pokemon_predictions.csv', index=False)

    cm = confusion_matrix(y_test, y_pred)
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.savefig('confusion_matrix.png')
