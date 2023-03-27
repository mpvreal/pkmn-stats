from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.preprocessing import normalize, StandardScaler, RobustScaler, MinMaxScaler

from mlxtend.plotting import plot_confusion_matrix

import matplotlib.pyplot as plt

import pandas as pd

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['is_legendary', 'name'], axis=1)
    y = df['is_legendary']

    # Scaler 1
    # s_scaler = StandardScaler()
    # x = pd.DataFrame(s_scaler.fit_transform(x), index=x.index, columns=x.columns)
    # Normalizer
    # x = pd.DataFrame(normalize(x), index=x.index, columns=x.columns)
    # Scaler 2
    # r_scaler = RobustScaler()
    # x = pd.DataFrame(r_scaler.fit_transform(x), index=x.index, columns=x.columns)
    # MinMaxScaler
    # mm_scaler = MinMaxScaler()
    # x = pd.DataFrame(mm_scaler.fit_transform(x), index=x.index, columns=x.columns)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    model = DecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print('Accuracy: {}%'.format(accuracy_score(y_test, y_pred) * 100))

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=confusion_matrix(y_test, y_pred), figsize=(6, 6), cmap=plt.cm.Greens)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    for t in ax.texts:
        t.set_color('red')
    plt.savefig('confusion_matrix.png')
    plt.clf()

    fig = plt.figure(figsize=(25, 20))
    tree.plot_tree(model,
                   feature_names=x.columns,
                   class_names=['Not Legendary', 'Legendary'],
                   filled=True)
    fig.savefig("decision_tree.png")

    # Isso aqui é para gerar o dataframe e podermos ver as predições dos Pokémon
    # Só não inclui os Pokémon que não são Lendários e foram classificados corretamente
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
