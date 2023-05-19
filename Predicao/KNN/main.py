import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.plotting import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

CSV_FULL_PATH = '../CSVs/pokemon.csv'

if __name__ == '__main__':
    data_pandas = pd.read_csv(CSV_FULL_PATH)
    df = pd.DataFrame(data_pandas)

    x = df.drop(['attack', 'defense', 'sp_attack', 'sp_defense', 'is_legendary', 'name'], axis=1)
    y = df['is_legendary']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    training_accuracy = []
    test_accuracy = []

    precision_list = []
    recall_list = []
    f1_list = []

    neighbors_settings = range(1, 16)

    for n_neighbors in neighbors_settings:
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(x_train, y_train)

        training_accuracy.append(model.score(x_train, y_train))
        test_accuracy.append(model.score(x_test, y_test))

        y_pred = model.predict(x_test)
        recall_list.append(confusion_matrix(y_test, y_pred)[1][1] / (
                    confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[1][0]))
        precision_list.append(confusion_matrix(y_test, y_pred)[1][1] / (
                    confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[0][1]))
        f1_list.append(2 * (precision_list[-1] * recall_list[-1]) / (precision_list[-1] + recall_list[-1]))

    fig = plt.figure(figsize=(10, 5))
    plt.plot(neighbors_settings, training_accuracy, label='Training Accuracy')
    plt.plot(neighbors_settings, test_accuracy, label='Test Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('n_neighbors')
    plt.legend()
    fig.savefig("knn_compare_model.png")

    best_n = test_accuracy.index(max(test_accuracy)) + 1
    best_model = KNeighborsClassifier(n_neighbors=best_n)
    best_model.fit(x_train, y_train)
    y_pred = best_model.predict(x_test)
    print('Accuracy: {}%'.format(accuracy_score(y_test, y_pred) * 100))
    print('Precision: {}%'.format(precision_score(y_test, y_pred) * 100))
    print('Recall: {}%'.format(recall_score(y_test, y_pred) * 100))
    print('F1: {}%'.format(f1_score(y_test, y_pred) * 100))

    print('n_neighbors\t\tTest Accuracy\t\tPrecision\t\tRecall\t\tF1')
    results = pd.DataFrame({'n_neighbors': neighbors_settings, 'Test Accuracy': test_accuracy, 'Precision': precision_list,
                            'Recall': recall_list, 'F1': f1_list})
    results = results.round(2)
    results.to_csv('results.csv', index=False)

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

    # Para duas variaveis
    # fig = plt.figure(figsize=(10, 5))
    # plt.scatter(df['hp'], df['base_total'], c=df['is_legendary'])
    # plt.xlabel('HP')
    # plt.ylabel('Total')
    # plt.savefig('scatter_plot.png')

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
