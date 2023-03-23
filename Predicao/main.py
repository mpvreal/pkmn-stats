from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

import pandas as pd

CSV_TRAIN_PATH = 'Predicao/treino.csv'
CSV_TEST_PATH =  'Predicao/testes.csv'
CSV_FULL_PATH = 'Predicao/pokemon.csv'

if __name__ == '__main__':
    #load pokemon and tran test split
    data_pandas = pd.read_csv(CSV_FULL_PATH)


