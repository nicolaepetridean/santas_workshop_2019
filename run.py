from analyze.load_data import TitanicDataLoad
from analyze.feature_importance import analze_feature_importance
from model.first_model import create_model_and_train
from model.first_model import labelEncoder
import numpy as np

if __name__ == "__main__":
    data = TitanicDataLoad()
    data.loadFile("/Users/nicolaepetridean/jde/projects/titanic/try/data/")

    #analze_feature_importance(data.train_data[:,-2], data.train_data[-2])

    train_data_X = labelEncoder(data.train_data_X, 1)

    test_data_X = labelEncoder(data.test_data_X, 1)

    #train_data_X = labelEncoder(train_data_X, 6)

    train_data_X = np.hstack((train_data_X, train_data_X[:, 3].reshape(891,1) + train_data_X[:, 4].reshape(891,1)))
    train_data_X = np.delete(train_data_X, 3, 1)
    train_data_X = np.delete(train_data_X, 3, 1)

    test_data_X = np.hstack((test_data_X, test_data_X[:, 3].reshape(418,1) + test_data_X[:, 4].reshape(418,1)))
    test_data_X = np.delete(test_data_X, 3, 1)
    test_data_X = np.delete(test_data_X, 3, 1)

    se = create_model_and_train(train_data_X, data.train_data_Y, test_data_X)

    data.submission['check'] = se

    series = []
    for val in data.submission.check:
        if val[0] >= 0.5:
            series.append(1)
        else:
            series.append(0)
    data.submission['final'] = series
    final_data = data.submission.drop(['Survived', 'check'], axis=1)
    final_data = final_data.rename(columns={"final": "Survived"})
    final_data.to_csv("/Users/nicolaepetridean/jde/projects/titanic/try/data/submission.csv", index = None, header=True)


