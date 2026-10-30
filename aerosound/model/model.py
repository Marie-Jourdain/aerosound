import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
import pandas as pd


def data_obtain(path):
# obtain data from the path and saperated the data by machine type and save the dataframes into a dictionary and return to this dictionary
    data_ori = pd.read_csv(path)
    machine_dfs = {}

    for machine in data_ori['machine'].unique():
        machine_dfs[machine] = data_ori[
            data_ori['machine'] == machine
        ].copy()

    print(machine_dfs.keys())
    return machine_dfs

def prep_data(data_ori):
    data = data_ori.copy()
    data['label'] = data['label'].map({
        'normal': 0,
        'abnormal': 1
    })
    data = data.drop(columns=['noise','machine','id','file','snr','file_num'])
    data = data.rename(columns={'label':'target'})
    X = data.drop(columns = 'target')
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def best_knn_model(X_train, y_train, score):
# automaticly find the model with the best set of hyperparameters
    # Step 1: Create pipeline
    pipeline = Pipeline([
        ('oversampler', RandomOverSampler(
            sampling_strategy=1.0,
            random_state=42
        )),
        ('knn', KNeighborsClassifier())
    ])

    # Step 2: Define hyperparameters
    param_grid = {
        'knn__n_neighbors': [2, 3, 5, 7, 9, 11, 15, 21],
        'knn__weights': ['uniform', 'distance'],
        'knn__p': [1, 2]
    }

    # Step 3: GridSearchCV
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring=score,
        n_jobs=-1
    )

    # Step 4: Fit original training data
    grid.fit(X_train, y_train)

    # Step 5: Results
    print("Best parameters:", grid.best_params_)
    print("Best CV score:", grid.best_score_)
    return grid.best_estimator_



def show_result(model, X_test, y_test):
# to visulize how good the model is with the test data
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        display_labels=["Normal", "Abnormal"]
    )

    plt.title("Confusion Matrix")
    plt.show()


# test functions above
if __name__ == "__main__":
    machine_dfs = data_obtain('../../features_0dB.csv')
    data_ori = machine_dfs['fan']
    X_train, X_test, y_train, y_test = prep_data(data_ori)
    knn_model = best_knn_model(X_train, y_train, 'recall')
    show_result(knn_model, X_test, y_test)
