# validation_methods.py

from sklearn.model_selection import train_test_split, KFold, LeaveOneOut
from sklearn.metrics import accuracy_score, confusion_matrix

def hold_out_evaluation(X, y, classifier):
    """
    Realiza una validación Hold-Out dividiendo el conjunto en 70% entrenamiento y 30% prueba.
    Retorna la accuracy y la matriz de confusión.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    return accuracy, confusion

def cross_validation_evaluation(X, y, classifier):
    """
    Realiza una validación 10-Fold Cross-Validation y retorna el accuracy promedio.
    """
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return sum(accuracies) / len(accuracies)

def leave_one_out_evaluation(X, y, classifier):
    """
    Realiza una validación Leave-One-Out y retorna el accuracy promedio.
    """
    loo = LeaveOneOut()
    accuracies = []
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    return sum(accuracies) / len(accuracies)
