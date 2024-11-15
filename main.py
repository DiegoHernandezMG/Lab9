# main.py

from data_loading import load_dataset
from classifiers import create_1nn_classifier
from validation_methods import hold_out_evaluation, cross_validation_evaluation, leave_one_out_evaluation

# Crear el clasificador 1NN
classifier = create_1nn_classifier()

# Evaluar el dataset Iris
print("\n*** Iniciando Evaluación del Dataset Iris ***")
X_iris, y_iris = load_dataset("iris")
accuracy_holdout, confusion_holdout = hold_out_evaluation(X_iris, y_iris, classifier)
print(f"[Hold-Out 70/30] Accuracy: {accuracy_holdout}")
print(f"[Hold-Out 70/30] Matriz de Confusión:\n{confusion_holdout}")
accuracy_cross_val = cross_validation_evaluation(X_iris, y_iris, classifier)
print(f"[10-Fold Cross-Validation] Accuracy promedio: {accuracy_cross_val}")
accuracy_loo = leave_one_out_evaluation(X_iris, y_iris, classifier)
print(f"[Leave-One-Out] Accuracy promedio: {accuracy_loo}")

# Evaluar el dataset Wine
print("\n*** Iniciando Evaluación del Dataset Wine ***")
X_wine, y_wine = load_dataset("wine")
accuracy_holdout, confusion_holdout = hold_out_evaluation(X_wine, y_wine, classifier)
print(f"[Hold-Out 70/30] Accuracy: {accuracy_holdout}")
print(f"[Hold-Out 70/30] Matriz de Confusión:\n{confusion_holdout}")
accuracy_cross_val = cross_validation_evaluation(X_wine, y_wine, classifier)
print(f"[10-Fold Cross-Validation] Accuracy promedio: {accuracy_cross_val}")
accuracy_loo = leave_one_out_evaluation(X_wine, y_wine, classifier)
print(f"[Leave-One-Out] Accuracy promedio: {accuracy_loo}")

# Evaluar el dataset Breast Cancer
print("\n*** Iniciando Evaluación del Dataset Breast Cancer ***")
X_cancer, y_cancer = load_dataset("breast_cancer")
accuracy_holdout, confusion_holdout = hold_out_evaluation(X_cancer, y_cancer, classifier)
print(f"[Hold-Out 70/30] Accuracy: {accuracy_holdout}")
print(f"[Hold-Out 70/30] Matriz de Confusión:\n{confusion_holdout}")
accuracy_cross_val = cross_validation_evaluation(X_cancer, y_cancer, classifier)
print(f"[10-Fold Cross-Validation] Accuracy promedio: {accuracy_cross_val}")
accuracy_loo = leave_one_out_evaluation(X_cancer, y_cancer, classifier)
print(f"[Leave-One-Out] Accuracy promedio: {accuracy_loo}")
