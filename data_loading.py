
# data_loading.py

from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import pandas as pd

def load_dataset(dataset_name):
    """
    Carga un dataset específico de scikit-learn basado en el nombre dado.
    Retorna las características y etiquetas del dataset.
    """
    if dataset_name == "iris":
        data = load_iris()
        print("\nDataset Iris cargado exitosamente.")
    elif dataset_name == "wine":
        data = load_wine()
        print("\nDataset Wine cargado exitosamente.")
    elif dataset_name == "breast_cancer":
        data = load_breast_cancer()
        print("\nDataset Breast Cancer cargado exitosamente.")
    else:
        raise ValueError("Nombre de dataset no reconocido. Usa 'iris', 'wine' o 'breast_cancer'.")

    X = data.data
    y = data.target
    df = pd.DataFrame(X, columns=data.feature_names)
    df['target'] = y
    print(f"Primeras filas del dataset {dataset_name}:\n", df.head())
    print(f"\nDescripción del dataset {dataset_name}:\n", df.describe())
    return X, y
