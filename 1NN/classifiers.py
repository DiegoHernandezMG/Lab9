# classifiers.py

from sklearn.neighbors import KNeighborsClassifier

def create_1nn_classifier():
    """
    Se crea un clasificador 1-Nearest Neighbor (1NN) usando distancia Euclidiana.
    Retorna el modelo clasificador.
    """
    classifier = KNeighborsClassifier(n_neighbors=1)  # Configuramos 1NN
    print("Clasificador 1-Nearest Neighbor creado exitosamente.")
    return classifier



