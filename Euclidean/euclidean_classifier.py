# euclidean_classifier.py

import numpy as np

class EuclideanClassifier:
    def __init__(self):
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Almacena los datos de entrenamiento.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predice la clase para cada punto en X basado en la distancia Euclidiana al punto de entrenamiento más cercano.
        """
        predictions = []
        for x in X:
            # Calcula la distancia Euclidiana entre x y todos los puntos de entrenamiento
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            # Encuentra el índice del punto más cercano
            nearest_neighbor_index = np.argmin(distances)
            # Asigna la clase del vecino más cercano
            predictions.append(self.y_train[nearest_neighbor_index])
        return np.array(predictions)
