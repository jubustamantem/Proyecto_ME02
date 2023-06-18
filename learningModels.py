import numpy as np
import pandas as pd
from tabulate import tabulate
import shutil

class Distribucion_Valores_Observados:

    """Distribución de probabilidad formada al observar y contar ejemplos."""

    def __init__(self, iniciales = []):
        """Constructor: crea una distribución, inicializa el total en 0 y agrega las observaciones inciales."""
        self.distribucion = {}
        self.total_observaciones = 0

        for observacion in iniciales:
            self.agregar(observacion)

    def agregar(self, observacion):
        """Agrega una observación a la distribución y suma uno al total."""
        if observacion not in self.distribucion:
            self.distribucion[observacion] = 0
        self.distribucion[observacion] += 1
        self.total_observaciones += 1

    def __getitem__(self, item):
        """Devuelve la probabilidad del item al hacer un llamado de tipo diccionario."""
        if item not in self.distribucion:
            self.distribucion[item] = 0
        return self.distribucion[item] / self.total_observaciones


class ClasificadorNaiveBayesDiscreto:

    def __init__(self, dataset=None):
        """Constructor: inicializa el dataset."""
        self.dataset = dataset

        # Traer las etiquetas de clase, por ejemplo 'spam' y 'ham'.
        self.etiquetas_de_clase = self.dataset.values[self.dataset.target]

        # Crea una distribución de probabilidad para esas etiquetas. Por ejemplo, {spam: 0.5, ham: 0.5}
        self.distribucion_objetivo = Distribucion_Valores_Observados(self.etiquetas_de_clase)

        # Crea una distribución de probabilidad para cada par (etiqueta, input): P(xi|C)
        self.distribuciones_atributos = {(c, atributo): Distribucion_Valores_Observados(self.dataset.values[atributo]) for c in self.etiquetas_de_clase for atributo in self.dataset.inputs}

    def entrenar(self):
        # Iteramos sobre las filas del dataset.
        for ejemplo in self.dataset.examples:

            # Trae la etiqueta de clase (dice si el mensaje es 'spam' o 'ham').
            valor_objetivo = ejemplo[self.dataset.target]

            # Agregar una observación a la distribución.
            self.distribucion_objetivo.agregar(valor_objetivo)

            # Agregar una observación a cada distribución (etiqueta, input): P(xi|C)
            for attr in self.dataset.inputs:
                self.distribuciones_atributos[valor_objetivo, attr].agregar(ejemplo[attr])

    def predecir(self, test):

        def probabilidad_clase(valor_objetivo):

            # Inicializar P(C)
            probabilidad = self.distribucion_objetivo[valor_objetivo]

            # Efectuar productoria P(xi|C)
            for atributo in self.dataset.inputs:
                probabilidad *= self.distribuciones_atributos[valor_objetivo, atributo][test[atributo]]

            return probabilidad

        # Elegir el más probable entre las clases.
        return max(self.etiquetas_de_clase, key = probabilidad_clase)


class ClasificadorNaiveBayesContinuo:

    def __init__(self, dataset=None):
        """Constructor: inicializa el dataset."""
        self.dataset = dataset

        # Traer las etiquetas de clase, por ejemplo 'spam' y 'ham'.
        self.etiquetas_de_clase = self.dataset.values[self.dataset.target]

        # Crea una distribución de probabilidad para esas etiquetas. Por ejemplo, {spam: 0.5, ham: 0.5}
        self.distribucion_objetivo = Distribucion_Valores_Observados(self.etiquetas_de_clase)
        
    def entrenar(self):
        # Encontrar las medias y desviaciones de los valores de los atributos de entrada para cada etiqueta de clase.
        self.medias, self.desviaciones = self.dataset.find_means_and_deviations()

    def predecir(self, test):

        def probabilidad_clase(valor_objetivo):

            # Inicializar P(C)
            probabilidad = self.distribucion_objetivo[valor_objetivo]

            # Efectuar productoria P(xi|C)
            for atributo in self.dataset.inputs:
                probabilidad *= distribucion_normal(self.medias[valor_objetivo][atributo], self.desviaciones[valor_objetivo][atributo], test[atributo])
            
            return probabilidad

        # Elegir el más probable.
        return max(self.etiquetas_de_clase, key = probabilidad_clase)
    
def distribucion_normal(media, desviacion, x):
    """Dada la media y la desviación estándar de una distribución, devuelve la probabilidad de x."""
    return 1 / (desviacion * np.sqrt(2 * np.pi)) * np.e ** ( - (float((x - media) ** 2) / float(2 * (desviacion ** 2))))