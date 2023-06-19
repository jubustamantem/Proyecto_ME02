import numpy as np
import pandas as pd
from collections import defaultdict
from statistics import mean, stdev
from tabulate import tabulate
import shutil

class Distribucion_Valores_Observados:
    # Distribución de probabilidad formada al observar y contar ejemplos.

    def __init__(self, iniciales = []):
        # Constructor: crea una distribución, inicializa el total en 0 y agrega las observaciones inciales.
        self.distribucion = {}
        self.total_observaciones = 0

        for observacion in iniciales:
            self.agregar(observacion)

    def agregar(self, observacion):
        # Se agrega una observación a la distribución y se suma uno al total.
        if observacion not in self.distribucion:
            self.distribucion[observacion] = 0
        self.distribucion[observacion] += 1
        self.total_observaciones += 1

    def __getitem__(self, item):
        # Retorna la probabilidad del item al hacer un llamado de tipo diccionario.
        if item in self.distribucion:
            return self.distribucion[item] / self.total_observaciones
        else:
            return 0.0


class Clasificador_Naive_Bayes_Discreto:

    def __init__(self, dataset=None):
        # Constructor: inicializar el dataset.
        self.dataset = dataset

        # Traer las etiquetas de clase, por ejemplo 'spam' y 'ham'.
        self.etiquetas_de_clase = self.dataset.valores_posibles[self.dataset.columna_objetivo]

        # Se crea una distribución de probabilidad para esas etiquetas. Por ejemplo, {spam: 0.5, ham: 0.5}
        self.distribucion_etiquetas = Distribucion_Valores_Observados(self.etiquetas_de_clase)

        # Se crea una distribución de probabilidad para cada par (etiqueta, atributo) a partir de sus valores posibles, es decir, todos los P(xi|C)
        self.distribuciones_atributos = {(c, atributo): Distribucion_Valores_Observados(self.dataset.valores_posibles[atributo]) for c in self.etiquetas_de_clase for atributo in self.dataset.atributos_de_entrada}

    def entrenar(self):
        # Se itera sobre las filas del dataset.
        for fila in self.dataset.datos:

            # Se trae la etiqueta de clase (dice si el mensaje es 'spam' o 'ham').
            valor_objetivo = fila[self.dataset.columna_objetivo]

            # Agregar una observación a la distribución, alterando las P(C)
            self.distribucion_etiquetas.agregar(valor_objetivo)

            # Agregar una observación a cada distribución (etiqueta, atributo), alterando las P(xi|C)
            for atributo in self.dataset.atributos_de_entrada:
                self.distribuciones_atributos[valor_objetivo, atributo].agregar(fila[atributo])

    def predecir(self, ejemplo):
        # Calcular P(C|x1,...,xn)

        def probabilidad_clase(clase):

            # Inicializar P(C)
            probabilidad = self.distribucion_etiquetas[clase]

            # Efectuar productoria P(xi|C)
            for atributo in self.dataset.atributos_de_entrada:
                probabilidad *= self.distribuciones_atributos[clase, atributo][ejemplo[atributo]]

            return probabilidad

        # Elegir el más probable entre las clases.
        return max(self.etiquetas_de_clase, key = probabilidad_clase)


class Clasificador_Naive_Bayes_Continuo:

    def __init__(self, dataset=None):
        # Constructor: inicializar el dataset.
        self.dataset = dataset

        # Traer las etiquetas de clase, por ejemplo 'spam' y 'ham'.
        self.etiquetas_de_clase = self.dataset.valores_posibles[self.dataset.columna_objetivo]

        # Crear una distribución de probabilidad para esas etiquetas. Por ejemplo, {spam: 0.5, ham: 0.5}
        self.distribucion_etiquetas = Distribucion_Valores_Observados(self.etiquetas_de_clase)

        # Inicializar las medias y las desviaciones que se obtendrán en el entrenamiento.
        self.medias = None
        self.desviaciones = None
        
    def entrenar(self):
        # Calcular el total de atributos de entrada
        total_de_entradas = len(self.dataset.atributos_de_entrada)

        # Crear diccionario para organizar los datos por cada clase.
        clases = defaultdict(lambda: [])

        # Agregar datos al diccionario.
        for fila in self.dataset.datos:
            clases[fila[self.dataset.columna_objetivo]].append([val for val in fila if val not in self.etiquetas_de_clase])

        # Crear diccionarios para almacenar las medias y las desviaciones de cada atributo de entrada según la clase.
        self.medias = defaultdict(lambda: [0] * total_de_entradas)
        self.desviaciones = defaultdict(lambda: [0] * total_de_entradas)

        for etiqueta in self.etiquetas_de_clase:
            # Crear una lista de listas para almacenar los valores de características.
            caracteristicas = [[] for _ in range(total_de_entradas)]

            # Se ordenan, por columnas, los datos asociados a la etiqueta.
            for item in clases[etiqueta]:
                for i in range(total_de_entradas):
                    caracteristicas[i].append(item[i])

            # Calcular las medias y desviaciones de cada atributo de entrada para la clase.
            for i in range(total_de_entradas):
                self.medias[etiqueta][i] = mean(caracteristicas[i])
                self.desviaciones[etiqueta][i] = stdev(caracteristicas[i])

    def predecir(self, ejemplo):
        # Calcular P(C|x1,...,xn)

        def probabilidad_clase(clase):

            # Inicializar P(C)
            probabilidad = self.distribucion_etiquetas[clase]

            # Efectuar productoria P(xi|C)
            for atributo in self.dataset.atributos_de_entrada:

                # Cada probabilidad se calcula a partir de la función de densidad normal (o gaussiana).
                if self.desviaciones[clase][atributo] != 0:
                    probabilidad *= distribucion_normal(self.medias[clase][atributo], self.desviaciones[clase][atributo], ejemplo[atributo])
            
            return probabilidad

        # Elegir el más probable.
        return max(self.etiquetas_de_clase, key = probabilidad_clase)
    
def distribucion_normal(media, desviacion, x):
    # Dada la media y la desviación estándar de una distribución, devuelve la probabilidad de x.
    return 1 / float(desviacion * np.sqrt(2 * np.pi)) * float(np.e) ** ( - (float((x - media) ** 2) / float(2 * (desviacion ** 2))))