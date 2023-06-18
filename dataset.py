from statistics import mean, stdev
import os
from collections import defaultdict

class DataSet:
    """
    Clase para definir un conjunto de datos. Tiene los siguientes campos:

    d.examples   Una lista de ejemplos. Cada uno es una lista de valores de atributos.
    d.attrs      Una lista de enteros para indexar un ejemplo, por lo que example[attr]
                 devuelve un valor. Normalmente es lo mismo que range(len(d.examples[0])).

    d.target     El atributo que el algoritmo de aprendizaje intentará predecir.
                 Por defecto es el último atributo.
    d.inputs     La lista de atributos sin el objetivo.
    d.values     Una lista de listas: cada sublista es el conjunto de valores posibles
                 para el atributo correspondiente. Si inicialmente es None,
                 se calcula a partir de los ejemplos conocidos mediante self.set_problem.
                 Si no es None, un valor erróneo genera ValueError.

    d.name       Nombre del conjunto de datos (solo para mostrar en la salida).

    Normalmente, se llama al constructor y ya está; luego solo se accede a los campos como d.examples y d.target y d.inputs.
    """

    def __init__(self, name = '', target = -1):
        """
        Constructor
        """
        self.name = name

        # Inicializar los ejemplos
        self.examples = parse_csv(open(os.path.join(os.path.dirname(__file__), *['datasets', name]), mode='r').read())

        # los attrs son los índices de los ejemplos.
        self.attrs = list(range(len(self.examples[0])))
            
        if target < 0:
            self.target = len(self.attrs) + target
        else:
            self.target = target

        self.inputs = [a for a in self.attrs if a != self.target]

        self.values = list(map(unique, zip(*self.examples)))
        
        """Comprueba que mis campos tengan sentido."""
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))

    def find_means_and_deviations(self):
        """
        Encuentra las medias y desviaciones estándar del dataset.
        medias:         Diccionario que contiene una lista de las medias de los inputs para la clase.
        desviaciones:   Diccionario que contiene una lista de las desviaciones estándar de los inputs para la clase.
        """
        etiquetas_de_clase = self.values[self.target]
        total_de_inputs = len(self.inputs)

        items = defaultdict(lambda: [])

        for ejemplo in self.examples:
            items[ejemplo[self.target]].append([a for a in ejemplo if a not in etiquetas_de_clase])

        medias = defaultdict(lambda: [0] * total_de_inputs)
        desviaciones = defaultdict(lambda: [0] * total_de_inputs)

        for t in etiquetas_de_clase:
            # Encontrar todos los valores de características del elemento para la clase t
            caracteristicas = [[] for _ in range(total_de_inputs)]
            for item in items[t]:
                for i in range(total_de_inputs):
                    caracteristicas[i].append(item[i])

            # Calcular las medias y desviaciones para la clase
            for i in range(total_de_inputs):
                medias[t][i] = mean(caracteristicas[i])
                desviaciones[t][i] = stdev(caracteristicas[i])

        return medias, desviaciones

    def __repr__(self):
        return '<DataSet({}): {:d} ejemplos, {:d} atributos>'.format(self.name, len(self.examples), len(self.attrs))


def parse_csv(input, delimitador = ','):
    """
    Función para convertir archivos .csv en una lista de listas
    """
    archivo = [linea for linea in input.splitlines() if linea.strip()]
    return [list(map(convierte_a_numero, linea.split(delimitador))) for linea in archivo]

def convierte_a_numero(x):
    """El argumento es una cadena; convierte a número si es posible, o elimina los espacios."""
    try:
        return int(x)
    except ValueError:
        try:
            return float(x)
        except ValueError:
            return str(x).strip()

def unique(lista):
    """Elimina los elementos duplicados de seq. Supone que los elementos son hashables."""
    return list(set(lista))