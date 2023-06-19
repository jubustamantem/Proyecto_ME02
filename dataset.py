import os

class DataSet:
    # Clase para definir un conjunto de datos.

    def __init__(self, nombre = '', columna_objetivo = -1):

        # Constructor: recibe el nombre del conjunto de datos, que equivale al nombre del archivo de entrada.
        # También puede recibir la columna donde se tienen las etiquetas de clase.
        self.nombre = nombre

        # Se inicializan los datos que vienen desde el archivo csv.
        self.datos = self.leer_csv(open(os.path.join(os.path.dirname(__file__), *['datasets', nombre]), mode='r').read())

        # Se definen los valores posibles que puede tomar cada atributo/columna en una lista de listas.
        self.valores_posibles = list(map(self.eliminar_repeticiones, zip(*self.datos)))

        # Las columnas son los índices de los datos; se guardan para indexarlos con facilidad.
        self.columnas = list(range(len(self.datos[0])))

        # Se inicializa la columna objetivo, es decir, la que se intentará predecir.    
        if columna_objetivo < 0:
            self.columna_objetivo = len(self.columnas) + columna_objetivo
        else:
            self.columna_objetivo = columna_objetivo

        # Los atributos de entrada corresponden a las columnas sin incluir el objetivo.
        self.atributos_de_entrada = [a for a in self.columnas if a != self.columna_objetivo]
        
        # Se verifica que la inicialización tenga sentido.
        assert self.columna_objetivo in self.columnas
        assert self.columna_objetivo not in self.atributos_de_entrada
        assert set(self.atributos_de_entrada).issubset(set(self.columnas))

    def eliminar_repeticiones(self, lista):
        # Se eliminan los elementos duplicados de una lista.
        return list(set(lista))

    def leer_csv(self, archivo, delimitador = ','):
        # Función para convertir archivos .csv en una lista de listas
        dataset = [fila for fila in archivo.splitlines() if fila.strip()]
        return [list(map(self.convertir_a_numero, fila.split(delimitador))) for fila in dataset]

    def convertir_a_numero(self, x):
        # Recibe una cadena e intenta convertirla a entero o flotante; de no ser posible, se deja como texto sin espacios.
        try:
            return int(x)
        except ValueError:
            try:
                return float(x)
            except ValueError:
                return str(x).strip()