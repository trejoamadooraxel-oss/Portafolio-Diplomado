import numpy as np
import matplotlib.pyplot as plt
import math

def graficacion_funcion_real_ajustada(X,y, y_predicha):
    plt.plot(X, y, label="Función real: y = x²")
    plt.plot(X, y_predicha, label="Función ajustada por la red")
    plt.title("Ajuste de función usando red neuronal")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

def grafiacion_error_entrenamiento(errores):
    plt.plot(errores)
    plt.title("Error durante el entrenamiento")
    plt.xlabel("Épocas")
    plt.ylabel("Error cuadrático medio")
    plt.grid(True)
    plt.show()



def graficacion(X, y):
    plt.plot(X, y)
    plt.title("Función original: y = x²")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()


def tanh(x):
    return np.tanh(x)

def derivada_tanh(x):
    return 1 - np.tanh(x) ** 2

def red_neuronal():
    np.random.seed(42)
    neuronas_entrada = 1
    neuronas_ocultas = 5
    neuronas_salida = 1
    W1 = np.random.randn(neuronas_entrada, neuronas_ocultas)
    b1 = np.zeros((1, neuronas_ocultas))
    W2 = np.random.randn(neuronas_ocultas, neuronas_salida)
    b2 = np.zeros((1, neuronas_salida))

    return W1, b1, W2, b2

def propagacion_adelante(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = tanh(Z1)
    Z2 = np.dot(A1, W2) + b2

    return Z1, A1, Z2

def calcular_error(y_real, y_predicha):
    return np.mean((y_real - y_predicha) ** 2)

def entrenar_red(X, y, W1, b1, W2, b2, tasa_aprendizaje, epocas):
    errores = []
    n = len(X)
    for epoca in range(epocas):
        Z1, A1, y_predicha = propagacion_adelante(X, W1, b1, W2, b2)
        error = calcular_error(y, y_predicha)
        errores.append(error)
        dZ2 = (y_predicha - y)
        dW2 = np.dot(A1.T, dZ2) / n
        db2 = np.sum(dZ2, axis=0, keepdims=True) / n
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * derivada_tanh(Z1)
        dW1 = np.dot(X.T, dZ1) / n
        db1 = np.sum(dZ1, axis=0, keepdims=True) / n
        W2 = W2 - tasa_aprendizaje * dW2
        b2 = b2 - tasa_aprendizaje * db2
        W1 = W1 - tasa_aprendizaje * dW1
        b1 = b1 - tasa_aprendizaje * db1
        if epoca % 500 == 0:
            print("Época:", epoca, "Error:", error)

    return W1, b1, W2, b2, errores

if __name__ == '__main__':

    """
        Accion:
            Inicializamos las variables y mandamos a llamar las funciones para el calculo de 
            "pesos_finales", "sesgo_final" y "errores_por_epoca" y poder graficar el resultado
            de la neurona.
    """

    x = np.linspace(-1, 1, 100).reshape(-1, 1)
    y = x ** 2
    tasa_aprendizaje = 0.1
    epocas = 5000


    #print(y)
    print("\n")
    print("-------------------------")

    #graficacion(x, y)
    print("\n")
    print("-------------------------")
    
   
    print("\n")
    print("-------------------------")
    np.random.seed(42)
    neuronas_entrada = 1
    neuronas_ocultas = 5
    neuronas_salida = 1
    W1 = np.random.randn(neuronas_entrada, neuronas_ocultas)
    b1 = np.zeros((1, neuronas_ocultas))
    W2 = np.random.randn(neuronas_ocultas, neuronas_salida)
    b2 = np.zeros((1, neuronas_salida))

    z1 = (x) + b1
    a1 = tanh(z1)
    z2 = np.dot(a1, W2) + b2
    print("\n")
    print("-------------------------")

    print(f'Valor de Z1: {z1}')
    print(f'Valor de Z1: {z2}')
    print(f'Valor de Z1: {a1}')

    print("\n")
    print("-------------------------")
    W1, b1, W2, b2, errores = entrenar_red( x,
                                            y,
                                            W1,
                                            b1,
                                            W2,
                                            b2,
                                            tasa_aprendizaje,
                                            epocas)

Z1, A1, y_predicha = propagacion_adelante(x, W1, b1, W2, b2)                

graficacion_funcion_real_ajustada(x,y, y_predicha)

grafiacion_error_entrenamiento(errores)

print('Probar vvalores con nuevas entradas:')
nuevos_valores = np.array([
 [-0.8],
 [-0.5],
 [0.0],
 [0.5],
 [0.8]
])
Z1_nuevo, A1_nuevo, predicciones_nuevas = propagacion_adelante(
 nuevos_valores, W1, b1,  W2,  b2)

for i in range(len(nuevos_valores)):
    print("Entrada:", nuevos_valores[i][0])
    print("Predicción de la red:", predicciones_nuevas[i][0])
    print("Valor real:", nuevos_valores[i][0] ** 2)
    print("-------------------------")


'''
¿Qué significa ajustar una función utilizando una red neuronal?
Consiste en capacitar a la red para que descubra y comprenda el vínculo matemático existente entre los datos de entrada y salida. De esta manera, logra imitar el comportamiento de una función continua modificando sus pesos y sesgos.
¿En qué se diferencian la clasificación y el ajuste de funciones?
La clasificación se encarga de agrupar las entradas en categorías o etiquetas independientes (por ejemplo, identificar si una fotografía muestra un perro o un gato). En contraste, el ajuste de funciones —o regresión— calcula valores numéricos continuos basándose en las entradas (por ejemplo, estimar el valor continuo de y para cualquier x).
¿Cuál fue la función matemática empleada en esta práctica?
Se utilizó la función cuadrática y=x^2.
¿Qué define a la variable X?
Corresponde al conjunto de datos de entrada o variable independiente, los cuales están delimitados en el intervalo [-1,1].
¿Qué indica la variable y?
Hace referencia a los valores objetivo o salidas reales (variable dependiente) que se obtienen al resolver la ecuación y=x^2.
¿Para qué sirven los pesos W_1 y W_2?
Funcionan como parámetros ajustables que regulan la fuerza de las conexiones neuronales. Específicamente, W_1 pondera las señales que van hacia la capa oculta, mientras que W_2 hace lo propio desde la capa oculta hacia la capa de salida.
¿Cuál es el propósito de los sesgos b_1 y b_2?
Su función es desplazar la función de activación sobre el eje, brindando al modelo la flexibilidad necesaria para representar curvas que no cruzan obligatoriamente por el origen.
¿Cuál es la utilidad de la función tangente hiperbólica?
Corresponde a la función de activación no lineal (tanh) implementada en la capa oculta. Su propósito es aportar la no linealidad indispensable para que la red sea capaz de replicar trayectorias complejas, como la parábola y=x^2, algo inviable mediante operaciones puramente lineales.
¿Qué evalúa el error cuadrático medio?
Calcula la media de las diferencias al cuadrado entre los valores verdaderos (y) y las estimaciones emitidas por la red (y ˆ). Funciona como un indicador de pérdida para valorar la efectividad general del modelo.
¿Qué significa que el error se reduzca durante la fase de entrenamiento?
Evidencia que los procesos de propagación hacia atrás (backpropagation) y descenso de gradiente están optimizando de forma adecuada los parámetros (W y b). Como resultado, las estimaciones del modelo se vuelven cada vez más exactas y cercanas a los valores reales.
¿Qué ilustra la gráfica que contrasta la función real frente a la ajustada?
Muestra de manera visual la diferencia entre la parábola teórica y=x^2 y la traza estimada por la red neuronal, lo que facilita evaluar gráficamente el nivel de aprendizaje del patrón.
¿La red neuronal consiguió aproximar la función de forma adecuada? ¿Por qué?
Sí, el modelo alcanzó un nivel de aproximación sumamente exacto. Esto se comprueba al observar que el Error Cuadrático Medio se reduce de manera constante hasta cifras casi nulas a lo largo de las 5000 épocas, lo que permite que la curva estimada coincida prácticamente con la función real y=x^2, tanto en los datos de entrenamiento como en las evaluaciones con nuevos valores.


'''