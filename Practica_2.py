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
