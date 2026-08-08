import numpy as np
import matplotlib.pyplot as plt

def funcion_activacion(valor):
    if valor >= 0: 
        return 1 
    else: 
        return 0 


def predecir(entradas, pesos, sesgo): 
    suma_ponderada = np.dot(entradas, pesos) + sesgo 
    salida = funcion_activacion(suma_ponderada) 
    return salida

def entrenar_perceptron(X, y, tasa_aprendizaje, epocas):
    pesos = np.zeros(X.shape[1])
    sesgo = 0.0
    errores_por_epoca = []
    for epoca in range(epocas):
        error_total = 0
        for i in range(len(X)):
            prediccion = predecir(X[i], pesos, sesgo)
            error = y[i] - prediccion
            pesos = pesos + tasa_aprendizaje * error * X[i]
            sesgo = sesgo + tasa_aprendizaje * error
            error_total = error_total + abs(error)
        errores_por_epoca.append(error_total)
        print("Época:", epoca + 1)
        print("Error total:", error_total)
        print("Pesos:", pesos)
        print("Sesgo:", sesgo)
        print("-------------------------")
    return pesos, sesgo, errores_por_epoca

if __name__ == '__main__':


    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y = np.array([0, 0, 0, 1]) 

    pesos = np.array([0.0, 0.0]) 
    sesgo = 0.0
    tasa_aprendizaje = 0.1 
    epocas = 10 

    pesos_finales, sesgo_final, errores_por_epoca = entrenar_perceptron(x, y, tasa_aprendizaje, epocas)

    print("Pesos finales:", pesos_finales)
    print("Sesgo final:", sesgo_final) 

    print("Pruebas del modelo entrenado:") 
    for entrada in x: 
        salida = predecir(entrada, pesos_finales, sesgo_final) 
        print("Entrada:", entrada, "Predicción:", salida) 

    
    print(f'{errores_por_epoca}')
    print(f'{epocas}')
    plt.plot(range(1, epocas + 1), errores_por_epoca, marker='*') 
    plt.title("Error durante el entrenamiento") 
    plt.xlabel("Época") 
    plt.ylabel("Error total") 
    plt.grid(True) 
    plt.show() 

    entrada_1 = np.array([0, 0]) 
    entrada_2 = np.array([0, 1]) 
    entrada_3 = np.array([1, 0]) 
    entrada_4 = np.array([1, 1]) 
    print("Entrada [0, 0]:", predecir(entrada_1, pesos_finales, sesgo_final)) 
    print("Entrada [0, 1]:", predecir(entrada_2, pesos_finales, sesgo_final)) 
    print("Entrada [1, 0]:", predecir(entrada_3, pesos_finales, sesgo_final)) 
    print("Entrada [1, 1]:", predecir(entrada_4, pesos_finales, sesgo_final)) 


    """
    ¿Qué es una neurona artificial? • 
    R: Son modelados simulando el funcionamiento del cerebro humano
    ¿Qué representan las entradas en el perceptrón? •
    R:  'X' es la entradas del modelo
        'y' es salida
        'taza_aprendizaje' es aquel que controla el tamaño y ajusta durante el entrenamiento
        'epoca' es indica el numero de veces que se ejeutara el entrenamiento
    ¿Qué función cumplen los pesos dentro del modelo? • 
        R:sirven para ajustar la importancia de cada entrada.  ERR
    ¿Qué es el sesgo y por qué es importante? •
        R: es el error, es importante ya que indica al perceptron aprena poco a poco de sus error
    ¿Qué hace la función de activación escalón? • 
        R: Retorna un valor 1 o 0 
    ¿Cómo se calcula el error del modelo? • 
        R: error = salida_esperada - prediccion 
    ¿Qué ocurre cuando el error es igual a cero? • 
        R:La prediccion es correcta
    ¿Qué ocurre cuando la predicción es diferente a la salida esperada? • 
        R:La salida es incorrecta ERROR
    ¿Por qué se actualizan los pesos durante el entrenamiento? •
        R: para ajsutar la importancia de cada entrada  
    ¿Qué indica la gráfica del error? •
        R: Como avanza la epoca con respecto al error 
    ¿El perceptrón logró aprender la compuerta AND? 
    Justifica tu respuesta.
        R:
    """