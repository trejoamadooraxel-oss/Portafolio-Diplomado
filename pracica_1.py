import numpy as np
#import matplotlib.pyplot as plt

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

    entrenar_perceptron(x, y, tasa_aprendizaje, epocas)
