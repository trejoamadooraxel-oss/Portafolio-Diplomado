import numpy as np
import matplotlib.pyplot as plt



def predecir(entradas, pesos, sesgo): 
    """
        varaibles "parametros":
            entradas: valor indice de un arreglo multidimencional
            pesos: arreglo multidimencional
            sesgo: valor
        
        varaibles:
            suma_ponderada: valor entero
            salida:  valor entero

        accion: 
            Realiza una sumatoria con no.dot() viendose asi:
            |entrada| * |pesos|
            [0, 0] · [0., 0.]

            Nota: Entrada puede cambiar de [0,0] a [1,1] eso 
            dependera de como hayas hecho el arreglo y que indice
            le pases como entrada.

            una vez teniendo la "suma_ponderada" la pasamos como parametro
            a "funcion_vectorial" y este retornara una salida
        
        return:
            salida
        
    """
    suma_ponderada = np.dot(entradas, pesos) + sesgo 
    salida = funcion_activacion(suma_ponderada) 
    return salida

def entrenar_perceptron(X, y, tasa_aprendizaje, epocas):
    """
        varaibles "parametros":
            X: arreglo multidimencional
            y: arreglo multidimencional
            tasa_aprendizaje: valor doble
            epocas: valor entero
        
        varaibles:
            error_total: valor cero
            pesos: arreglo multidimencional en ceros
            errores_por_epocas: arreglo vacio
            sesgo: valor entero
            prediccion: valor retornado de la funcion "predecir"
        
        accion:
            Realiza un ciclo anidado del rango de "epocas" y lonitud
            de "x" donde en este ultimo ciclo obtendremos valores   
            redefinidos de varaibles "prediccion", "error", "pesos",
            "sesgo" y "error_total", mientras al finalizar cada ciclo
            de "epocas" se almacenara "error_total" en el arreglo 
            "errores_por_epocas"
        
        return:
            pesos
            sesgo
            errores_por_epoca
    """

    # no.zeros crea un arreglo de la misma caracteristica del arreglo multidimencional
    # X pero este nuevo arreglo esta lleno de ceros y el shape[#], indicara cuantas 
    # columas va a tomar
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

def funcion_activacion(valor):
    """
        variables "parametros"
            valor: valor double

        accion:
            compara si el valor es mayor a 0 ó menor a 1 

        return:
            0 ó 1    
    """

    if valor >= 0: 
        return 1 
    else: 
        return 0 

def graficacion(errores_por_epoca,epocas):
    """
        varaibles "parametros":
            errores_por_epoca: valor entero
            epocas: valor entero
        
        accion:
            Graficamos para el eje "Y" = "errores por epocas"
            mientras el eje "X" = "epocas"
            
    """
    #print(f'{errores_por_epoca}')
    #print(f'{epocas}')
    plt.plot(range(1, epocas + 1), errores_por_epoca, marker='*') 
    plt.title("Error durante el entrenamiento") 
    plt.xlabel("Época") 
    plt.ylabel("Error total") 
    plt.grid(True) 
    plt.show() 


if __name__ == '__main__':

    """
        Accion:
            Inicializamos las variables y mandamos a llamar las funciones para el calculo de 
            "pesos_finales", "sesgo_final" y "errores_por_epoca" y poder graficar el resultado
            de la neurona.
    """

    #np.array sirve para crear arreglos multudimencionales
    x = np.array([ [0, 0], [0, 1], [1, 0], [1, 1] ])
    y = np.array([0, 0, 0, 1]) 
    tasa_aprendizaje = 0.1 
    epocas = 10 

    pesos_finales, sesgo_final, errores_por_epoca = entrenar_perceptron(x, y, tasa_aprendizaje, epocas)

    print("\n")
    print("Pesos finales:", pesos_finales)
    print("Sesgo final:", sesgo_final) 
    print("\n")

    print("Pruebas del modelo entrenado:") 
    for entrada in x: 
        salida = predecir(entrada, pesos_finales, sesgo_final) 
        print("Entrada:", entrada, "Predicción:", salida) 


    graficacion(errores_por_epoca,epocas)

    print("-------------------------") 
    print("Pruebas indipendientes al proceso 'Pruebas'.") 
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
        R:mide la importancia de la columa o caracteristica para el resultado
    ¿Qué es el sesgo y por qué es importante? •
        R: es el error, es importante ya que indica al perceptron aprena poco a poco de sus error
    ¿Qué hace la función de activación escalón? • 
        R: Retorna un valor 1 o 0 
    ¿Cómo se calcula el error del modelo? • 
        R: error = salida_esperada - prediccion 
    ¿Qué ocurre cuando el error es igual a cero? • 
        R:La prediccion es correcta
    ¿Qué ocurre cuando la predicción es diferente a la salida esperada? • 
        R:La salida es correcta
    ¿Por qué se actualizan los pesos durante el entrenamiento? •
        R: para ajsutar la importancia de cada entrada  
    ¿Qué indica la gráfica del error? •
        R: Como avanza la epoca con respecto al error 
    ¿El perceptrón logró aprender la compuerta AND? 
    Justifica tu respuesta.
        R: SI. por que durante el proceder de las epocas el perceptron va aprendiendo
        el comportamiento y va a justando los valores para llegar al resultado una vez
        cumplido el ciclo.
    """