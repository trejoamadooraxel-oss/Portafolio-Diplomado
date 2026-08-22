import numpy as np 
import matplotlib.pyplot as plt 

def crear_secuencias(serie, pasos): 
    X = [] 
    y = [] 
    for i in range(len(serie) - pasos): 
        X.append(serie[i:i + pasos]) 
        y.append(serie[i + pasos]) 

    return np.array(X), np.array(y) 

def grafica_error_entrenamiento(errores):
    plt.plot(errores) 
    plt.title("Error durante el entrenamiento") 
    plt.xlabel("Épocas") 
    plt.ylabel("Error cuadrático medio") 
    plt.grid(True) 
    plt.show()

def grafica_serie_pronostico(pronosticos):
    plt.plot(y, label="Serie real") 
    plt.plot(pronosticos, label="Pronóstico de la RNN") 
    plt.title("Pronóstico de serie temporal con RNN") 
    plt.xlabel("Tiempo") 
    plt.ylabel("Valor") 
    plt.legend() 
    plt.grid(True) 
    plt.show() 

def tanh(x): 
    return np.tanh(x) 

def derivada_tanh(x): 
    return 1 - x ** 2 

def propagacion_adelante_rnn(secuencia, Wxh, Whh, Why, bh, by):     
    h = np.zeros((1, Whh.shape[0]))     
    estados = []     
    for t in range(len(secuencia)):         
        x_t = secuencia[t].reshape(1, 1)         
        h = tanh(np.dot(x_t, Wxh) + np.dot(h, Whh) + bh)         
        estados.append(h)      

    y_predicha = np.dot(h, Why) + by     

    return y_predicha, estados

def calcular_error(y_real, y_predicha):     
    return np.mean((y_real - y_predicha) ** 2)   

def entrenar_rnn(X, y, Wxh, Whh, Why, bh, by, tasa_aprendizaje, epocas):     
    errores = []      
    for epoca in range(epocas):         
        error_total = 0          

        for i in range(len(X)):             
            secuencia = X[i]             
            valor_real = y[i].reshape(1, 1)              
            y_predicha, estados = propagacion_adelante_rnn(secuencia,Wxh,Whh,Why,bh,by)              

            error = valor_real - y_predicha             
            error_total = error_total + np.mean(error ** 2)              
            h_final = estados[-1]              
            Why = Why + tasa_aprendizaje * np.dot(h_final.T, error)             
            by = by + tasa_aprendizaje * error              
            ajuste_oculto = np.dot(error, Why.T) * derivada_tanh(h_final)              
            ultimo_valor = secuencia[-1].reshape(1, 1)              
            Wxh = Wxh + tasa_aprendizaje * np.dot(ultimo_valor.T, ajuste_oculto)             
            bh = bh + tasa_aprendizaje * ajuste_oculto          

        error_promedio = error_total / len(X)         
        errores.append(error_promedio)            
        if epoca % 100 == 0:
            print("Época:", epoca, "Error:", error_promedio)      
    return Wxh, Whh, Why, bh, by, errores 

if __name__ == '__main__':
    
    tiempo = np.linspace(0, 20, 200) 
    serie = np.sin(tiempo) 

    plt.plot(tiempo, serie) 
    plt.title("Serie temporal original") 
    plt.xlabel("Tiempo") 
    plt.ylabel("Valor") 
    plt.grid(True) 
    plt.show() 

    print("==================================")

    pasos = 5 
    X, y = crear_secuencias(serie, pasos) 
    X = X.reshape(X.shape[0], X.shape[1], 1) 
    y = y.reshape(-1, 1)

    np.random.seed(42) 

    neuronas_entrada = 1 
    neuronas_ocultas = 8 
    neuronas_salida = 1 
    Wxh = np.random.randn(neuronas_entrada, neuronas_ocultas) * 0.1   
    Whh = np.random.randn(neuronas_ocultas, neuronas_ocultas) * 0.1 
    Why = np.random.randn(neuronas_ocultas, neuronas_salida) * 0.1  
    bh = np.zeros((1, neuronas_ocultas)) 
    by = np.zeros((1, neuronas_salida)) 

    tasa_aprendizaje = 0.01 
    epocas = 1000

    Wxh, Whh, Why, bh, by, errores = entrenar_rnn(X,y,Wxh,Whh,Why,bh,by,tasa_aprendizaje,epocas) 

    pronosticos = []  
    for i in range(len(X)):     
        y_predicha, estados = propagacion_adelante_rnn(X[i], Wxh, Whh, Why, bh, by ) 
        pronosticos.append(y_predicha[0, 0]) 
    pronosticos = np.array(pronosticos) 

    grafica_serie_pronostico(pronosticos)

    grafica_error_entrenamiento(errores)

    ultima_secuencia = serie[-pasos:].reshape(pasos, 1) 
    siguiente_valor, estados = propagacion_adelante_rnn( ultima_secuencia, Wxh, Whh, Why, bh, by ) 
    print("Últimos valores utilizados:") 
    print(ultima_secuencia.flatten()) 
    print("Pronóstico del siguiente valor:") 
    print(siguiente_valor[0, 0]) 

    """
    • ¿Qué es una serie temporal? 
        R: Conjunto de observaciones secuanciales 

    • ¿Qué significa realizar un pronóstico? 
        R: realizar estimaciones de valores futuros a partir de datos ya obtenidos

    • ¿Cuál es la diferencia entre clasificación, ajuste de funciones y pronóstico? 
        R:Pronostico: memoria interna almacenando informacion de procesos anteriores
                        para generar una prediccion
          

    • ¿Qué tipo de red neuronal se utilizó en esta práctica? 
        R: red neuronal recurrente simple, conocida como RNN.

    • ¿Por qué una RNN es útil para datos secuenciales? 
        R: Tiene memoria y orden de importacia

    • ¿Qué representa el estado oculto en una RNN? 
        R: Memoria de red, actualizacion constante 

    • ¿Para qué se utilizan los pesos recurrentes Whh? 
        R: Conecta los datos del pasado y con el presente

    • ¿Qué función cumple la tasa de aprendizaje? 
        R: La velocidad y el punto de equilibrio

    • ¿Qué mide el error cuadrático medio? 
        R: Distancia del error y penalizacion de errores grandes

    • ¿Qué representa la gráfica de la serie real y el pronóstico? 
        R: Comparacion visual y ajuste de la evaluacion durante el aprendizaje
        
    • ¿La red neuronal logró pronosticar correctamente la serie temporal? Justifica tu respuesta. 
    """