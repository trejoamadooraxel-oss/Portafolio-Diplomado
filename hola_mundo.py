import pandas as pd
# Realizar una neurona artificial de 4 entradas y la clasificacion 3 "gato", "perro", "ave" donde la probabilidad de gato
# sea ‹0.15 , La probabilidad de perro sea › 0.68, la probabilidad de 0.15 a 0.68 sea de "ave" x1, x2, x3 y x4 valdran
# 0.5,0. 67,0.85, 0.32 respectivamente y los pesos w1,w2, w3,w4 seran 0.14,0.22,0.07,0.10 respectivamente y el bias(sesgo) valdra 0.




entradas = [0.5,0.67,0.85,0.32]
pesos = [0.14,0.22,0.07,0.10]
sesgo = 0.09
yp = 0

for i in range(len(entradas)):
    yp = yp + (entradas[i] * pesos[i])

yp = yp + sesgo

if yp < 0.15:
    print('GATO')
elif yp > 0.68:
    print('PERRO')
else:
    print('AVE')