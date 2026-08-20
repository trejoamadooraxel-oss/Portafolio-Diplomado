"""
Tarea  - Analisis de Dataset (Finanzas / Economia)
Dataset: US Macroeconomic Data (macrodata) - incluido en statsmodels
Fuente original: Federal Reserve Economic Data (FRED) / Bureau of Economic Analysis

Requisitos:
    pip install pandas matplotlib statsmodels
"""

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------
# 1) BUSCAR / CARGAR EL DATASET
# -----------------------------------------------------------------

df = pd.read_csv("/Users/axel/Documents/Portafolio-Diplomado/tarea_df_grafica/macrodata.csv")

# Crear una columna de fecha real combinando year + quarter (mas facil de graficar)
periodos = [
    f"{int(y)}Q{int(q)}" for y, q in zip(df["year"], df["quarter"])
]
df["fecha"] = pd.PeriodIndex(periodos, freq="Q").to_timestamp()

# -----------------------------------------------------------------
# 2) ANALIZAR PARA QUE SIRVE
# -----------------------------------------------------------------
print("=" * 70)
print("PARA QUE SIRVE ESTE DATASET")
print("=" * 70)
print("""
Contiene indicadores macroeconomicos trimestrales de Estados Unidos
entre 1959 y 2009 (203 observaciones). Sirve para estudiar la relacion
entre variables economicas clave a lo largo del tiempo: crecimiento
del PIB, consumo, inversion, gasto de gobierno, inflacion, desempleo
y tasas de interes. Es un dataset clasico para practicar analisis de
series de tiempo y modelos economicos (regresion, correlacion, etc).
""")

# -----------------------------------------------------------------
# 3) LOS DATOS QUE LO INTEGRAN (estructura)
# -----------------------------------------------------------------
print("=" * 70)
print("ESTRUCTURA DEL DATASET (columnas y tipos)")
print("=" * 70)
print(df.info())

descripcion_columnas = {
    "year": "Anio",
    "quarter": "Trimestre (1-4)",
    "realgdp": "PIB real (miles de millones USD)",
    "realcons": "Consumo real de las personas",
    "realinv": "Inversion privada real",
    "realgovt": "Gasto real del gobierno federal",
    "realdpi": "Ingreso disponible real",
    "cpi": "Indice de precios al consumidor",
    "m1": "Oferta monetaria M1",
    "tbilrate": "Tasa de interes de bonos del Tesoro a 3 meses",
    "unemp": "Tasa de desempleo (%)",
    "pop": "Poblacion total",
    "infl": "Tasa de inflacion",
    "realint": "Tasa de interes real",
    "fecha": "Fecha (derivada de year+quarter)",
}
print("\nDescripcion de columnas:")
for col, desc in descripcion_columnas.items():
    print(f"  - {col}: {desc}")

# -----------------------------------------------------------------
# 4) DATOS DE MUESTRA
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("MUESTRA DE DATOS (primeras 5 filas)")
print("=" * 70)
print(df.head())

print("\n" + "=" * 70)
print("ESTADISTICAS DESCRIPTIVAS")
print("=" * 70)
print(df.describe())

# -----------------------------------------------------------------
# 5) CONFIRMAR QUE ES CANDIDATO PARA DATAFRAME
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("VALIDACION COMO DATAFRAME")
print("=" * 70)
print(f"Tipo de objeto: {type(df)}")
print(f"Filas x Columnas: {df.shape}")
print(f"Sin valores nulos: {df.isnull().sum().sum() == 0}")

# -----------------------------------------------------------------
# 6) GRAFICOS
# -----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Indicadores macroeconomicos de EE.UU. (1959-2009)", fontsize=14)

axes[0, 0].plot(df["fecha"], df["realgdp"], color="#1f77b4")
axes[0, 0].set_title("PIB real")
axes[0, 0].set_ylabel("Miles de millones USD")

axes[0, 1].plot(df["fecha"], df["unemp"], color="#d62728")
axes[0, 1].set_title("Tasa de desempleo")
axes[0, 1].set_ylabel("%")

axes[1, 0].plot(df["fecha"], df["infl"], color="#2ca02c")
axes[1, 0].set_title("Inflacion")
axes[1, 0].set_ylabel("%")

axes[1, 1].plot(df["fecha"], df["tbilrate"], color="#9467bd")
axes[1, 1].set_title("Tasa de interes (T-bill 3 meses)")
axes[1, 1].set_ylabel("%")

for ax in axes.flat:
    ax.set_xlabel("Fecha")
    ax.grid(alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("indicadores_macro.png", dpi=150)
plt.close()

# Grafico extra: correlacion entre variables
plt.figure(figsize=(8, 6))
cols_num = ["realgdp", "realcons", "realinv", "unemp", "infl", "tbilrate"]
corr = df[cols_num].corr()
plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
plt.xticks(range(len(cols_num)), cols_num, rotation=45, ha="right")
plt.yticks(range(len(cols_num)), cols_num)
plt.colorbar(label="Correlacion")
plt.title("Matriz de correlacion")
for i in range(len(cols_num)):
    for j in range(len(cols_num)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.tight_layout()
plt.savefig("matriz_correlacion.png", dpi=150)
plt.close()

# Guardar el dataset como CSV por si se necesita
df.to_csv("macrodata.csv", index=False)

print("\nArchivos generados: indicadores_macro.png, matriz_correlacion.png, macrodata.csv")
