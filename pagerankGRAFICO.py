# -*- coding: utf-8 -*-
"""
Created on Sat Apr 19 17:16:42 2025

@author: pedro
"""

# Carga de paquetes necesarios para graficar
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd # Para leer archivos
import geopandas as gpd # Para hacer cosas geográficas
import seaborn as sns # Para hacer plots lindos
#import networkx as nx # Construcción de la red en NetworkX
from scipy.linalg import lu, solve_triangular


# Leemos el archivo, retenemos aquellos museos que están en CABA, y descartamos aquellos que no tienen latitud y longitud
museos = gpd.read_file('https://raw.githubusercontent.com/MuseosAbiertos/Leaflet-museums-OpenStreetMap/refs/heads/principal/data/export.geojson')
barrios = gpd.read_file('https://cdn.buenosaires.gob.ar/datosabiertos/datasets/ministerio-de-educacion/barrios/barrios.geojson')



#Matriz de Distancia

# En esta línea:
# Tomamos museos, lo convertimos al sistema de coordenadas de interés, extraemos su geometría (los puntos del mapa), 
# calculamos sus distancias a los otros puntos de df, redondeamos (obteniendo distancia en metros), y lo convertimos a un array 2D de numpy
D = museos.to_crs("EPSG:22184").geometry.apply(lambda g: museos.to_crs("EPSG:22184").distance(g)).round().to_numpy()


#Matriz de adyacencia

def construye_adyacencia(D,m): 
    # Función que construye la matriz de adyacencia del grafo de museos
    # D matriz de distancias, m cantidad de links por nodo
    # Retorna la matriz de adyacencia como un numpy.
    D = D.copy()
    l = [] # Lista para guardar las filas
    for fila in D: # recorriendo las filas, anexamos vectores lógicos
        l.append(fila<=fila[np.argsort(fila)[m]] ) # En realidad, elegimos todos los nodos que estén a una distancia menor o igual a la del m-esimo más cercano
    A = np.asarray(l).astype(int) # Convertimos a entero
    np.fill_diagonal(A,0) # Borramos diagonal para eliminar autolinks
    return(A)


#Calculo de la inversa

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    P, L, U = lu(A)

    # Inicializamos la matriz identidad I
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float)

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):
        
        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = solve_triangular(L, P @ b, lower=True)
        
        x = solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv


#Matriz de Grado

def matriz_Grado (A): 
    
    n = A.shape[0]
    m = A.shape[1]
    K = np.zeros((m, n)) 
    sumaFilasA = np.sum(A, axis = 1)
    
    if m!=n:
        print('Estamos trabajando con una matriz no cuadrada')
        return
    
    for i in range (len (sumaFilasA)):
        K[i, i] = sumaFilasA[i] 
        
    return K   


#Matriz de transicion

def matriz_Transicion1 (A):
    A_traspuesta = np.transpose(A)
    K = matriz_Grado(A)
    K_inversa = inversa_por_lu(K)
    C = A_traspuesta @ K_inversa
    
    return C


# Calculo del PageRank

def calculo_Page_Rank(A, alpha, N):
    n = A.shape[0]
    
    # Genero la matriz de transicion en base a A y una identidad en base a n de A
    C = matriz_Transicion1(A)
    I = np.eye(n)
    
    # Genero b = 1 y hago la cuenta de P
    
    M = (N/alpha) * ( I - (1-alpha) * C )
    b = np.ones(n)
    
    P = inversa_por_lu(M) @ b
    
    return P

A1 = construye_adyacencia(D,1) # Construimos la matriz de adyacencia

page_Rank1 = calculo_Page_Rank(A1, 0.2, 1)
page_Rank1 = page_Rank1 / np.sum(page_Rank1) # Normalizamos para hacer mas viable la visualizacion

print (page_Rank1) 

#Nprincipales = 3
#principales = np.argsort(page_Rank1)[-Nprincipales:]

#print (principales)  

A3 = construye_adyacencia(D,3) # Construimos la matriz de adyacencia

page_Rank3 = calculo_Page_Rank(A3, 0.2, 3)
page_Rank3 = page_Rank3 / np.sum(page_Rank3)

#print (page_Rank3)

A5 = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rank5 = calculo_Page_Rank(A5, 0.2, 5)
page_Rank5 = page_Rank5 / np.sum(page_Rank5)

#print (page_Rank5)

A10 = construye_adyacencia(D,10) # Construimos la matriz de adyacencia

page_Rank10 = calculo_Page_Rank(A10, 0.2, 10)
page_Rank10 = page_Rank10 / np.sum(page_Rank10)

#print (page_Rank10)

# Ahora veamos si podemos hacer el grafico

"""
m_values = [1, 3, 5, 10]

# Hago un dict de los page rank de los museos con mayor promedio para todos los m


page_rank = {
    65:  [0.02212418, 0.02413501, 0.01952598, 0.01697050],
    107: [0.02170588, 0.02352940, 0.02161402, 0.01033586],
    34:  [0.02128758, 0.02128085, 0.00545939, 0.00813473],
    117: [0.01794118, 0.02413501, 0.02106566, 0.01829068],
    
}


plt.figure(figsize=(8, 5))

for museo_id, valores in page_rank.items():
    plt.plot(m_values, valores, marker='o', label=f'Museo {museo_id}')

plt.title('Evolución del PageRank de Museos según m')
plt.xlabel('m (número de museos cercanos considerados)')
plt.ylabel('PageRank')
plt.xticks(m_values)
plt.grid(True)
plt.legend(title='Museos', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


plt.show()
"""

#ahora veamos cuando varia alpha:
    

ASeisSeptimos = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha1 = calculo_Page_Rank(ASeisSeptimos, 6/7, 5)
page_Rankalpha1 = page_Rankalpha1 / np.sum(page_Rankalpha1)

#print (page_Rankalpha1)



ACuatroQuintos = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha2 = calculo_Page_Rank(ACuatroQuintos, 4/5, 5)
page_Rankalpha2 = page_Rankalpha2 / np.sum(page_Rankalpha2)

#print (page_Rankalpha2)

ADosTercios = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha3 = calculo_Page_Rank(ADosTercios, 2/3, 5)
page_Rankalpha3 = page_Rankalpha3 / np.sum(page_Rankalpha3)

#print (page_Rankalpha3) 


AUnMedio = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha4 = calculo_Page_Rank(AUnMedio, 1/2, 5)
page_Rankalpha4 = page_Rankalpha4 / np.sum(page_Rankalpha4)

#print (page_Rankalpha4) 

# me dio fiaca hacer una para cada una de las q faltaba pero las tengo en un doc







