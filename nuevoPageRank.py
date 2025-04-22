#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 12:57:19 2025

@author: Estudiante
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

def calculaLU(A):
    
    # matriz es una matriz de NxN
    # Retorna la factorización LU a través de una lista con dos matrices L y U de NxN.
    
    def construir_P(A):
        
        n = A.shape[0]
        P = np.eye(n) # comentar aca
        A_permutada = A.copy()
    
        for k in range(n):
            #Tomamos los valores de la columna k desde la fila k  hasta el final
            columna = A_permutada[k:, k]
    
            #Hacemos que todos los valores de la columna sean su absoluto
            largo_columna_abs = np.abs(columna)
    
            #Buscamos el indice de la columna al que le pertenece el valor mas grande
            max_indice_columna = 0
            maxValor = largo_columna_abs[0]
    
            for i in range(1, len(columna)):
    
                if largo_columna_abs[i] > maxValor:
                    maxValor = largo_columna_abs[i]
                    max_indice_columna = i
    
            #Calculamos el indice correcto de la fila en A
            p = k + max_indice_columna
    
    
            # Intercambiamos filas en A_permutada y en P si es necesario
            if p != k:
    
                #Intercambiamos en A_copia
                A_permutada[[k, p], :] = A_permutada[[p, k], :]
    
                #Intercambiamos en P
                P[[k, p], :] = P[[p, k], :]
    
        return P, A_permutada
    
    
    P, A_permutada = construir_P(A) #Consigo la P, y en caso de que P != I la A con la filas reordenadas
    m = A.shape[0]
    n = A.shape[1]
    
    if m!=n:
        print('Matriz no cuadrada')
        return
    
    U = A_permutada # Comienza siendo una copia de A y deviene en U (triangulado superiormente)
    L = np.identity(n)  # comentar aca !!!
    
    

    for j in range(n):
        for i in range(j+1,n):
            L[i,j] = U[i,j] / U[j,j]
            U[i,:] = U[i,:] - L[i,j] * U[j,:]

    return L, U, P






# Calculo de la inversa usando descomposicion de LU para cualquier matriz inversible

def inversa_por_lu(A):
    n = A.shape[0]

    # Realizamos la factorización LU de la matriz A
    L, U, P = calculaLU(A)

    # Inicializamos la matriz identidad I
    I = np.eye(n)
    A_inv = np.zeros_like(A, dtype=float) #comentar aca !!!!

    # Resolvemos para cada columna de la matriz inversa
    for i in range(n):

        b = I[:, i]  # La columna i de la identidad

        # Resolvemos L y U
        y = solve_triangular(L, P @ b, lower=True)

        x = solve_triangular(U, y)

        A_inv[:, i] = x  # Guardamos el resultado en la columna i de A_inv

    return A_inv




def calcula_matriz_C(A):

    # Primero creo K a partir de la matriz A

    def crearK (A):

        n = A.shape[0]
        m = A.shape[1]
        K = np.zeros((m, n))
        sumaFilasA = np.sum(A, axis = 1)


        for i in range (len (sumaFilasA)):
            K[i, i] = sumaFilasA[i]

        return K

    # Nuestra matriz de transicion esta definida por A_traspuesta y K_inv, como nos pide la ecuacion (2)

    A_traspuesta = np.transpose(A) # Trasponemos A
    K = crearK(A) # Creamos K
    K_inv = inversa_por_lu(K) # Calcula inversa de la matriz K, que tiene en su diagonal la suma por filas de A
    C = A_traspuesta @ K_inv # Calcula C multiplicando A_traspuesta y K_inv

    return C



def calcula_pagerank(A,alpha):

    # Genero la matriz de transicion en base a A y una identidad en base a n de A

    n = A.shape[0]  # Dimension de A
    C = calcula_matriz_C(A)
    N = n   # Obtenemos el número de museos N a partir de la estructura de la matriz A
    I = np.eye(n)

    # Variamos la ecuacion dad, ya que tenemos una formula general para invertir con LU

    M = (N/alpha) * ( I - (1-alpha) * C )
    b = np.ones(n)  # Vector de 1s, multiplicado por el coeficiente correspondiente usando d y N.

    p = inversa_por_lu(M) @ b

    return p




# Aca variamos los datos, con m y alpha
"""
m = 1 # Cantidad de links por nodo
alpha = 1/5 # Cantidas de conexiones

A = construye_adyacencia(D,m) # Construimos la matriz de adyacencia

page_Rank = calcula_pagerank(A, alpha) # Realizamos el calculo

page_Rank = page_Rank / np.sum(page_Rank) # Normalizamos para hacer mas viable la visualizacion
"""

A1 = construye_adyacencia(D,1) # Construimos la matriz de adyacencia

page_Rank1 = calcula_pagerank(A1, 0.2)
page_Rank1 = page_Rank1 / np.sum(page_Rank1) # Normalizamos para hacer mas viable la visualizacion

#print(page_Rank1) 

#Nprincipales = 3
#principales = np.argsort(page_Rank1)[-Nprincipales:]

#print (principales)  

A3 = construye_adyacencia(D,3) # Construimos la matriz de adyacencia

page_Rank3 = calcula_pagerank(A3, 0.2)
page_Rank3 = page_Rank3 / np.sum(page_Rank3)

#print (page_Rank3)

A5 = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rank5 = calcula_pagerank(A5, 0.2)
page_Rank5 = page_Rank5 / np.sum(page_Rank5)

#print (page_Rank5)

A10 = construye_adyacencia(D,10) # Construimos la matriz de adyacencia

page_Rank10 = calcula_pagerank(A10, 0.2)
page_Rank10 = page_Rank10 / np.sum(page_Rank10)


#print (page_Rank10)
"""
# Construccion del Mapa sin nada

fig, ax = plt.subplots(figsize=(12, 12))
barrios.to_crs("EPSG:22184").boundary.plot(color='gray',ax=ax) # Graficamos Los barrios


# Armado del mapa

factor_escala = 2e4  # Escalamos los nodos para que sean visibles "(esto puedo variar)" !!!


# Construccion del mapa de redes
G = nx.from_numpy_array(A) # Construimos la red a partir de la matriz de adyacencia

# Construimos un layout a partir de las coordenadas geográficas
G_layout = {i:v for i,v in enumerate(zip(museos.to_crs("EPSG:22184").get_coordinates()['x'],museos.to_crs("EPSG:22184").get_coordinates()['y']))}


Nprincipales = 3 # Cantidad de principales
principales = np.argsort(page_Rank)[-Nprincipales:] # Identificamos a los N principales

# Imprimir información sobre los 3 museos principales
print("Los 3 museos con mayor PageRank son:", '\n')
for i, idx in enumerate(principales[::-1]):  # Invertir para mostrar en orden descendente
    print(f"{i+1}. Museo {idx}: PageRank = {page_Rank[idx]:.6f}", '\n')

labels = {n: str(n) if i in principales else "" for i, n in enumerate(G.nodes)} # Nombres para esos nodos


# Graficamos red

nx.draw_networkx(G,G_layout,
                 node_size = page_Rank*factor_escala,
                 node_color = page_Rank,
                 cmap = plt.cm.viridis,
                 ax=ax,
                 with_labels=False)
nx.draw_networkx_labels(G, G_layout, labels=labels, font_size=10, font_color="k") # Agregamos los nombres


sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(page_Rank), vmax=max(page_Rank)))
sm._A = []
cbar = plt.colorbar(sm, ax=ax, shrink=0.5)
cbar.set_label("PageRank")

# Añadir título y leyenda
plt.title(f'Red de Museos - PageRank (m = {m}, α = {alpha})')
plt.axis('off')


plt.show()


page_rank = {
    15 : [0.006458, 0.003866, 0.010792, 0.018291],
    18 : [0.009967, 0.021280, 0.019525, 0.016743],
    34 : [0.021287, 0.021280, 0.005459, 0.008134],
    65 : [0.022124, 0.024135, 0.019525, 0.016970],
    93 : [0.014150, 0.011385, 0.021736, 0.017335],
    107: [0.021705, 0.023529, 0.021614, 0.010335],
    117: [0.017941, 0.024135, 0.021065, 0.018290],
    124: [0.018835, 0.019692, 0.019307, 0.018665], 
    125: [0.003823, 0.023529, 0.021614, 0.016571], 
    135: [0.010620, 0.020145,  0.019525, 0.018338]	
    
    }
"""

 #ahora veamos cuando varia alpha:
    

ASeisSeptimos = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha1 = calcula_pagerank(ASeisSeptimos, 6/7)
page_Rankalpha1 = page_Rankalpha1 / np.sum(page_Rankalpha1)

#print (page_Rankalpha1)



ACuatroQuintos = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha2 = calcula_pagerank(ACuatroQuintos, 4/5)
page_Rankalpha2 = page_Rankalpha2 / np.sum(page_Rankalpha2)

#print (page_Rankalpha2)

ADosTercios = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha3 = calcula_pagerank(ADosTercios, 2/3)
page_Rankalpha3 = page_Rankalpha3 / np.sum(page_Rankalpha3)

#print (page_Rankalpha3) 


AUnMedio = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha4 = calcula_pagerank(AUnMedio, 1/2)
page_Rankalpha4 = page_Rankalpha4 / np.sum(page_Rankalpha4)

#print (page_Rankalpha4) 

AUnTercio = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha5 = calcula_pagerank(AUnTercio, 1/3)
page_Rankalpha5 = page_Rankalpha5 / np.sum(page_Rankalpha5)

#print (page_Rankalpha5) 

AUnQuinto = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha6 = calcula_pagerank(AUnQuinto, 1/5)
page_Rankalpha6 = page_Rankalpha6 / np.sum(page_Rankalpha6)

#print (page_Rankalpha6) 

AUnSeptimo = construye_adyacencia(D,5) # Construimos la matriz de adyacencia

page_Rankalpha7 = calcula_pagerank(AUnSeptimo, 1/7)
page_Rankalpha7 = page_Rankalpha7 / np.sum(page_Rankalpha7)

#print (page_Rankalpha7) 

listaDieciocho : list = []
listaDieciocho.append(float(page_Rankalpha7[18]))
listaDieciocho.append(float(page_Rankalpha6[18]))
listaDieciocho.append(float(page_Rankalpha5[18]))
listaDieciocho.append(float(page_Rankalpha4[18]))
listaDieciocho.append(float(page_Rankalpha3[18]))
listaDieciocho.append(float(page_Rankalpha2[18]))
listaDieciocho.append(float(page_Rankalpha1[18])) 


listaDieciocho = [round(x, 6) for x in listaDieciocho]
#print(listaDieciocho) 

listaNOVENTAYTRES : list = []
listaNOVENTAYTRES.append(float(page_Rankalpha7[93]))
listaNOVENTAYTRES.append(float(page_Rankalpha6[93]))
listaNOVENTAYTRES.append(float(page_Rankalpha5[93]))
listaNOVENTAYTRES.append(float(page_Rankalpha4[93]))
listaNOVENTAYTRES.append(float(page_Rankalpha3[93]))
listaNOVENTAYTRES.append(float(page_Rankalpha2[93]))
listaNOVENTAYTRES.append(float(page_Rankalpha1[93])) 


listaNOVENTAYTRES = [round(x, 6) for x in listaNOVENTAYTRES]
#print(listaNOVENTAYTRES) 

listaCIENTOSIETE : list = []
listaCIENTOSIETE.append(float(page_Rankalpha7[107]))
listaCIENTOSIETE.append(float(page_Rankalpha6[107]))
listaCIENTOSIETE.append(float(page_Rankalpha5[107]))
listaCIENTOSIETE.append(float(page_Rankalpha4[107]))
listaCIENTOSIETE.append(float(page_Rankalpha3[107]))
listaCIENTOSIETE.append(float(page_Rankalpha2[107]))
listaCIENTOSIETE.append(float(page_Rankalpha1[107])) 


listaCIENTOSIETE = [round(x, 6) for x in listaCIENTOSIETE]
#print(listaCIENTOSIETE) 

listaCIENTODIECISIETE : list = []
listaCIENTODIECISIETE.append(float(page_Rankalpha7[117]))
listaCIENTODIECISIETE.append(float(page_Rankalpha6[117]))
listaCIENTODIECISIETE.append(float(page_Rankalpha5[117]))
listaCIENTODIECISIETE.append(float(page_Rankalpha4[117]))
listaCIENTODIECISIETE.append(float(page_Rankalpha3[117]))
listaCIENTODIECISIETE.append(float(page_Rankalpha2[117]))
listaCIENTODIECISIETE.append(float(page_Rankalpha1[117])) 


listaCIENTODIECISIETE = [round(x, 6) for x in listaCIENTODIECISIETE]
#print(listaCIENTODIECISIETE) 

listaCIENTOVEINTICINCO : list = []
listaCIENTOVEINTICINCO.append(float(page_Rankalpha7[125]))
listaCIENTOVEINTICINCO.append(float(page_Rankalpha6[125]))
listaCIENTOVEINTICINCO.append(float(page_Rankalpha5[125]))
listaCIENTOVEINTICINCO.append(float(page_Rankalpha4[125]))
listaCIENTOVEINTICINCO.append(float(page_Rankalpha3[125]))
listaCIENTOVEINTICINCO.append(float(page_Rankalpha2[125]))
listaCIENTOVEINTICINCO.append(float(page_Rankalpha1[125])) 


listaCIENTOVEINTICINCO = [round(x, 6) for x in listaCIENTOVEINTICINCO]
#print(listaCIENTOVEINTICINCO) 

listaCIENTOTREINTAYCINCO : list = []
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha7[135]))
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha6[135]))
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha5[135]))
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha4[135]))
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha3[135]))
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha2[135]))
listaCIENTOTREINTAYCINCO.append(float(page_Rankalpha1[135])) 


listaCIENTOTREINTAYCINCO = [round(x, 6) for x in listaCIENTOTREINTAYCINCO]
print(listaCIENTOTREINTAYCINCO) 







    
   


