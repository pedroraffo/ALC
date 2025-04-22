# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 20:11:13 2025

@author: pedro
"""


import matplotlib.pyplot as plt

alpha_values = [1/7, 1/5, 1/3, 1/2, 2/3, 4/5, 6/7]

# Hago un dict de los page rank de los museos con mayor promedio para todos los m

page_rank = {
    18  : [0.021983, 0.019526, 0.015685, 0.012585, 0.01039, 0.009013, 0.008496],
    93  : [0.025896, 0.021736, 0.01623, 0.012601, 0.010339, 0.008985, 0.00848],
    107 : [0.021478, 0.019299, 0.015848, 0.012931, 0.010727, 0.009256, 0.008681], 
    117 : [0.025274, 0.021066, 0.015556, 0.012024, 0.009915, 0.008715, 0.008282], 
    125 : [0.025781, 0.021614, 0.016112, 0.012513, 0.010291, 0.008965, 0.00847], 
    135 : [0.021983, 0.019526, 0.015685, 0.012585, 0.01039, 0.009013, 0.008496]



}

labels_frac = [r"$\frac{1}{7}$", r"$\frac{1}{5}$", r"$\frac{1}{3}$", r"$\frac{1}{2}$", 
               r"$\frac{2}{3}$", r"$\frac{4}{5}$", r"$\frac{6}{7}$"]



plt.figure(figsize=(8, 5))

for museo_id, valores in page_rank.items():
    plt.plot(alpha_values, valores, marker='o', label=f'Museo {museo_id}')

plt.title('Evolución del PageRank de Museos según alpha')
plt.xlabel('alpha')
plt.ylabel('PageRank')
plt.xticks(alpha_values, labels_frac)
plt.grid(True)
plt.legend(title='Museos', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()


plt.show()

