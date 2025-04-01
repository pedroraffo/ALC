#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 12:44:45 2025

@author: Estudiante
"""

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt 

#1

T = np.array([[2, 0], [0, 3]])
t = np.array([[0.5, 0], [0, 1/3]])

w = 2
z = 3 

f =np.array[[w], [z]]

I = T @ t
x = T @ f

 