# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 0.5
r = 0.1
sigma = 0.5
N = 100
deltat = T / N
Nmc = 1000

W = np.zeros((N + 1, Nmc))  # Brownian Motion
S = np.zeros((N + 1, Nmc))  # Underlying Asset
finalvalueW = np.zeros(Nmc + 1)
finalvalueS = np.zeros(Nmc + 1)
S[0, :] = 10  # Underlying Asset Initial Value
t = np.linspace(0, T, N + 1)  

# Simulation
for k in range(Nmc):
    for i in range(1, N + 1):
        g = np.random.randn()
        W[i, k] = W[i-1, k] + g * np.sqrt(deltat)
        S[i, k] = S[i-1, k] * (1 + r * deltat + sigma * g * np.sqrt(deltat))
    finalvalueW[k] = W[N, k]
    finalvalueS[k] = S[N, k]

        
Esperance_W = np.mean(finalvalueW)
Variance_W = np.var(finalvalueW)
Esperance_S = np.mean(finalvalueS)
Variance_S = np.var(finalvalueS)

print(f'Esperance W: {Esperance_W}')
print(f'Variance W: {Variance_W}')
print(f'Esperance S: {Esperance_S}')
print(f'Variance S: {Variance_S}')

plt.figure()
for k in range(Nmc):
    plt.plot(t, W[:, k])
plt.title('Brownian Motion Paths')
plt.xlabel('Time')
plt.ylabel('W(t)')

plt.figure()
for k in range(Nmc):
    plt.plot(t, S[:, k])
plt.title('Underlying Asset Price')
plt.xlabel('Time')
plt.ylabel('S(t)')

plt.show()
