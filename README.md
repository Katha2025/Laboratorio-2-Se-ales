# Laboratorio-2-Señales
En esta práctica se trabajaron tres conceptos importantes del procesamiento de señales: la convolución, la correlación y la transformada de Fourier. Para empezar, se aplicó la convolución con señales definidas a partir de la cédula y el código de cada integrante, tanto a mano como en Python, pudiendo así analizar la respuesta de un sistema. Luego, se estudió la correlación cruzada, lo que ayudó a entender cómo se mide la similitud entre dos señales. Después, se generó una señal biológica, se digitalizó aplicando el criterio de Nyquist y se revisaron sus características estadísticas. Por último, se aplicó la transformada de Fourier, lo que permitió observar la señal en el dominio de la frecuencia y analizar su espectro.
# Parte A
En la primera parte de la prática se trabajó la convolución entre un sistema h[n], definido a partir de los dígitos del código de cada uno de los inetgrantes y una señal x[n] construida con los dígitos de la cédula. Inicialmente, se realizó el cálculo manual y se representó el resultado de forma gráfica. Posteriormente, el mismo procedimiento se repitió en Python, lo que permitió verificar los resultados obtenidos y generar las gráficas correspondientes.

**Luciana :**

**Convolución manual**

<img width="784" height="312" alt="image" src="https://github.com/user-attachments/assets/fbff055f-ce87-4ee1-8e14-67eef69d1bb5" />

**Gráfica manual**

<img width="631" height="819" alt="image" src="https://github.com/user-attachments/assets/4ec6cf9a-b838-4265-b21a-c8ddfe83cfb4" />

**Convolución y gráfica en python**

```python
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Señales de entrada
# -------------------------------
h = [5, 6, 0, 0, 8, 8, 2]  # Código
x = [4, 5, 4, 7, 0, 3]  # Cédula

# Convolución
y = np.convolve(x, h)

# Mostrar valores secuenciales

print("Valores de y[n]:")
for i, val in enumerate(y):
    print(f"y[{i}] = {val}")

# -------------------------------
# Graficar señal
# -------------------------------
n = np.array([0,1,2,3,4,5,6,7,8,9,10,11])
plt.stem(n,y)
# Índices de tiempo
plt.title("Señal y[n] = x[n] * h[n]")
plt.xlabel("n (índice)")
plt.ylabel("y[n]")
plt.grid(True)
plt.tight_layout()
plt.show()
