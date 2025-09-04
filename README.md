# Laboratorio 2 Señales
En esta práctica se trabajaron tres conceptos importantes del procesamiento de señales: la convolución, la correlación y la transformada de Fourier. Para empezar, se aplicó la convolución con señales definidas a partir de la cédula y el código de cada integrante, tanto a mano como en Python, pudiendo así analizar la respuesta de un sistema. Luego, se estudió la correlación cruzada, lo que ayudó a entender cómo se mide la similitud entre dos señales. Después, se generó una señal biológica, se digitalizó aplicando el criterio de Nyquist y se revisaron sus características estadísticas. Por último, se aplicó la transformada de Fourier, lo que permitió observar la señal en el dominio de la frecuencia y analizar su espectro.
# Parte A
En la primera parte de la prática se trabajó la convolución entre un sistema h[n], definido a partir de los dígitos del código de cada uno de los inetgrantes y una señal x[n] construida con los dígitos de la cédula. Inicialmente, se realizó el cálculo manual y se representó el resultado de forma gráfica. Posteriormente, el mismo procedimiento se repitió en Python, lo que permitió verificar los resultados obtenidos y generar las gráficas correspondientes.

**Luciana :**


**-Convolución manual**

<img width="784" height="312" alt="image" src="https://github.com/user-attachments/assets/fbff055f-ce87-4ee1-8e14-67eef69d1bb5" />

<img width="604" height="473" alt="image" src="https://github.com/user-attachments/assets/16c15482-96cc-4f21-a5cb-aa133520e31d" />


<img width="1599" height="149" alt="image" src="https://github.com/user-attachments/assets/19facc74-440c-46e8-8463-9d02a543c35d" />


**-Gráfica manual**

<img width="631" height="819" alt="image" src="https://github.com/user-attachments/assets/4ec6cf9a-b838-4265-b21a-c8ddfe83cfb4" />

**-Convolución y gráfica en python**

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
```

<img width="303" height="286" alt="image" src="https://github.com/user-attachments/assets/36337bfa-5a1a-4867-bda0-d9e12519e6c8" />

<img width="934" height="562" alt="image" src="https://github.com/user-attachments/assets/019e0ef5-93ee-42ed-bf00-2740244ca783" />

**Ana María**

**-Convolución manual**

<img width="755" height="239" alt="image" src="https://github.com/user-attachments/assets/4c1aea4d-8c66-45b8-97fe-9921c7cd68d8" />

<img width="1445" height="715" alt="image" src="https://github.com/user-attachments/assets/e18fc353-27df-4323-9046-1d3c25eb116f" />


**-Gráfica manual**


<img width="843" height="724" alt="image" src="https://github.com/user-attachments/assets/c2da7938-831e-4d52-9476-42d81958d448" />


**-Convolución y gráfica en python**

```python
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Señales de entrada
# -------------------------------
h = [5, 6, 0, 0, 8, 7, 0]  # Código
x = [1, 0, 7, 2, 6, 4, 3, 3, 6, 5]  # Cédula

# Convolución
y = np.convolve(x, h)

# Mostrar valores secuenciales

print("Valores de y[n]:")
for i, val in enumerate(y):
    print(f"y[{i}] = {val}")

# -------------------------------
# Graficar señal
# -------------------------------
n = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
plt.stem(n,y)
# Índices de tiempo
plt.title("Señal y[n] = x[n] * h[n]")
plt.xlabel("n (índice)")
plt.ylabel("y[n]")
plt.grid(True)
plt.tight_layout()
plt.show()
```
<img width="493" height="377" alt="image" src="https://github.com/user-attachments/assets/5fafa07d-4103-4169-8888-4a4b723bdb4e" />

<img width="864" height="587" alt="image" src="https://github.com/user-attachments/assets/996a2343-14fd-4edf-aa4a-98eb100c3fe5" />

**Kathalina**


**-Convolución manual**




**-Gráfica manual**


**-Convolución y gráfica en python**

```python
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Señales de entrada
# -------------------------------
h = [5, 6, 0, 0, 8, 7, 5]  # Código
x = [1, 0, 9, 6, 9, 4, 7, 8, 4, 4]  # Cédula

# Convolución
y = np.convolve(x, h)

# Mostrar valores secuenciales

print("Valores de y[n]:")
for i, val in enumerate(y):
    print(f"y[{i}] = {val}")

# -------------------------------
# Graficar señal
# -------------------------------
n = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
plt.stem(n,y)
# Índices de tiempo
plt.title("Señal y[n] = x[n] * h[n]")
plt.xlabel("n (índice)")
plt.ylabel("y[n]")
plt.grid(True)
plt.tight_layout()
plt.show()
```

<img width="320" height="383" alt="image" src="https://github.com/user-attachments/assets/d76f9eac-adc9-4806-809b-183c8f951af3" />

<img width="892" height="596" alt="image" src="https://github.com/user-attachments/assets/64b59765-26e9-4ec6-b4df-ea809cf5db9f" />
