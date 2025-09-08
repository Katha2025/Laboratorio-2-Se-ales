# Laboratorio 2 Señales
En esta práctica se trabajaron tres conceptos importantes del procesamiento de señales: la convolución, la correlación y la transformada de Fourier. Para empezar, se aplicó la convolución con señales definidas a partir de la cédula y el código de cada integrante, tanto a mano como en Python, pudiendo así analizar la respuesta de un sistema. Luego, se estudió la correlación cruzada, lo que ayudó a entender cómo se mide la similitud entre dos señales. Después, se generó una señal biológica, se digitalizó aplicando el criterio de Nyquist y se revisaron sus características estadísticas. Por último, se aplicó la transformada de Fourier, lo que permitió observar la señal en el dominio de la frecuencia y analizar su espectro.
# Parte A
En la primera parte de la prática se trabajó la convolución entre un sistema h[n], definido a partir de los dígitos del código de cada uno de los integrantes y una señal x[n] construida con los dígitos de la cédula. Inicialmente, se realizó el cálculo manual y se representó el resultado de forma gráfica. Posteriormente, el mismo procedimiento se repitió en Python, lo que permitió verificar los resultados obtenidos y generar las gráficas correspondientes.

**1. Luciana :**


**-Convolución manual**

<img width="784" height="312" alt="image" src="https://github.com/user-attachments/assets/fbff055f-ce87-4ee1-8e14-67eef69d1bb5" />

<img width="604" height="473" alt="image" src="https://github.com/user-attachments/assets/16c15482-96cc-4f21-a5cb-aa133520e31d" />


<img width="1599" height="149" alt="image" src="https://github.com/user-attachments/assets/19facc74-440c-46e8-8463-9d02a543c35d" />


**-Gráfica manual**


![Imagen de WhatsApp 2025-09-08 a las 14 51 13_6ec22eb1](https://github.com/user-attachments/assets/a7763568-ea37-4c1f-b011-8a63bf91bf0d)



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


**2. Ana María :**

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

**3. Kathalina :**


**-Convolución manual**


<img width="911" height="805" alt="image" src="https://github.com/user-attachments/assets/a65da8be-0b8d-486f-9771-54fdd8e98a21" />



**-Gráfica manual**


<img width="754" height="833" alt="image" src="https://github.com/user-attachments/assets/132dda20-4ae9-4e69-9233-d42d4848150e" />


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



<img width="214" height="672" alt="image" src="https://github.com/user-attachments/assets/5ca1d458-5c2c-4c78-b8fb-447c2e852f73" />
Imagen 8. Diagrama de flujo parte A

# Parte B 
En la segunda parte de la práctica se definieron dos señales y se calculó su correlación cruzada, con el objetivo de medir el grado de similitud entre ellas. Una vez obtenida la secuencia resultante, se representó gráficamente para analizar su comportamiento y se discutió en qué situaciones resulta útil aplicar este procedimiento dentro del procesamiento digital de señales.
Las señales cruzadas fueron:

<img width="443" height="69" alt="image" src="https://github.com/user-attachments/assets/b08cc3c9-efef-40bf-b183-dc08c7b8b651" />

**1. Correlación Cruzada entre ambas señales**

A continuación se presenta el código en python empleado para calcular la correlación de las señales:

```python
import numpy as np
import matplotlib.pyplot as plt

# Parámetros
Ts = 1.25e-3   # 1.25 ms
f = 100        # Hz
N = 9          # número de muestras
n = np.arange(N)
w0 = 2*np.pi*f*Ts  # frecuencia digital

# Definición de señales
x1 = np.cos(w0 * n)
x2 = np.sin(w0 * n)

print("x1[n] =", np.round(x1, 4))
print("x2[n] =", np.round(x2, 4))

# Correlación cruzada
r12 = np.correlate(x1, x2, mode='full')
lags = np.arange(-N+1, N)

print("\nCorrelación cruzada r12[l]:")
for l, val in zip(lags, r12):
    print(f"l={l:2d}, r12={val:.4f}")

# Gráfica
plt.figure(figsize=(7,4))
plt.stem(lags, r12, basefmt="r-")
plt.xlabel("Retardo l (muestras)")
plt.ylabel("r12[l]")
plt.title("Correlación cruzada entre x1[n] y x2[n]")
plt.grid(True)
plt.show()
```
Ahora bien, estos fueron los resultados obtenidos:

<img width="819" height="468" alt="image" src="https://github.com/user-attachments/assets/17e280a0-0c94-4f31-874e-5423938e5abd" />


**2. Gráfica y secuencia resultante**

<img width="830" height="497" alt="image" src="https://github.com/user-attachments/assets/82ab92f4-d5f0-4469-9211-b9d47a09deae" />

La correlación cruzada entre las señales X_1[n] y X_2 [n] produjo una secuencia con valores positivos y negativos que dependen del retardo 𝑙, en la gráfia se puede observar que los picos más altos aparecen alrededor de 𝑙=-2 e 𝑙=2, lo que indica que las señales presentan una fuerte similitud cuando una se desplaza respecto a la otra en esas posiciones. Además, la secuencia muestra un comportamiento casi simétrico con respecto al origen, lo que significa que la similitud se manifiesta tanto para retardos positivos como para retardos negativos. En conclusión, el resultado evidencia que las dos señales comparten características comunes, aunque estas se presentan en distintos desfases temporales.


**3. Importancia de la correlación cruzada en el procesamiento digital de señales**

La correlación cruzada es útil porque permite identificar similitudes, patrones o retrasos entre señales, lo que resulta esencial en aplicaciones prácticas. Por ejemplo, en el análisis de señales biológicas puede ayudar a detectar ritmos o sincronizaciones; en comunicaciones, permite encontrar la llegada de una señal en presencia de ruido; y en el procesamiento de imágenes, sirve para reconocer patrones o coincidencias. En este caso, su aplicación permitió evidenciar cómo dos señales relacionadas pueden compararse y medirse en función de su desplazamiento temporal.

<img width="251" height="860" alt="image" src="https://github.com/user-attachments/assets/a3b867c3-625b-4e6b-9907-c31ec2627009" />
Imagen 9. Diagrama de flujo parte B

# Parte C

En la ultima parte de esta práctica, se capto una señal EOG (electrooculograma) del generador de señales con ayuda del DAQ. Para esto, se utilizó el siguiente codigo:

```python
import nidaqmx                     # Librería daq
from nidaqmx.constants import AcquisitionType 
import matplotlib.pyplot as plt    # Librería para graficar
import numpy as np                 # Librería de funciones matemáticas

#%% Adquisición de la señal por tiempo definido
fs = 400           # Frecuencia de muestreo en Hz
duracion = 5        # Periodo por el cual desea medir en segundos
senal = []          # Vector vacío en el que se guardará la señal
dispositivo = 'Dev1/ai0' # Nombre del dispositivo/canal 
total_muestras = int(fs * duracion)

with nidaqmx.Task() as task:
    # Configuración del canal
    task.ai_channels.add_ai_voltage_chan(dispositivo)
    # Configuración del reloj de muestreo
    task.timing.cfg_samp_clk_timing(
        fs,
        sample_mode=AcquisitionType.FINITE,   # Adquisición finita
        samps_per_chan=total_muestras        # Total de muestras que quiero
    )

    # Lectura de todas las muestras de una vez
    senal = np.array(task.read(number_of_samples_per_channel=total_muestras))

t = np.arange(len(senal))/fs # Crea el vector de tiempo 
plt.plot(t,senal)
plt.axis([0,duracion,min(senal),max(senal)])
plt.grid()
plt.title(f"fs={fs}Hz, duración={duracion}s, muestras={len(senal)}")
plt.show()
np.savetxt('Senal_lab_2.txt', [t,  senal])
```

Para esta señal, se determinó la frecuencia de Nyquist. Se utilizó la frecuencia máxima de la señal EOG que es de aproximadamente de 50 Hz, esta se multiplicó por 2 para obtener la frecuencia de Nyquist (50 Hz* 2 = 100 Hz). Para completar el ejercicio, se digitalizó la señal usando una frecuencia de muestreo de 4 veces la frecuancia de Nyquist (400 Hz). La grafica de la señal en una duración de 5 se gundos, frecuencia de muestreo de 400 Hz y 2000 muestras dio de la siguiente manera:

<img width="578" height="455" alt="partec" src="https://github.com/user-attachments/assets/23365eb9-69be-4139-831e-91bc92424178" />

Siguiendo, se caracterizó la señal por su media, mediana, desviación estandar, valor máximo y valor mínimo. 

```python
import numpy as np
import matplotlib.pyplot as plt

#media
media=np.mean(voltaje)
print (f"Media: {media} ")

#mediana
mediana=np.median(voltaje)
print (f"Mediana: {mediana} ")

#desviación estandar
desviacion_muestra=np.std(voltaje, ddof=1)
print (f"Desviacion estandar: {desviacion_muestra} ")

print(f"Máximo:", np.max(voltaje))
print(f"Mínimo:", np.min(voltaje))
```
Los valores fueron: 
* Media: 0.00608
* Mediana: -0.0561
* Desviacion estandar: 0.389
* Máximo: 1.268
* Mínimo: -1.573

La señal se puede clasificar como aleatoria, ya que depende de movimientos oculares y ruido fisiológico, y aunque un movimiento ocular repetido puede generar una onda predecible, la señal EOG es biológica y puede experimentar cambios a causa de elementos como la respuesta individual del paciente. Es aperiódica porque los movimientos oculares no ocurren en ciclos regulares. Y por ultimo, a pesar de que se digitaliza al muestrearla, se representa una señal originalmente análoga tomada con DAQ.

Para finalizar, se aplicó la transformada de Fourier: 

```python
import numpy as np
import matplotlib.pyplot as plt

#centra señal en cero
x = voltaje - np.mean(voltaje)
#numero de muestras y frecuencia de muestreo
N = len(x)
fs = 400

#funcion de transformada rapida de Fourier, rfft porque la señal es real y para devolver las frecuencias positivas
X = np.fft.rfft(x)
#vector de frecuencias para cada valor de la transformada, d=1/fs es el período de muestreo
f = np.fft.rfftfreq(N, d=1/fs)

#margnitud del espectro
mag = np.abs(X) / N

#grafica
plt.figure()
plt.plot(f, mag, color="skyblue")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Transformada de Fourier")
plt.grid(True)
plt.show()
```
 <img width="576" height="455" alt="fft" src="https://github.com/user-attachments/assets/163fc91e-5a73-426a-9537-83a0ebd79898" />

En adición, se grafico la densidad espectral de potencia:

```python
import numpy as np
import matplotlib.pyplot as plt

#centra señal en cero
x = voltaje - np.mean(voltaje)
#numero de muestras y frecuencia de muestreo
N = len(x)
fs = 400

#funcion de transformada rapida de Fourier, rfft porque la señal es real y para devolver las frecuencias positivas
X = np.fft.rfft(x)
#vector de frecuencias para cada valor de la transformada, d=1/fs es el período de muestreo
f = np.fft.rfftfreq(N, d=1/fs)

#margnitud del espectro
mag = np.abs(X) / N

#grafica
plt.figure()
plt.plot(f, mag, color="skyblue")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Transformada de Fourier")
plt.grid(True)
plt.show()
```

<img width="702" height="393" alt="espectro" src="https://github.com/user-attachments/assets/0a73faaa-ae45-4a2c-aee4-1c91fa2fb582" />

Se computaron los estadísticos de media, mediana y desviación estandar en el dominio de la frecuencia:
```python
#media
f_media = np.sum(f * mag) / np.sum(mag)
print(f"Frecuencia media: {f_media} Hz")

#mediana
mag_acumulativa = np.cumsum(mag)
total_mag = mag_acumulativa[-1]
f_mediana = f[np.searchsorted(mag_acumulativa, total_mag / 2)]
print(f"Frecuencia mediana: {f_mediana} Hz")

#desviacion estandar
f_std = np.sqrt(np.sum(mag * (f - f_media)**2) / np.sum(mag))
print(f"Desviación estándar: {f_std} Hz")
```
Los valores fueron:
* Frecuencia media: 57.507 Hz
* Frecuencia mediana: 33.0 Hz
* Desviación estándar: 58.975 Hz

De igual manera, se realizó un histograma de frecuancias:
```python
plt.figure(figsize=(8, 4))
plt.hist(f, bins=50, weights=mag, color='lightgreen', edgecolor='green')
plt.title("Histograma de Frecuencias")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud acumulada")
plt.grid(True)
plt.show()
```
<img width="700" height="393" alt="histo" src="https://github.com/user-attachments/assets/46438c40-a621-4f1d-a000-8d281ae6e898" /><br>



<img width="233" height="899" alt="image" src="https://github.com/user-attachments/assets/212c9ed1-f483-4e37-8773-14e34092588e" />
Imagen 10. Diagrama de flujo parte C
