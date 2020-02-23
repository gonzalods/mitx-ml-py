# Regresión Lineal
Al final de esta lección, podrás
- escribir el error de entrenamiento como criterio de mínimos cuadrados para regresión lineal.
- utilizar el gradiente descendente estocástico para ajustar modelos de regresión lineal.
- resolver solución de regresión lineal de forma cerrada.
- identificar el término de regularización y cómo cambia la solución, generalización
## Intruducción
Vamos a recordar el rol de la clasificación para recordar el marco de trabajo general supervisado
- Se proporciona un **conjunto de datos de entrenamiento** que son un conjunto de pares compuestos por vectores de caracteristicas y sus etiquetas asociadas
$$S_n=\Big\{ \big(x^{(t)},y^{(t)}\big)|t=1\dots n \Big\}$$
- Donde el vector de caracteristicas $x^{(t)} \in \R^d$ y en el caso de una clasificación binaria $y^{(t)} \in \{-1, 1\}$.
- El objetivo del modelo es encotrar la **correspondencia** entre los vectores de caracteristicas y su etiqueta correspondiente.

Pero en muchos problemas no es suficiente un respuesta del tipo si-no o sobre un conjunto discreto finito. Y lo que se quiere es el grado en el que algo pueda pasar. Esto es un **problema de regresión** donde lo que interesa son **valores continuos**. En este caso:

- Se proporciona un **conjunto de datos de entrenamiento** que son un conjunto de pares compuestos por vectores de caracteristicas y sus valores asociadas

$$S_n=\Big\{ \big(x^{(t)},y^{(t)}\big)|t=1\dots n \Big\}$$

- Donde el vector de caracteristicas $x^{(t)} \in \R^d$ y los valores asociados son $y^{(t)} \in \R$
- El objetivo del modelo es encotrar la **correspondencia** entre los vectores de caracteristicas y su valor correspondiente. En este caso se esta buscando una conbinación lineal:
$$f(x;\theta, \theta_0)=\sum_{i=1}^d \theta_i x_i + \theta_0=\theta x + \theta_0$$

Para las demostraciones y explicaxiones siguientes se va a asumir que $\theta_0 = 0$.

Tres cuestiones surgen en la regresión lineal:
- Hay que formular la **función objetivo** que pueda cuantificar el grado de error.
- El **algoritmo de aprendizaje** para encontrar el mejor conjunto de parámetros. Se van a dar dos algoritmos:
    - Algoritmos basados en gradiente descendente
    - Algoritmo en forma cerrada.
- Noción de **regularización**. Es un término que permite corregir la **ausencia de suficientes datos de entrenamiento** o **el ruido** en los mismos.

## Función Objetivo: Riesgo Empírico
Intuitivamente, lo que se pretende hacer, el objetico, es cuantificar la desvicación de la predicción del valor conocido de $y$. Este objetivo se denomina **Riesgo empírico**, que se va a denominar $R_n(\theta)$. Es una función de los $n$ ejemplos de entrenamiento y depende de los parámetros o pesos $\theta$.

Se calcula la desviación de la predicción con respecto a su valor real $y$ en cada ejemplo de entrenamiento, mediante una **función de pérdida**, se suman estas pérdidas, y se promedia. La función de pérdida que se va a utilizar en el **error cuadrático**.

$$R_n(\theta) = \frac 1 n \sum_{t=1}^n \big(y^{(t)}-(\theta·x^{(t)}) \big)^2/2$$

La **función de pérdida es el error cuadrático** porque ya que los datos de entrenamiento puede contener ruido, si hay una pequeña desviación entre la predicción y el valor real, esto está bien, pero si hay una gran desvicación queremos que se penaliza mucho.

Dos tipos errores que pueden aparecer al optimizar el riesgo empírico:
1. **Error estructural**: Aparece cuando la función lineal no es suficiente para modelar los datos de entrenamiento. Es decir, la correspodencia entre los datos de entrenamiento y sus valores es altamente no lieneal. Independientemente de lo que se haga, se incurre en grandes errores.
2. **Error de estimación**: Aparece cuando aunque la correspondecia es lineal, no hay suficientes datos de entrenamiento para estimar correctamente. Cuantos más ejemplos de entrenamiento se tenga menos errores de estimación se hara.

Estos dos errores enpujan en direcciones diferentes. Por un lada, si se buscan correspondencia complejas para eliminar errores estructurales puede que se incurra en errores de estimación pues a correspondencias más complejas se necesitan mas ejemplos de entrenamiento para estimar bién. Por el otro lado, si se quiere un número minimo de parámetros que es bueno para el conjunto de datos se puede llegar a incurrir en errores estructurales.

### Descomposición del Error y Balance Bias-Varianza
Definición más formal de error estructural y de estimación.

Supongamos que se quiere conocer la relación entre las variables aleatorias $x \in \R^d$ e $y \in \R$, donde la relación verdadera es $Y=f(x)$. $f$ es desconocida, pero se observan un conjunto de entrenamiento $(x_1, y_1), (x_1, y_1), \dots, (x_n, y_n)$ y se pretende encontrar una función $\hat{f}$ que se aproxime a la función real $f$.

No obstante, los datos observados podrían no ser $\textbf{100\%}$ precisos debido a algún tipo de ruido o incertidumbre que contiene los datos. Por ello, además se asume que una variable de ruido aleatorio $\pmb{\epsilon}$ se añade a $\pmb{y}$:

$$y = f(x) + \epsilon$$

donde $\epsilon \backsim \bm{N}(0,\sigma^2)$.

Se ha visto que se puede encontrar $\hat{f}$ minimizando el riesgo empírico en el conjunto de entrenamiento. Se sabe que el conjunto de entrenamiento es una observación aleatoria de la relación subyacente y contine ruido, conjuntos de entrenamiento diferente darán estimadores $\hat{f}$ diferentes. Se puede definir $\Bbb{E}[\hat{f}(x)]$ como el estimador esperado sobre todos los conjuntos de entrenamiento posibles.

Ahora veamos cuando tenemos una nueva $x$ con $y$ desconocida, cuál es el error de predicción esperado para nuestro estimador dados todos los conjuntos de entrenamiento posibles:

$$\begin{aligned}\Bbb{E}[(y- \hat{f})^2] & = \Bbb{E}[(f+ \epsilon - \hat{f})^2] \\
&=\Bbb{E}[(f+ \epsilon - \hat{f} + \Bbb{E}[ \hat{f}] - \Bbb{E}[ \hat{f}])^2 \\
&=\Bbb{E}[(f - \Bbb{E}[\hat{f}])^2] + \Bbb{E}[\epsilon^2] +  \Bbb{E}[(\Bbb{E}[\hat{f}]-\hat{f})^2] + \\
&+2\Bbb{E}[(f-\Bbb{E}[\hat{f}])\epsilon] +2\Bbb{E}[\epsilon(\Bbb{E}[\hat{f}]-\hat{f})] + 2\Bbb{E}[(\Bbb{E}[\hat{f}]-\hat{f})(f-\Bbb{E}[\hat{f}])] \\
&=(f - \Bbb{E}[\hat{f}])^2 + \Bbb{E}[\epsilon^2] +  \Bbb{E}[(\Bbb{E}[\hat{f}]-\hat{f})^2] + \\
&+2\Bbb{E}[(f-\Bbb{E}[\hat{f}])\epsilon] +2\Bbb{E}[\epsilon(\Bbb{E}[\hat{f}]-\hat{f})] + 2\Bbb{E}[(\Bbb{E}[\hat{f}]-\hat{f})](f-\Bbb{E}[\hat{f}]) \\
&=(f - \Bbb{E}[\hat{f}])^2  +  \Bbb{E}[(\Bbb{E}[\hat{f}]-\hat{f})^2] + \Bbb{E}[\epsilon^2]
\end{aligned}$$

ya que como $f$ es determinista $\Bbb{E}[f] = f$
como $\epsilon \backsim \bm{N}(0,\sigma^2)$, entonces $\Bbb{E}[\epsilon] = 0$ y el cuarto y quinto termino es $0$
y como $\Bbb{E}[(\Bbb{E}[\hat{f}]-\hat{f})] = 0$, el último termino es $0$.

Se ve que hay tres términos en la descomposición del error cuadrático:

1. Es el **sesgo o bias al cuadrado**, $(f(x) - \Bbb{E}[\hat{f}(x)])^2$, describe en cuanto se desvia el estimador promedio encajado sobre todos los conjuntos de entrenamiento de la verdad subyacente $f(x)$. Corresponde al **error estructural**.
2. Es la **varianza del estimador**, $\Bbb{E}[(\Bbb{E}[\hat{f}(x)]-\hat{f}(x))^2]$. Describe en promedio cuanto un estimador se desvia del estimador esperado sobre todos los conjuntos de datos. Corresponde al **error de estimación**.
3.  Es el **error irreductible**, $\Bbb{E}[\epsilon^2]=\sigma^2$, inherente al ruido de los datos y que no se puede minimizar. Es un **límite inferior** al error de predicción esperado.

La tarea del aprendizaje supervisado es reducir el sesgo y la varianza al mismo tiempo, pero debido al ruido en los datos de entrenamiento, no es posible minimizar simultaneamente las dos fuentes de error. Esto es lo que es conocido como el **equilibrio sesgo-varianza** (bias-variance trade-off).

Para reducir el sesgo, podemos asumir un espacio de hipótesis más complejo y ajustar un modelo más potente. El modelo podrá adaptarse incluso al ruido en el conjunto de entrenamiento. Sin embargo, esto aumenta el error de la varianza debido a otro conjunto de entrenamiento, la aleatoriedad del ruido dará como resultado un modelo muy diferente. Esta situación a menudo se llama **sobreajuste** (overfitting).

Por otro lado, se puede tener un modelo más simple para reducir la varianza, pero esto puede hacer que el sesgo sea muy grande. Por ejemplo, en el caso extremo, sea el modelo $\hat{f}(x) = c$, donde $c$ es una constante. Esto nos dará una variación de $0$ pero puedes imaginar que difícilmente puede hacer predicciones correctas. Esta situación se conoce como **subajuste** (underfitting).

## Enfoque Basado en Gradiente
La buena noticia a cerca del riesgo empírico es que es diferenciable en cualquier punto. 

Loque se va a hacer es seleccinar un ejemplo aleatoriamente, mirar la dirección del gradiente y como se está intentando miminizar se empujan los parámetros ligeramente en la dirección correcta, que es la contraria a la del gradiente.

$$\begin{aligned}\nabla_{\theta} (y^{(t)} - \theta x^{(t)})^2/2&=(y^{(t)} - \theta x^{(t)})\nabla_{\theta}(y^{(t)} - \theta x^{(t)})\\
&= - (y^{(t)} - \theta x^{(t)})x^{(t)}
\end{aligned}$$

El **algoritmo implica**:
- un paso de inicialización de los parámetros $\theta=0$
- escoger aleatoriamente un ejemplo $t=\{1,\dots, n\}$
- actualizar los parámetros 
$$\boxed{\theta = \theta + \eta(y^{(t)} - \theta x^{(t)})x^{(t)}}$$ 

Donde $\eta$ es **la tasa de aprendizaje** que normalmente depende de los pasos ya realizados. Si se mira en cada iteración $k$, se puede calcular $\eta$ en función de $k$ de la siguente manera

$$\eta_k=\frac{1}{1+k}$$

A diferncia de la clasificación, este algoritmo siempre actualiza cuando hay una discrepacia, se corrige más cuanto más discrepancia hay y menos cuando la discrepancia es menor.

Por otro lado si la predicción es menor que el valor real, $y > \theta x$ entonces $y - \theta x$ es positiva y se empuja a los parámetros $\theta$ en la dirección positiva del gradiente.

## Solución en Forma Cerrada
Como la función de riesgo empirico es una función convexa se puede encontrar una solución en forma cerrada.

Partiend del riesgo empirico

$$R_n(\theta) = \frac 1 n \sum_{t=1}^n \big(y^{(t)}-(\theta·x^{(t)}) \big)^2/2$$

Obtenemos su gradiente

$$\begin{aligned}\nabla_{\theta}R_n(\theta)_{|\theta=\hat{\theta}}&=\frac 1 n \sum_{t=1}^n\nabla_{\theta}\Big[(y^{(t)}-\theta · x^{(t)})^2/2\Big]_{|\theta=\hat{\theta}}\\
&=-\frac 1 n \sum_{t=1}^n (y^{(t)}-\hat{\theta} · x^{(t)})x^{(t)}\\
&=-\frac 1 n \sum_{t=1}^n y^{(t)}x^{(t)} + \frac 1 n \sum_{t=1}^n \hat{\theta} · x^{(t)}x^{(t)}
\end{aligned}$$

El primer termino es el producto de un número $y^{(t)}$ por un vector $x^{(t)}$ y se le va a denotar por $b$:

$$b = \frac 1 n \sum_{t=1}^n y^{(t)}x^{(t)}$$

En el segundo término $\hat{\theta} · x^{(t)}$ es un producto escalar y su resultado es un número y vamos a colocar al final

$$\frac 1 n \sum_{t=1}^n x^{(t)}\hat{\theta} · x^{(t)}=\frac 1 n \sum_{t=1}^n \hat{\theta} · x^{(t)}x^{(t)}$$

Como el producto escalar es un escalar, su transpuesto es igual a si mismo y asi colocamos los $x^{(t)}$ uno junto al otro

$$\frac 1 n \sum_{t=1}^n x^{(t)}(x^{(t)})^T·\hat{\theta} =\frac 1 n \sum_{t=1}^n x^{(t)}\hat{\theta} · x^{(t)}$$

Como los $x^{(t)}$ son vectores de dimensión $d$, resulta que 

$$A = \frac 1 n \sum_{t=1}^nx^{(t)}(x^{(t)})^T$$

es una matriz de diemensiones $d\times d$

Asi que la función del gradiente se puede expresar como

$$\boxed{A\hat{\theta}-b=0 \rArr A\hat{\theta}=b}$$

Que tiene una **solución en forma cerrada** si la matriz $A$ es invertible que es de la forma

$$\boxed{\hat{\theta}=A^{-1}b}$$

Esta operación solo se puede hacer si la matriz $A$ es invertible, y es invertible si los vectores $\{x_1,\dots ,x_n\}$ soportan $\R^d$ y esto ocurre cuando el número de vectores $n$ es sustancialmente mayor que la dimensionalidad $d$ de los vactores $x$.

Otro problema al utilizar esta solución es el coste de realizar esta operación. Esta operación es del orden de $O(d^3)$

## Generalización y Regularización
Un  mecanismo muy potente y común para solucionar el problema de ausencia de suficientes ejemplos de entrenamiento o unos ejemplos de entrenamiento con algo de ruido es el mecanisnmo denominado **regularización**.

Intuitivamente, lo que la regularización hace es apartarnos de intentar el encaje perfecto a los datos de entrenamiento. Con la formulación del riesgo empirico visto hasta ahora se está intentando encontrar los parámetros $\theta$ que mejor encajan a los datos de entrenamiento, esto supone encajar todos los errores que hay en los datos de entrenamiento. 

Lo que hace la regularización es empujar a los parámetros $\theta$ cerca de cero.

## Regularización
Vamos a ver como se incorpora la regularización a la función objetivo, riesgo empírico. Ahora se denomina **regresión de cresta** (ridge regression) Se va a cambiar un poco la notación. Además de depender del número de ejemplos de entrenamiento $n$, también depende de un nuevo parámetro, denominado **parámetro de regularización** $\lambda$

$$J_{\lambda,n}(\theta)=\frac{\lambda}{2}\Vert\theta \Vert + R_n(\theta)$$

El riesgo empirico controla en cuanta perdida se incurre con los datos de entrenamiento. Intentar minimizar el riesgo empirico supone intentar encajar los datos de entrenamiento lo mejor posible (sobreajunte).

El término de regularización intenta mantener los parámetros $\theta$ lo mas próximos a cero. Minimizar la regularización supone minimizar la norma del vector de los parámetros $\theta$ (subajuste).

El rol del parámetro de regularización $\lambda$ es controlar el balance entre encajar los parámetros $\theta$ a los datos de entrenamiento $x$ para que se ajusten a los cambios significativos y mantener los parámetros $\theta$ cerca de cero sin perder esos cambios significativos.

Por lo tanto la nueva función objetivo regresión de cresta sera:

$$J_{\lambda,n}(\theta)=\frac{\lambda}{2}\Vert\theta \Vert + \frac 1 n \sum_{t=1}^n \big(y^{(t)} - \theta x^{(t)}\big)^2/2$$

A esta nueva función objetivo se le puede aplicar fácilmente tanto el algoritmo basado en gradiente como la forma cerrada.

Seleccionado un ejemplo aleatorio se va a aplicar el algoritmo del gradiente para que los parámetros $\theta$ se desplacen ligeramente en la dirección opuesta la gradiente.

$$\nabla_{\theta}\Big(\frac{\lambda}{2}\Vert \theta \Vert^2+\big(y^{(t)}-\theta x^{(t)}\big)^2/2\Big)=\lambda \theta - \big(y^{(t)}-\theta x^{(t)}\big)x^{(t)}$$

El nuevo **algoritmo implica**:
- un paso de inicialización de los parámetros $\theta=0$
- escoger aleatoriamente un ejemplo $t=\{1,\dots, n\}$
- actualizar los parámetros 
$$\boxed{\theta = (1 - \eta\lambda)\theta + \eta(y^{(t)} - \theta x^{(t)})x^{(t)}}$$ 

Si se **incrementa** $\pmb\lambda$ hasta $\infty$, minimizar $J$ es equivalente a minimizar $\Vert\theta\Vert$. Por lo tanto, $\theta$ tiene que ser un vector cero. Por esto, $f(x)=\theta x+ \theta_0$ se convierte en $f(x)=\theta_0$, una líena horizontal.

Si se **decrementa** $\pmb\lambda$ hasta cero,  minimizar $J$ es equivalente a minimizar $\frac 1 n \sum_{t=1}^n \big(y^{(t)} - \theta x^{(t)}\big)^2/2$, que es el encaje de los datos de entrenamiento.