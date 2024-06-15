# Clustering de series de tiempo
Por: Federico Salinas Samaniego

## Introducción
Los métodos de clustering de tiempo resuelven la tarea de formar clústeres de diferentes series de tiempo para obtener un avance ventajoso sobre el entendimiento de un conjunto dado de series de tiempo; tanto para el descubrimiento de patrones así como identificar 

## Datos
Los datos de estudio consisten en series de tiempo que dentro 
del contexto de astronomía tienen el nombre de curvas de
luz, en las cuales se registran la amplitud del brillo de un objeto
celestre respecto al tiempo; en particular, el conjunto de
datos contiene información proveniente del _Optical Gravitational Lensing Experiment_ y en el cual se cuenta con tres diferentes clases de estrellas variables descritas brevemente en seguida:
- __Cefeida__. Es una estrella que cuyo brillo variable depende
de pulsos en radio y temperatur bajo periodos regulares.

- __Sistema binario eclipsante__. Un par de estrellas orbitan
una sobre la otra, provocando que la variabilidad del brillo
cambie debido a que una estrella eclipsa a la otra; generándose
un par de picos en los que el mayor de ellos se debe a que la 
estrella más pequeña eclipsó a la mayor, reduciendo el brillo
recibido de este último.

- __Estrella RR-Lyrae__. Posee una variación semejante a las
estrellas tipo cefeidas, aunque con periodos de abrillantamiento
menos estables que estas últimas, teniendo una banda de ruido
más amplia que las estrellas variables anteriores.

La base de datos contiene 1000 series de tiempo con las fracciones por clase mostradas en la siguiente tabla.

| Clase | # de series | % de población |
| :-: | :-: | :-: |
| 1 | 152 | 15.2 |
| 2 | 275 | 27.5 |
| 3 | 573 | 57.3 |

## Métodos aplicados


## Resultados

Se utilizaron los índices rand y de información mutúa normalizada para medir la eficiencia de los métodos de clustering utilizados. Este índice evaluado se muestra en la siguiente tabla.

| Método | RAND | NMI |
| :-: | :-: | :-: |
| Jerárquico | 0.7982 | 0.6080 |
| Descomposición Espectral | 0.7666 | 0.5064 |
| Escalamiento Multidimensional | 0.7897 | 0.5881 |
