# Medidor de la madurez del banano mediante espectrometría multicanal y aprendizaje de máquina para la clasificación de su estado de cosecha

Este proyecto busca implementar un sistema que pemita la medición y clasificación de bananos según su estado de madurez. Para ello, emplea un microcontrolador Arduino Nano 33 Sense BLE, en el cual se encuentra embebido un modelo de aprendizaje de máquina, y un espectrómetro AS7341, el cual se encarga de medir el espectro de luz sobre el sensor y clasificarlo en ocho canales según la longitud de onda de luz visible, infrarrojo cercano e intensidad.

## Dependencias

Para ejecutar el código contenido en este repositorio, es neceario instalar las siguientes dependencias:

- `matplotlib==3.7.1`
- `micromlgen==1.1.28`
- `numpy==1.23.3`
- `pandas==2.0.0`
- `scikit-learn==1.2.2`
- `xgboost==1.7.5`

## Análisis del proyecto

### Descripción de los datos de entrada

Existen cuatro archivos CSV de datos disponibles para su uso en este proyecto. Estos se encuentran dentro de la carpeta `Data/TrainingValidation` y `Data/Test` dividos en conjunto de entrenamiento y validación y conjunto de pruebas, respectivamente. Cada archivo está nombrado con un color distinto, de tal manera que describe de qué fase de maduración contiene los datos.

Estos datos se presentan en ocho canales de luz visible y un canal de infrarrojo cercano. Se omite el canal de intensidad para tomar únicamente los datos que aportan valor al sistema. El formato de estos datos pueden visualizarse en esta tabla demostrativa:

| channel1 | channel2 | channel3 | channel4 | channel5 | channel6 | channel7 | channel8 | nir |
| -------- | -------- | -------- | -------- | -------- | -------- | -------- | -------- | --- |
| 185      | 1264     | 758      | 893      | 1122     | 1063     | 838      | 473      | 176 |
| 120      | 865      | 506      | 588      | 689      | 649      | 500      | 287      | 122 |
| 123      | 865      | 511      | 597      | 882      | 849      | 669      | 374      | 130 |

## Estrategia

A continuación se describe la estrategia para el desarrollo del sistema:

1. Realizar mediciones con muestras de banano: el primer paso es recolectar muestras de banano para realizar mediciones con el espectrómetro y almacenar los datos.

2. Realizar un análisis exploratorio de los datos: luego de realizar las mediciones, analizar los datos obtenidos para identificar patrones o relaciones, además de posibles anomalías con ayuda de estadística y representaciones gráficas.

3. Desarrollar y entrenar cinco modelos: ya que los datos han sido analizados y las características relevantes han sido identificadas, se debe desarrollar y entrenar cinco modelos de aprendizaje de máquina: decision tree, K neighbors, random forest, support vector machines y XGBoost. Estos modelos se eligieron debido a su buen desempeño y amplio uso en problemas de clasificación. Se eligió la librería scikit-learn para la aplicación de los modelos debido a su amplio catálogo de algoritmos y herramientas de procesamiento de datos.

4. Comparar métricas y curva ROC de cada modelo: para evaluar el desempeño de cada modelo, se puede realizar un análisis del área bajo la curva ROC (AUC-ROC), además de comparar los informes presentados en cada clasificador.

5. Implementar el mejor modelo basado en las métricas: finalmente, al elegir el modelo que mejor se adapta al sistema, se debe implementar en el microcontrolador. Para ello, puede emplearse una herramienta de generación o la implementación de TensorFlow Lite, según sea el caso apropiado. Con esto, será posible clasificar muestras de banano según su estado de madurez.

### Objetivos

El objetivo general es diseñar y desarrollar un sistema aplicable a la industria guatemalteca capaz de medir el nivel de madurez del banano a través de espectrometría y aprendizaje de máquina.

Para poder alcanzar este objetivo, el proyecto involucra la medición y exploración de datos, entrenamiento y evaluación de distintos modelos de aprendizaje de máquina e identificación de las mejores métricas. El resultado final será un sistema que incluye hardware y software capaz de clasificar bananos según se estado de cosecha.

### Modelado

Este es el procedimiento para modelar cada clasificador:

#### Carga de datos

El primer paso es cargar los datos analizados. Estos datos provienen de los archivos CSV y pueden ser cargados fácilmente usando una función o librería de carga de datos.

#### Preparación de datos

Antes de entrenar cada modelo, los datos deben ser preparados acorde al tipo de clasificador a emplear, esto implica aplicar escalado estándar en los modelos K Neighbors y SVM.

Luego de realizar el preprocesamiento, los datos se dividen en conjuntos de entrenamiento, validación y pruebas. Los primeros dos se emplean para entrenar el modelo y ajustarlo, mientras que el tercero se emplea para evaluar su desempeño.

#### Entrenamiento

Con los datos preparados, el siguiente paso es entrenar cada modelo con el conjunto de entrenamiento y validación.

#### Evaluación

Después de entrenar cada modelo, se evalúa el rendimiento utilizando métricas como *precision*, *recall* y *f1-score*. También se analiza la curva ROC y la matriz de confusión para cada modelo. Un buen clasificador tendrá una curva cerca a la esquina superior izquierda de la gráfica. Esto indica que el clasificador tiene una alta tasa de verdaderos positivos y una baja tasa de falsos positivos.

### Resultados

El mejor modelo para la clasificación del estado de madurez de bananos fue el clasificador *random forest*. En este caso, el modelo con los hiperparámetros `max_depth: None`, `max_features: 'sqrt'`, `min_samples_leaf: 1`, `min_samples_split: 2` y `n_estimators: 200` se desempeñó mejor tomando en cuenta sus métricas y su área bajo la curva ROC.

### Tabla de comparación

|     | Model         | Precision | Recall | F1-score |
| --- | ------------- | --------- | ------ | -------- |
| 1   | Decision tree | 0.82      | 0.82   | 0.815    |
| 2   | K Neighbors   | 0.95      | 0.9575 | 0.955    |
| 3   | Random forest | 0.9625    | 0.965  | 0.965    |
| 4   | SVM           | 0.98      | 0.985  | 0.9825   |
| 5   | XGBoost       | 0.985     | 0.985  | 0.985    |

## Conclusión

Luego de entrenar y validar cada uno de los cinco modelos de aprendizaje de máquina - decision tree, K neighbors, random forest, svm y xgboost - los resultados indican que el clasificador random forest tiene el mejor desempeño, medido por la curva ROC. Esto sugiere que este modelo es efectivo para clasificar bananos según su estado de madurez en función de las características del conjunto de datos.

Este hallazgo es importante porque proporciona una recomendación clara sobre qué modelo emplear para realizar esta clasificación. Al implementar este sistema, la industria bananera puede beneficiarse al optimizar sus procesos de calidad al obtener clasificaciones precisas, lo cual elevaría su productividad.

Además, el hecho de que el análisis se haya hecho con los algoritmos de scikit-learn, demuestra la capacidad de esta librería para implementar distintos modelos de aprendizaje de máquina. Esta capacidad es valiosa para para quienes buscan entrenar estos modelos de manera escalable y eficiente.

## Recomendaciones

Si bien los resultados de este proyecto demuestran que el clasificador random forest es un modelo efectivo para abordar este sistema, existen oportunidades para un mayor desarrollo y mejora. Una vía posible para el trabajo futuro es explorar otros modelos de aprendizaje de máquina y evaluar su desempeño con conjuntos de datos similares.

Además de explorar nuevos modelos, también es posible refinar los modelos actuales mediante la ampliación del volumen de los conjuntos de datos. Esto permitiría ajustar sus hiperparámetros y optimizar aún más el rendimiento, logrando mejores resultados en el futuro, aunque implica la necesidad de evaluar la capacidad de carga con las herramientas actuales.

## Reconocimiento

Este proyecto fue realizado como parte del trabajo de graduación para la licenciatura en Ingeniería en Informática y Sistemas de la Universidad Rafael Landívar en el año 2023.

## Licencia

Este trabajo es licenciado bajo la licencia MIT. Para más información, ver el archivo LICENSE.