

**Autor**: Alejandro Javier Morejón Santiesteban


# Informe de Proyecto: Desarrollo de un Algoritmo de Aprendizaje por Refuerzo para el Juego de Dominó
---

## Índice

1. [Introducción](#introducción)
   - 1.1 Descripción General del Proyecto
   - 1.2 Objetivos
   - 1.3 Abstracción del Problema
2. [Desarrollo](#desarrollo)
   - 2.1 Arquitectura del Sistema
   - 2.2 Motor de Juego
   - 2.3 Entorno de Aprendizaje por Refuerzo
   - 2.4 Modelo de Inferencia
   - 2.5 Estrategias de Entrenamiento
   - 2.6 Integración con la Interfaz Gráfica
3. [Conclusiones](#conclusiones)
   - 3.1 Resultados Obtenidos
   - 3.2 Limitaciones
   

---

## 1. Introducción

### 1.1 Descripción General del Proyecto

Este proyecto implementa un proceso completo de entrenamiento de un algoritmo para el juego de dominó utilizando aprendizaje por refuerzo profundo. El sistema combina un motor de juego personalizado, una interfaz de entorno OpenAI Gym y optimización de políticas proximales (PPO) con redes neuronales recurrentes para crear agentes competitivos capaces de jugar dominó a un nivel estratégico.

La arquitectura del proyecto sigue una clara separación de responsabilidades: el motor del juego implementa las reglas puras del dominó sin conocimiento del algoritmo de aprendizaje, el entorno Gym envuelve el motor para proporcionar compatibilidad con los algoritmos de aprendizaje por refuerzo, y el módulo de inferencia gestiona la carga del modelo entrenado junto con los estados recurrentes de la red LSTM. Los canales de entrenamiento funcionan de forma independiente, lo que permite elegir entre un entrenamiento básico contra oponentes basados en reglas aleatorias o un entrenamiento avanzado mediante juego propio con aprendizaje curricular.

El proyecto proporciona una solución completa para entrenar y evaluar agentes que juegan dominó, tanto a través de código como mediante una interfaz gráfica de usuario que permite la interacción directa con el algoritmo entrenado.

### 1.2 Objetivos

El objetivo principal del proyecto es desarrollar un algoritmo capaz de aprender a jugar dominó de manera competitiva mediante técnicas de aprendizaje por refuerzo profundo. Los objetivos específicos incluyen:

- Implementar un motor de juego que codifique las reglas del dominó de manera precisa y eficiente.
- Diseñar un entorno de aprendizaje por refuerzo compatible con la interfaz estándar de OpenAI Gym.
- Desarrollar un modelo basado en redes neuronales recurrentes (LSTM) capaz de capturar dependencias temporales en el juego.
- Implementar dos estrategias de entrenamiento: entrenamiento básico contra oponentes aleatorios y entrenamiento avanzado mediante juego propio con aprendizaje curricular.
- Crear una interfaz gráfica que permita la interacción humana con el algoritmo entrenado.

### 1.3 Abstracción del Problema

El juego de dominó presenta características que lo hacen particularmente interesante para el aprendizaje por refuerzo. A diferencia de juegos con información perfecta como el ajedrez, el dominó es un juego de información parcial: cada jugador conoce sus propias fichas y las fichas jugadas en la mesa, pero desconoce las fichas del oponente y las que permanecen en el pozo (boneyard).

Esta característica de información parcial requiere que el algoritmo aprenda a inferir información sobre las fichas del oponente basándose en las jugadas realizadas. Por ejemplo, si un oponente pasa cuando el extremo de la mesa muestra un 5, el algoritmo puede inferir que el oponente no tiene fichas con el número 5. Esta capacidad de razonamiento probabilístico sobre el estado oculto del juego es fundamental para jugar bien al dominó.

La abstracción del problema desde el mundo real hacia el algoritmo se realiza mediante los siguientes pasos:

1. **Representación del estado del juego**: El estado del juego se codifica como un vector de 169 dimensiones que incluye la mano del jugador, las piezas ya jugadas, los extremos de la mesa, el tamaño de las manos y una máscara de movimientos legales.

2. **Formulación como proceso de decisión de Markov**: El juego se modela como un Proceso de Decisión de Markov Parcialmente Observable (POMDP), donde el agente recibe observaciones parciales del estado real del juego.

3. **Diseño de la función de recompensa**: Se utiliza una función de recompensa escasa centrada en el resultado del juego: recompensa positiva por ganar, negativa por perder y neutra por empate, con una pequeña penalización por paso para fomentar la eficiencia.

4. **Captura de dependencias temporales**: Mediante el uso de redes LSTM, el algoritmo puede mantener memoria de las jugadas anteriores, lo que le permite realizar inferencias sobre las fichas del oponente.

---

## 2. Desarrollo

### 2.1 Arquitectura del Sistema

El sistema sigue una arquitectura modular que separa claramente la lógica del juego, el entorno de aprendizaje, el entrenamiento y la inferencia. Esta separación permite el desarrollo y las pruebas independientes de cada componente, facilitando el mantenimiento y la extensión del sistema.

La arquitectura consta de cuatro capas principales:

- **Capa del núcleo del juego**: Maneja las reglas del dominó y la mecánica del juego. Esta capa es completamente independiente del algoritmo de aprendizaje y proporciona una interfaz limpia para la manipulación del estado del juego.

- **Capa del entorno de aprendizaje**: Gestiona la interfaz con los algoritmos de aprendizaje por refuerzo, implementando la API estándar de OpenAI Gym. Esta capa traduce el estado del juego a observaciones y las acciones del agente a movimientos en el juego.

- **Capa de entrenamiento**: Implementa los algoritmos de aprendizaje por refuerzo y las estrategias curriculares. Incluye tanto el entrenamiento básico contra oponentes aleatorios como el entrenamiento avanzado mediante juego propio.

- **Capa de interfaz**: Proporciona visualización y permite la interacción humana con el algoritmo entrenado a través de una interfaz gráfica desarrollada con Pygame.

Los componentes clave del sistema y sus responsabilidades se resumen en la siguiente tabla:

| Componente | Rol | Características Principales |
|------------|-----|----------------------------|
| domino_engine.py | Motor de juego | Soporte para 2-4 jugadores, modo equipo, fichas configurables por jugador, implementación completa de reglas |
| domino_gym.py | Entorno RL | API Gym estándar, enmascaramiento de acciones legales, políticas configurables para oponentes, espacio de observación de 169 dimensiones |
| domino_ai.py | Motor de inferencia | Carga del modelo PPO recurrente, gestión del estado LSTM, aceleración GPU CUDA, filtrado de movimientos legales |
| train_recurrent.py | Entrenamiento básico | Entrenamiento paralelo en múltiples entornos, red de políticas LSTM, hiperparámetros configurables |
| train_self_play.py | Entrenamiento avanzado | Aprendizaje curricular (80% profesor, 20% aleatorio), evolución generacional, ciclo de evaluación de modelos |
| gui_domino.py | Interfaz humana | Visualización con Pygame, interacción en tiempo real, sistema de menú para control de juegos |

### 2.2 Motor de Juego

El motor de juego (domino_engine.py) implementa las reglas del dominó de manera completa y precisa. Este componente fue desarrollado antes de la implementación del modelo de aprendizaje, lo que explica que incluya soporte para modos de juego de 2 a 4 jugadores, aunque el entrenamiento final se realizó únicamente para el modo 1vs1 debido a la complejidad adicional que implica entrenar para más jugadores.

La clase principal `DominoGame` gestiona el estado completo del juego, incluyendo las manos de los jugadores, la mesa, los extremos de la cadena de fichas, el historial de jugadas y la detección de condiciones de victoria. El motor implementa las siguientes funcionalidades:

**Gestión del estado del juego**: El motor mantiene el estado completo del juego, incluyendo las manos de cada jugador (diccionario de listas de fichas), la mesa (lista de tuplas con jugador, ficha y lado de colocación), los extremos de la cadena, y el historial de jugadas tanto a la izquierda como a la derecha de la ficha central.

**Validación de movimientos**: El método `get_valid_moves(player)` devuelve la lista de movimientos legales para un jugador dado. Un movimiento es legal si la ficha tiene al menos un valor que coincide con uno de los extremos de la cadena actual. Si no hay fichas en la mesa (primer movimiento), cualquier ficha de la mano es jugable.

**Colocación de fichas**: El método `_place_tile(player, ficha, lado)` maneja la lógica de colocar una ficha en la mesa. Cuando se coloca una ficha, el motor determina automáticamente la orientación correcta basándose en el valor que debe conectar con el extremo de la mesa.

**Detección de victoria**: El motor detecta tres condiciones de finalización del juego: cuando un jugador vacía su mano (victoria directa), cuando todos los jugadores pasan consecutivamente (juego bloqueado), y cuando el juego se bloquea, se determina el ganador comparando los puntos totales de las fichas restantes en cada mano.

**Soporte para múltiples configuraciones**: Aunque el entrenamiento se realizó solo para 2 jugadores, el motor soporta configuraciones de 2 a 4 jugadores, modo de equipos, y número configurable de fichas por jugador. Esta flexibilidad permite futuras extensiones del proyecto.

### 2.3 Entorno de Aprendizaje por Refuerzo

El entorno de aprendizaje por refuerzo (domino_gym.py) implementa la interfaz estándar de OpenAI Gym, proporcionando un puente entre el motor de juego y los algoritmos de aprendizaje por refuerzo. Este componente es fundamental para el entrenamiento del modelo, ya que define cómo se representa el estado del juego y cómo se traducen las acciones del agente.

**Espacio de observación**: El espacio de observación es un vector de 169 dimensiones con valores float32 acotados entre -1.0 y 1.0. La estructura del vector de observación es la siguiente:

| Rango de índices | Componente | Descripción |
|------------------|------------|-------------|
| [0-54] | Mano del agente | Codificación one-hot de los 55 tipos de fichas en la mano del agente |
| [55-109] | Piezas jugadas | Codificación one-hot de las fichas ya colocadas en la mesa |
| [110-111] | Extremos de la mesa | Valores normalizados de los extremos izquierdo y derecho de la cadena |
| [112] | Tamaño de mano del oponente | Conteo normalizado de fichas en la mano del oponente |
| [113] | Tamaño de mano del agente | Conteo normalizado de fichas en la mano del agente |
| [114-168] | Máscara de acciones legales | Máscara binaria que indica las jugadas de fichas válidas (55 fichas) |

Esta representación captura toda la información observable del estado del juego en un formato adecuado para el procesamiento de redes neuronales. La normalización de valores (división por MAX_PIP=9 para los extremos, y por el tamaño máximo de mano para los conteos) facilita el aprendizaje del modelo.

**Espacio de acción**: El espacio de acción es discreto con 55 acciones posibles, correspondientes a todas las fichas de dominó de (0,0) a (9,9) en un conjunto doble 9 estándar. No todas las acciones son legales en un momento dado; solo se pueden jugar fichas que coincidan con los extremos abiertos de la mesa y estén presentes en la mano del agente.

**Manejo de acciones ilegales**: El entorno maneja acciones ilegales mediante el recorte de acciones: si el agente selecciona un movimiento ilegal, el entorno selecciona aleatoriamente entre los movimientos válidos y marca este evento en el diccionario de información. Durante el entrenamiento, este mecanismo permite al agente explorar el espacio de acción inicialmente mientras aprende gradualmente a seleccionar solo movimientos válidos.

**Estructura de recompensas**: El entorno utiliza recompensas escasas centradas en el resultado del juego:

| Tipo de recompensa | Valor | Condición de activación |
|-------------------|-------|------------------------|
| Victoria | +1.0 | El agente vacía su mano primero o tiene menos puntos en juego bloqueado |
| Derrota | -1.0 | El oponente vacía su mano primero o tiene menos puntos |
| Empate | 0.0 | Los puntos son iguales en juego bloqueado |
| Penalización por paso | -0.01 | Cada movimiento del agente (fomenta eficiencia) |

La penalización por paso es pequeña para no interferir significativamente con la señal de recompensa principal, pero suficiente para fomentar un juego eficiente y evitar que el agente aprenda estrategias pasivas.

### 2.4 Modelo de Inferencia

El módulo de inferencia (domino_ai.py) gestiona la carga del modelo entrenado y la generación de acciones durante el juego. Este componente es crucial para la fase de despliegue del modelo, donde el algoritmo entrenado debe tomar decisiones en tiempo real.

**Gestión del estado LSTM**: La característica más importante del módulo de inferencia es la gestión de los estados ocultos de la red LSTM. La clase `DominoAI` mantiene dos componentes de estado:

- **Estado oculto (h)**: Representa la memoria a corto plazo de la red.
- **Estado de celda (c)**: Representa la memoria a largo plazo de la red.

El método `reset_states()` inicializa los estados ocultos a None y establece el flag `episode_starts` a True, señalando el comienzo de un nuevo episodio. Durante la predicción, el flag de inicio de episodio se establece a False después de la primera predicción, indicando que las predicciones posteriores continúan la misma secuencia temporal.

Esta gestión de estados es fundamental para que la política recurrente funcione correctamente: el LSTM debe saber cuándo un nuevo juego comienza para reiniciar su memoria, y cuándo las predicciones continúan un juego en curso para preservar el contexto temporal acumulado.

**Construcción de observaciones**: El método `_build_obs()` construye la observación vectorial esperada por el modelo a partir del estado del juego proporcionado por el motor. Esta construcción sigue exactamente el mismo formato utilizado durante el entrenamiento, garantizando la coherencia entre ambas fases.

**Selección de acciones**: El método `predict()` genera una acción dada el estado actual del juego. El proceso incluye:

1. Obtener los movimientos legales del estado del juego.
2. Construir la observación vectorial.
3. Pasar la observación por el modelo para obtener la acción.
4. Mapear el índice de acción a la ficha correspondiente.
5. Verificar que la acción sea legal; si no, seleccionar el primer movimiento legal disponible.

La verificación de legalidad es una salvaguarda; un modelo bien entrenado debería seleccionar predominantemente acciones legales.

### 2.5 Estrategias de Entrenamiento

El proyecto implementa dos estrategias de entrenamiento complementarias: entrenamiento básico con PPO recurrente contra oponentes aleatorios, y entrenamiento avanzado mediante juego propio con aprendizaje curricular.

#### 2.5.1 Entrenamiento Básico con PPO Recurrente

El entrenamiento básico utiliza RecurrentPPO con estados ocultos LSTM para capturar dependencias temporales en el juego. Esta arquitectura es particularmente adecuada para el dominó porque la estrategia óptima a menudo implica planificación de múltiples turnos y memoria de las piezas jugadas.

**Arquitectura del modelo**: La red de políticas utiliza un perceptrón multicapa con capas LSTM para procesar observaciones secuenciales. La especificación de la arquitectura incluye:

- Redes separadas para el actor (política π) y el crítico (función de valor V), ambas con dos capas ocultas de 128 unidades.
- Capa LSTM con 128 unidades ocultas y una sola capa.
- Configuración `enable_critic_lstm: True` para que tanto el actor como el crítico tengan acceso a los estados ocultos de LSTM.

Los estados ocultos de LSTM se conservan a través de las predicciones de acción dentro de un episodio y se restablecen en los límites del episodio. Durante el entrenamiento, los estados ocultos se incluyen en los datos de la trayectoria, lo que permite que el gradiente fluya a través del tiempo y permite que la red aprenda qué información debe conservarse o descartarse.

**Configuración de hiperparámetros**: La configuración de entrenamiento utiliza parámetros optimizados para el dominó:

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| Pasos de tiempo totales | 5,000,000 | Pasos ambientales totales para entrenamiento |
| Entornos paralelos | 16 | Número de entornos de entrenamiento paralelos |
| Pasos por rollout | 128 | Pasos recopilados por entorno antes de actualización |
| Tamaño de lote | 1024 | Tamaño de minibatch para optimización |
| Épocas por actualización | 5 | Número de pases de optimización por rollout |
| Tasa de aprendizaje | 3e-4 | Tasa de aprendizaje del optimizador Adam |
| Factor de descuento (γ) | 0.99 | Factor de descuento para recompensas futuras |
| GAE λ | 0.95 | Parámetro de estimación de ventaja generalizada |
| Rango de clip PPO | 0.2 | Parámetro de recorte para actualizaciones de política |
| Coeficiente de entropía | 0.1 | Fomento de la exploración |
| Coeficiente de función de valor | 0.5 | Balance entre aprendizaje actor-crítico |
| Norma máxima de gradiente | 0.5 | Umbral de recorte de gradiente |
| Tamaño oculto LSTM | 128 | Tamaño del estado oculto de LSTM |
| Capas LSTM | 1 | Número de capas LSTM |

La tasa de aprendizaje de 3e-4 con un coeficiente de entropía de 0.1 promueve una exploración sostenida, evitando la convergencia prematura hacia políticas deterministas subóptimas.

**Configuración de entorno paralelo**: El entrenamiento emplea 16 entornos paralelos utilizando `SubprocVecEnv`, que distribuye la carga de trabajo entre múltiples procesos de CPU. Cada entorno se inicializa con una semilla diferente para garantizar la diversidad en los estados del juego y el comportamiento del oponente. Esta paralelización proporciona mayor velocidad de recolección de muestras, diversidad de experiencia y mejor utilización de la GPU durante las actualizaciones de entrenamiento.

**Monitoreo de progreso**: La clase `ProgressCallback` proporciona retroalimentación de entrenamiento en tiempo real, realizando un seguimiento de métricas incluyendo la recompensa media del episodio, la duración media del episodio y la fracción de clip de PPO. La métrica de fracción de clip es particularmente importante: un valor demasiado cercano a 0 indica una subutilización de la región de confianza de PPO, mientras que un valor demasiado cercano a 0.2 sugiere actualizaciones de política demasiado agresivas.

#### 2.5.2 Entrenamiento con Juego Propio y Aprendizaje Curricular

El sistema de entrenamiento avanzado implementa un enfoque de entrenamiento basado en generaciones con evolución automática del modelo. Este método permite que el algoritmo mejore jugando contra versiones anteriores de sí mismo, con un plan de estudios que incluye oponentes aleatorios para mayor robustez.

**Arquitectura generacional cíclica**: El sistema opera sobre una arquitectura donde cada ciclo de entrenamiento produce un modelo estudiante candidato que debe demostrar su superioridad frente al profesor actual (mejor modelo) a través de una evaluación competitiva. Esto crea un proceso de mejora iterativo que progresa las capacidades del modelo a través de presión evolutiva.

**Configuración del ciclo de entrenamiento**:

| Parámetro | Valor | Propósito |
|-----------|-------|-----------|
| Ciclos totales | 10 | Número de iteraciones generacionales |
| Pasos por ciclo | 750,000 | Pasos de tiempo de entrenamiento por ciclo |
| Juegos de evaluación | 500 | Juegos de evaluación por ciclo |
| Entornos paralelos | 30 | Entornos de entrenamiento paralelos |
| Ratio aleatorio | 20% | Proporción de movimientos aleatorios del oponente |
| Tasa de aprendizaje | 3e-4 | Tasa de aprendizaje del optimizador PPO |
| Tamaño de lote | 3,840 | N_ENVS × N_STEPS (30 × 128) |

**Oponente curricular**: La clase `CurriculumOpponent` implementa una política mixta que combina el 80% de acciones del modelo profesor con el 20% de acciones aleatorias. Este diseño tiene múltiples propósitos: las acciones del profesor proporcionan aprendizaje estructurado a partir de la estrategia más conocida, mientras que las acciones aleatorias introducen diversidad e impiden que el estudiante memorice patrones específicos. El componente aleatorio es particularmente importante en el dominó, donde existen múltiples movimientos viables, garantizando que el estudiante aprenda a manejar diversos estados del juego.

**Mecanismo de evaluación y promoción**: El mecanismo de evaluación implementa una evaluación competitiva estilo arena donde el modelo estudiante recién entrenado juega contra el mejor modelo maestro actual en condiciones puras de juego propio (sin movimientos aleatorios).

El umbral de promoción se establece en una tasa de victoria del 55%. Este umbral garantiza que solo se promuevan modelos que demuestren una mejora estadísticamente significativa, evitando que el sistema se degrade debido al ruido de entrenamiento o al sobreajuste. Cuando un modelo no alcanza el umbral, el siguiente ciclo continúa entrenando contra el mismo profesor, creando un ciclo de retroalimentación autocorrectivo que mantiene la calidad del modelo.

**Comparación con PPO estándar**:

| Característica | PPO Recurrente | PPO Estándar |
|----------------|----------------|--------------|
| Memoria | Mantiene estados ocultos LSTM entre pasos | Sin memoria entre observaciones |
| Parámetros | Pesos adicionales de celdas LSTM | Menos parámetros (solo MLP) |
| Velocidad de entrenamiento | Más lento (procesamiento secuencial) | Más rápido, totalmente paralelizable |
| Codificación de estado | Codificación temporal implícita | Requiere codificación manual del historial |
| Eficiencia de muestras | Mayor para tareas secuenciales | Menor, necesita más muestras |
| Inferencia | Requiere gestión de estado | Sin gestión de estado |

### 2.6 Integración con la Interfaz Gráfica

La interfaz gráfica (gui_domino.py) proporciona una forma interactiva de jugar contra el algoritmo entrenado. Fue desarrollada utilizando Pygame y permite la interacción en tiempo real con el modelo.

**Carga del modelo**: La interfaz carga el modelo entrenado desde la ruta `./best_models/domino_gen_10.zip`. El modelo se carga en CPU para máxima compatibilidad, aunque puede aprovechar GPU si está disponible.

**Gestión del estado LSTM durante el juego**: Al igual que el módulo de inferencia, la interfaz gestiona los estados ocultos de la LSTM durante el juego. Los estados se reinician al comenzar una nueva partida y se mantienen durante toda la partida para preservar el contexto temporal.

**Interacción del jugador**: El jugador humano puede seleccionar fichas de su mano haciendo clic sobre ellas. Si una ficha puede colocarse en ambos extremos de la mesa, la interfaz permite elegir el lado basándose en la posición del clic (mitad superior o inferior de la ficha). Cuando no hay movimientos disponibles, el sistema automáticamente pasa el turno.

**Visualización**: La interfaz muestra la mesa de juego con las fichas colocadas en un diseño de serpiente que se adapta a los límites de la pantalla. La mano del jugador se muestra en la parte inferior, mientras que las fichas del oponente se muestran como rectángulos ocultos en el lateral derecho.

---

## 3. Conclusiones

### 3.1 Resultados Obtenidos

El proyecto logró desarrollar exitosamente un algoritmo capaz de jugar dominó a un nivel competitivo mediante aprendizaje por refuerzo profundo. Los principales resultados incluyen:

**Motor de juego completo**: Se implementó un motor de juego que codifica correctamente las reglas del dominó, incluyendo la validación de movimientos, la detección de condiciones de victoria y el manejo de juegos bloqueados. Aunque el motor soporta de 2 a 4 jugadores, el entrenamiento se limitó al modo 1vs1 debido a la complejidad adicional que implica entrenar para más jugadores.

**Entorno de aprendizaje funcional**: El entorno Gym proporciona una interfaz estándar para el entrenamiento del modelo, con un espacio de observación bien diseñado que captura la información relevante del juego y un sistema de recompensas que guía el aprendizaje hacia el objetivo de ganar partidas.

**Arquitectura recurrente efectiva**: La arquitectura PPO con LSTM demostró ser efectiva para capturar las dependencias temporales del juego. El modelo aprendió a mantener información sobre las fichas jugadas y a realizar inferencias sobre las fichas del oponente basándose en su comportamiento.

**Entrenamiento progresivo**: El sistema de juego propio con aprendizaje curricular permitió una mejora progresiva del modelo a lo largo de las generaciones. El modelo final (generación 10) muestra un juego estratégico que combina tanto la explotación de patrones aprendidos como la adaptación a situaciones nuevas.

### 3.2 Limitaciones

El proyecto presenta varias limitaciones que deben considerarse:

**Modo de juego limitado**: El modelo fue entrenado únicamente para partidas de 2 jugadores (1vs1). Aunque el motor de juego soporta hasta 4 jugadores, el entrenamiento para más jugadores requeriría una redefinición del espacio de observación y un entrenamiento significativamente más complejo.

**Oponente basado en políticas aleatorias durante entrenamiento básico**: El entrenamiento básico utiliza oponentes con políticas aleatorias, lo que limita la calidad del juego aprendido en esa fase. El entrenamiento con juego propio mitiga esta limitación, pero requiere mayor tiempo de entrenamiento.

**Dependencia del hardware**: El entrenamiento eficiente requiere GPU con soporte CUDA. Aunque el modelo puede ejecutarse en CPU durante la inferencia, el entrenamiento en CPU sería prohibitivamente lento.

**Falta de evaluación contra jugadores humanos expertos**: La evaluación del modelo se realizó principalmente contra versiones anteriores de sí mismo y contra oponentes aleatorios. Una evaluación completa contra jugadores humanos expertos proporcionaría una medida más realista del nivel de juego alcanzado.



---

## Referencias Técnicas

### Requisitos del Sistema

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU | NVIDIA con soporte CUDA | NVIDIA RTX 3060 o superior |
| RAM | 8 GB | 16 GB o más |
| Almacenamiento | 5 GB de espacio libre | 10 GB o más |
| CPU | 4 núcleos | 8 núcleos o más |

### Dependencias de Python

El proyecto requiere Python 3.8+ y las siguientes bibliotecas principales:

- `stable-baselines3`: Framework de aprendizaje por refuerzo
- `sb3-contrib`: Extensión con algoritmos adicionales incluyendo RecurrentPPO
- `gymnasium`: Interfaz estándar para entornos de aprendizaje por refuerzo
- `torch`: Framework de deep learning
- `numpy`: Computación numérica
- `pygame`: Desarrollo de la interfaz gráfica

### Puntos de Entrada

- **Entrenamiento de modelo**: `train_recurrent.py` para entrenamiento básico o `train_self_play.py` para entrenamiento avanzado con juego propio.
- **Juego contra el algoritmo**: `gui_domino.py` para iniciar la interfaz de juego interactiva.
- **Gestión de checkpoints**: `self_play_manager.py` para flujos de trabajo de entrenamiento organizados.