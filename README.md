# Bioinspired Intrusion Detection

Proyecto de reproducción del paper **"Feature Selection for Network Intrusion Detection Using Bio-Inspired Metaheuristics"**.  
Implementa tres algoritmos metaheurísticos para selección de características:

- Artificial Bee Colony (ABC)
- Flower Pollination Algorithm (FPA)
- Ant Colony Optimization (ACO)

## Estructura del proyecto:
- data/ → datasets limpios, procesados y resultados
- src/ → implementación de selectores y fitness
- experiments/ → scripts para correr los experimentos
- figures/ → tablas y gráficas generadas
- notebooks/ → análisis complementarios

## Requisitos
- requirements.txt

## Ejecución de los Experimentos
python scripts/run_abc.py
python scripts/run_fpa.py
python scripts/run_aco.py
python scripts/run_pso.py

## Análisis e Interpretación

| Algoritmo  | Features  | Accuracy  | F1       | Precisión (weighted) | Recall (weighted) | Tiempo búsqueda (s) | Entrenamiento final (s) |
|------------|-----------|-----------|----------|----------------------|-------------------|---------------------|-------------------------|
| **ABC**    | 7         | 0.987904  | 0.981907 | 0.979999             | 0.987904          | 814.08              | 5.10                    |
| **FPA**    | 5         | 0.987909  | 0.981979 | 0.982048             | 0.987909          | 79.66               | 5.06                    |
| **ACO**    | 5         | 0.987907  | 0.981899 | 0.975964             | 0.987907          | 88.95               | 4.47                    |
| **PSO**    | 9         | 0.987909  | 0.98198  | 0.981501             | 0.9879            | 218.05              | 4.90                    | 

•	Todos los algoritmos alcanzaron desempeños equivalentes en accuracy y F1 (≈ 98.7 %).
•	FPA fue el más eficiente, logrando la mejor precisión con el menor tiempo de búsqueda.
•	ACO presentó una convergencia rápida con un número reducido de características.
•	ABC mantuvo resultados competitivos, aunque con un costo computacional alto.
•	PSO (configuración tight) obtuvo métricas equivalentes a las mejores, aunque con más características y tiempo medio de ejecución.

## Conclusiones
El uso de metaheurísticas bioinspiradas demostró ser eficaz para reducir la dimensionalidad del dataset sin comprometer el rendimiento del clasificador.
Entre las variantes probadas, FPA se posicionó como el método más balanceado entre rendimiento, tiempo de búsqueda y simplicidad de implementación.

## Autores
- Axel Damian Luna Hernandez
- Jose Alfredo Lopez Torres
