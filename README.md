# Proyecto de Investigación - Análisis de Imágenes Médicas con Deep Learning

## Descripción

Este proyecto implementa modelos de deep learning para la detección y análisis de patologías en imágenes médicas, específicamente radiografías de pecho (chest X-rays) y detección de cáncer de pulmón. Se utilizan arquitecturas modernas como ResNet18 y modelos híbridos con redes neuronales cuánticas (Hybrid Quantum CNN).

## Estructura del Proyecto

```
investigacion/
├── README.md                          # Este archivo
├── .gitignore                         # Configuración de Git
│
├── etl_datasets.ipynb                 # Pipeline ETL para procesamiento de datos
├── modelos_hqcnn.ipynb                # Implementación de Hybrid Quantum CNN
├── objetivo3_resnet18.ipynb           # Modelo ResNet18 para clasificación
│
├── dataset/                           # Directorio de datos (no versionado)
│   ├── chest_xray/                    # Dataset de radiografías de pecho
│   └── The IQ-OTHNCCD lung cancer dataset/  # Dataset de cáncer de pulmón
│
├── etl_output/                        # Datos procesados (generado automáticamente)
│   ├── chest_xray/
│   └── lung_cancer/
│
├── checkpoints/                       # Modelos entrenados (no versionado)
│   ├── chest/                         # Checkpoints para radiografías
│   └── lung/                          # Checkpoints para cáncer de pulmón
│
├── modelo_resnet/
│   └── resnet.py                      # Implementación del modelo ResNet18
│
└── modelo_pennylane_cnn_torch/
    └── torch.py                       # Implementación Hybrid Quantum CNN con PennyLane
```

## Modelos Implementados

### 1. ResNet18
- Arquitectura clásica de deep learning para clasificación de imágenes
- Entrenado en dataset de radiografías de pecho
- Archivo: `modelo_resnet/resnet.py`
- Notebook: `objetivo3_resnet18.ipynb`

### 2. Hybrid Quantum CNN (HQCNN)
- Combinación de redes neuronales clásicas con circuitos cuánticos
- Implementado con PennyLane y PyTorch
- Archivo: `modelo_pennylane_cnn_torch/torch.py`
- Notebook: `modelos_hqcnn.ipynb`

## Datasets

- **Chest X-ray Dataset**: Radiografías de pecho para detección de patologías
- **IQ-OTHNCCD Lung Cancer Dataset**: Dataset especializado para detección de cáncer de pulmón

**Nota**: Los datasets no están incluidos en el repositorio. Deben descargarse por separado y colocarse en la carpeta `dataset/`.

## Requisitos

- Python 3.8+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- PennyLane (para modelos cuánticos)
- Jupyter

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

2. Instalar dependencias:
```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn pennylane jupyterlab
```

3. Descargar los datasets en la carpeta `dataset/`

## Uso

### Ejecutar pipeline ETL
```bash
jupyter notebook etl_datasets.ipynb
```

### Entrenar modelo ResNet18
```bash
jupyter notebook objetivo3_resnet18.ipynb
```

### Entrenar modelo Hybrid Quantum CNN
```bash
jupyter notebook modelos_hqcnn.ipynb
```

## Flujo de Trabajo

1. **ETL**: Procesar y preparar los datos con `etl_datasets.ipynb`
2. **Entrenamiento**: Ejecutar los notebooks de modelos para entrenar
3. **Checkpoints**: Los modelos entrenados se guardan en `checkpoints/`
4. **Evaluación**: Los resultados se generan en los notebooks

## Archivos Importantes

- `etl_datasets.ipynb`: Preprocesamiento y limpieza de datos
- `objetivo3_resnet18.ipynb`: Entrenamiento y evaluación de ResNet18
- `modelos_hqcnn.ipynb`: Entrenamiento de HQCNN
- `modelo_resnet/resnet.py`: Definición de arquitectura ResNet18
- `modelo_pennylane_cnn_torch/torch.py`: Definición de HQCNN

## Notas

- Los checkpoints y datasets son archivos grandes y no están versionados en Git
- El directorio `etl_output/` se regenera automáticamente ejecutando el pipeline ETL
- Asegúrate de tener suficiente espacio en disco para los datasets

## Autor

Proyecto de investigación en Deep Learning y Computación Cuántica

## Licencia

[Especificar licencia si aplica]
