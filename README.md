# HQCNN vs CNN Clásica — Clasificación de Imagen Pulmonar
**PPI C8 2025-2 G9 — Universidad Peruana Unión**

Evaluación empírica de modelos híbridos cuántico-clásicos (HQCNN) sobre datasets de imagen médica pulmonar, comparados con ResNet-18 como línea base.

---

## Estructura del proyecto

```
investigacion/
├── etl_datasets.ipynb            # Objetivo 1 — ETL
├── modelos_hqcnn.ipynb           # Objetivo 2 — HQC-CNN, PEQML, HQCINN
├── objetivo3_resnet18.ipynb      # Objetivo 3 — ResNet-18 baseline
├── objetivo4_gpu_v2.ipynb        # Objetivo 4 — Entrenamiento + evaluación (versión corregida)
├── objetivo5_ablativo.ipynb      # Objetivo 5 — Análisis ablativo del circuito cuántico
│
├── dataset/                      # Datasets originales (no versionado)
│   ├── chest_xray/
│   └── The IQ-OTHNCCD lung cancer dataset/
├── etl_output/                   # Splits generados por Objetivo 1 (no versionado)
├── checkpoints_gpu_v2/           # Checkpoints entrenados (no versionado)
│
├── modelo_resnet/resnet.py       # Fuente ResNet torchvision
└── modelo_pennylane_cnn_torch/torch.py  # Fuente TorchLayer PennyLane
```

---

## Requisitos de hardware

| Componente | Mínimo | Recomendado (este proyecto) |
|---|---|---|
| GPU | NVIDIA RTX 30xx | RTX 5060 Laptop (8 GB) |
| Driver NVIDIA | 570+ | 591.91 |
| CUDA | 12.8 | 12.8 |
| RAM | 16 GB | 16 GB |

---

## 1 — Instalar driver NVIDIA y CUDA 12.8

> **RTX 5060 / RTX 50xx (Blackwell sm_120) requiere CUDA 12.8 y driver 570+.**

### Paso 1 — Verificar driver actual
Abre **cmd** y ejecuta:
```cmd
nvidia-smi
```
Busca la línea `Driver Version` y `CUDA Version`. Si el driver es menor a **570** o no aparece, actualízalo primero desde:
👉 https://www.nvidia.com/drivers

### Paso 2 — Descargar CUDA Toolkit 12.8
👉 https://developer.nvidia.com/cuda-12-8-0-download-archive

Selecciona: `Windows → x86_64 → tu versión de Windows → exe (local)`

Ejecuta el instalador y sigue los pasos (siguiente → siguiente → instalar).

### Paso 3 — Reiniciar el PC
Obligatorio después de instalar el driver y CUDA.

### Verificar instalación
```cmd
nvidia-smi
nvcc --version
```
Debes ver `CUDA Version: 12.8` o superior.

---

## 2 — Encontrar la ubicación del Python de Jupyter

Antes de instalar PyTorch, necesitas saber qué Python usa tu JupyterLab.  
Ejecuta esto en una **celda del notebook**:

```python
import sys
print(sys.executable)
```

Ejemplo de salida:
```
D:\CURSOS 2026 - 1\Cyberseguridad\jupyterlab\envprueba\Scripts\python.exe
```

Copia esa ruta — la usarás en el siguiente paso.

---

## 3 — Instalar PyTorch con CUDA 12.8

Abre **cmd** (no PowerShell) y usa la ruta exacta de tu Python:

```cmd
"D:\ruta\a\tu\python.exe" -m pip install --force-reinstall --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

> ⚠️ Reemplaza `D:\ruta\a\tu\python.exe` con la ruta que obtuviste en el paso anterior.  
> ⚠️ Usa **cmd**, no PowerShell — PowerShell tiene problemas con rutas que tienen espacios.

### Instalar el resto de dependencias

```cmd
"D:\ruta\a\tu\python.exe" -m pip install pennylane>=0.38 pennylane-lightning scikit-learn pandas matplotlib seaborn tqdm scipy
```

> **Nota:** `pennylane-lightning[gpu]` **no está disponible en Windows**.  
> Se usa `pennylane-lightning` (CPU con backend C++ optimizado), que es suficiente para 4–6 qubits.

### Verificar que CUDA quedó activo

En una celda de Jupyter:
```python
import torch
print(torch.__version__)          # debe decir 2.x.x+cu128
print(torch.cuda.is_available())  # debe decir True
print(torch.cuda.get_device_name(0))
```

---

## 4 — Orden de ejecución de los notebooks

```
etl_datasets.ipynb           ← Ejecutar primero (genera etl_output/)
    ↓
modelos_hqcnn.ipynb          ← Revisar arquitecturas (no entrena)
    ↓
objetivo3_resnet18.ipynb     ← Revisar ResNet-18 (no entrena)
    ↓
objetivo4_gpu_v2.ipynb       ← Entrenamiento completo GPU
    ↓
objetivo5_ablativo.ipynb     ← Análisis ablativo (requiere Obj 4 completo)
```

---

## 5 — Modelos implementados

| Modelo | Referencia | Qubits | Params (~) |
|---|---|---|---|
| ResNet-18 | He et al. (2016) | — | 11.2 M |
| HQC-CNN | Dong et al. (2023) | 4 | 90 K |
| PEQML | Abdur & Kim (2025) | 4 | 3 K |
| HQCINN-shallow | Akpinar et al. (2025) | 4 | 96 K |
| HQCINN-deep | Akpinar et al. (2025) | 6 | 96 K |

---

## 6 — Datasets

| Dataset | Clases | Total | Tarea |
|---|---|---|---|
| Chest X-Ray (Kaggle) | Normal, Pneumonia | 5 856 | Binaria |
| IQ-OTH/NCCD | Benign, Malignant, Normal | 1 097 | Multiclase |

Los datasets **no están versionados**. Descárgalos y colócalos en `dataset/`.

---

## Notas importantes

- `etl_output/`, `checkpoints*/`, `dataset/` y archivos `*.pt` están en `.gitignore` — son demasiado grandes para Git.
- El ETL es reproducible: siempre genera los mismos splits con `seed=42`.
- Los resultados CSV y JSON sí se versionan para trazabilidad.
