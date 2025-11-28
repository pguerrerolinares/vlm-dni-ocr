# Pipeline DNI OCR + Qwen3-VL

## Descripción
- Pipeline local que combina docTR (OCR) y Qwen3-VL-8B-Instruct (VLM) para extraer los campos clave de un DNI español a un JSON estructurado.
- Todo el procesamiento se ejecuta en local para preservar la privacidad: se carga la imagen, se obtiene el bloque OCR auxiliar y se consulta el modelo multimodal con instrucciones estrictas.
- El repositorio sigue estándares profesionales de ingeniería Python (pyproject.toml, tests automáticos, `Makefile`, CLI empaquetada y servidor local).

## Estructura del proyecto
```
├── Makefile
├── data/
│   ├── input/
│   └── output/
├── models/
├── pyproject.toml
├── requirements.txt          # Apunta al paquete instalado en modo editable
├── scripts/
│   ├── run_pipeline.py
│   └── run_server.py
├── src/
│   └── dni_pipeline/
│       ├── __init__.py
│       ├── api.py
│       ├── cli.py
│       ├── image_preprocessing.py
│       ├── logging_service.py
│       ├── ocr_doctr.py
│       ├── postprocess.py
│       ├── server.py
│       ├── ui.py
│       ├── vlm_qwen.py
│       └── workflow.py
└── tests/
    ├── test_ocr_prompt.py
    └── test_postprocess.py
```

## Requisitos
- Python 3.10 o superior.
- GPU NVIDIA (ej. RTX 4090) con PyTorch compilado para CUDA para obtener el mejor rendimiento; el código también puede ejecutarse en CPU (más lento).
- Acceso local a los pesos de docTR y de `Qwen/Qwen3-VL-8B-Instruct` (el proyecto no realiza llamadas a servicios externos).

## Preparación del entorno
1. Crea y activa un entorno virtual (opcional pero recomendado):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Actualiza `pip` y herramientas básicas:
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   ```
3. Instala el proyecto en modo editable junto con las extras necesarias:
   ```bash
   pip install -e .[api,ui]      # servidor + UI
   pip install -e .[dev,api,ui]  # entorno completo de desarrollo
   ```

## Descarga de modelos
### docTR
docTR descarga sus pesos automáticamente en `~/.cache/doctr` la primera vez que se ejecuta el pipeline. Si trabajas en un entorno sin salida a Internet, copia previamente esos ficheros desde una máquina con acceso usando (`pip install huggingface_hub` si aún no tienes `huggingface-cli`):
```bash
huggingface-cli download mindee/doctr-dbs --local-dir ~/.cache/doctr/models
huggingface-cli download mindee/doctr-crnn --local-dir ~/.cache/doctr/models
```

### Qwen3-VL-8B-Instruct
El pipeline busca los pesos de Qwen en este orden:
1. Ruta indicada mediante `--model-path` o variable `DNI_PIPELINE_QWEN_MODEL_PATH`.
2. Directorio `Qwen3-VL-8B-Instruct/` en la raíz del proyecto.
3. `models/Qwen3-VL-8B-Instruct/`.

Descarga los pesos una única vez y colócalos en cualquiera de las rutas anteriores:
```bash
mkdir -p Qwen3-VL-8B-Instruct
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct \
  --local-dir Qwen3-VL-8B-Instruct \
  --local-dir-use-symlinks False
```
> Necesitas haber hecho `huggingface-cli login` (o configurar `HF_TOKEN`) para poder descargar el modelo. Si prefieres otro directorio, exporta `DNI_PIPELINE_QWEN_MODEL_PATH=/ruta/a/Qwen3-VL-8B-Instruct` antes de lanzar la CLI o el servidor.
> Alternativamente, puedes descargar manualmente los ficheros desde la página oficial de Hugging Face (`https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct`) y descomprimirlos en `Qwen3-VL-8B-Instruct/` o en la ruta que vayas a referenciar con `--model-path`.

## Instalación rápida
```bash
# Dependencias principales
make install

# Entorno completo para desarrollo, API y UI
make install-dev
```
> Si prefieres hacerlo manualmente: `pip install -e .` y añade extras como `.[dev]` o `.[api,ui]` según lo necesites.

## Uso de la CLI
Una vez instalado el paquete, ejecuta:
```bash
dni-pipeline --image path/to/dni.jpg
```
Parámetros relevantes:
- `--ocr-max-side`: tamaño máximo del lado más largo antes del OCR (1600 por defecto).
- `--vlm-size`: tamaño del lienzo cuadrado para el VLM (512 por defecto).
- `--max-new-tokens`: tokens generados por Qwen (256 por defecto).
- `--model-path`: ruta local a los pesos de Qwen si no están en los directorios por defecto.
- `--verbose`: muestra logs `DEBUG`.

### Uso directo del script
`scripts/run_pipeline.py` evita el paso de instalación y añade la carpeta `src/` al `PYTHONPATH` automáticamente. Resulta útil para pruebas rápidas o para ejecutar el código en entornos donde no puedes instalar el paquete.

Ejemplos:
```bash
# Procesar una sola imagen
python scripts/run_pipeline.py --image data/input/samples/photo10.jpg \
  --model-path ./Qwen3-VL-8B-Instruct \
  --max-new-tokens 128 \
  --verbose

# Procesar un directorio completo y guardar los JSON en data/output/
python scripts/run_pipeline.py --input-dir ./data/input/samples \
  --output-dir ./data/output \
  --vlm-size 640 \
  --ocr-max-side 1800
```
Parámetros disponibles (idénticos a la CLI instalada):
- `--image` / `--input-dir`: modo individual o batch.
- `--output-dir`: carpeta donde se guardan los `.json` (si no se indica, solo se imprime por stdout).
- `--model-path`: ruta local al modelo Qwen; si se omite, usa las rutas por defecto descritas arriba.
- `--ocr-max-side`, `--vlm-size`, `--max-new-tokens`, `--verbose` y resto de flags expuestos en `dni-pipeline --help`.

Cada imagen procesada produce un JSON con los campos esperados, impreso por stdout y opcionalmente guardado en disco mediante `--output-dir`.

## Testing y calidad
- Ejecuta `make test` (o `pytest`) para lanzar los tests unitarios incluidos.
- `make lint` ejecuta Ruff sobre `src/` y `tests/`.
- `make format` intenta aplicar correcciones automáticas con Ruff.

## Despliegue local del servidor
El proyecto incluye un servidor FastAPI con interfaz Gradio opcional:
```bash
make serve
# o
dni-pipeline-server --host 0.0.0.0 --port 8000
```
El endpoint `POST /api/v1/extract` recibe el fichero del DNI y devuelve el JSON resultante. La ruta raíz monta la interfaz Gradio para pruebas manuales.

## Rendimiento y comportamiento
- Qwen3-VL-8B-Instruct se recomienda ejecutarlo en GPU (`torch.float16`). La primera inferencia puede tardar más por la carga y warm-up.
- En CPU el pipeline sigue siendo funcional, aunque la inferencia multimodal será significativamente más lenta.
- El pipeline está orientado a ejecuciones batch/nocturnas: prioridad en estabilidad, reproducibilidad y manejo seguro de campos (prefiere `null` a datos dudosos).

## Limitaciones actuales
- Optimizado para DNIs españoles; otros documentos requerirán ajustes en prompts y normalización.
- Imágenes muy deterioradas o con recortes agresivos pueden devolver campos `null` para mantener la fiabilidad.
