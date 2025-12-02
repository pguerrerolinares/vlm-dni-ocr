# Requerimientos técnicos y consideraciones de seguridad

Este documento resume todo lo que debe instalarse para ejecutar el pipeline **DNI OCR + Qwen3-VL** y las implicaciones de seguridad asociadas, con el objetivo de facilitar su revisión por parte de equipos corporativos de IT/GRC.

## 1. Alcance del pipeline
- El repositorio implementa un flujo totalmente local que combina OCR (docTR) y un VLM (Qwen3-VL-8B-Instruct) para extraer campos de un DNI español, sin enviar datos a servicios externos (`README.md:3-32`).
- La ejecución estándar comprende preprocesado de imagen, OCR, inferencia multimodal y un validador adicional también basado en Qwen (`README.md:8-31`, `src/dni_pipeline/core/vlm.py:1-160`).

## 2. Intérprete y entorno base
- **Versión mínima**: Python ≥ 3.10 (`pyproject.toml:5-20`).
- **Hardware recomendado**: GPU NVIDIA con CUDA para acelerar PyTorch; el flujo también funciona en CPU con más latencia (`README.md:90-94`).
- **Sistema operativo**: Linux x86_64 verificado; cualquier distribución capaz de ejecutar Python 3.10 y drivers CUDA funcionará.
- **Virtualización**: se recomienda entorno virtual (`python -m venv .venv`) para aislar dependencias; no se requieren privilegios de administrador.

## 3. Dependencias Python instaladas
Las dependencias se gestionan vía `pyproject.toml`. Al ejecutar `pip install -e .` se instalan los paquetes del núcleo (tabla 1); los extras `[api]`, `[ui]` y `[dev]` añaden componentes opcionales (tablas 2 y 3).

**Tabla 1 – Dependencias núcleo (`pyproject.toml:14-20`)**

| Paquete | Fuente | Rol en el sistema | Consideraciones de seguridad |
| --- | --- | --- | --- |
| `torch` | PyPI (PyTorch) | Inferencia en GPU/CPU para OCR, VLM y operaciones tensoriales. | Binarios incluyen código C++/CUDA; validar hashes oficiales y alinear con versión de driver. |
| `transformers` | Hugging Face | Carga de modelos Qwen y utilidades de tokenización. | Descarga código Python adicional; revisar versión y changelog antes de actualizar. |
| `python-doctr` | Mindee | Envoltura OCR (detector DB + CRNN). | Primer uso descarga pesos a `~/.cache/doctr`; mantener caché con permisos restringidos. |
| `Pillow` | PyPI | Lectura/escritura de imágenes. | Incluye código C; mantener actualizado para cierres CVE de codecs. |
| `opencv-python-headless` | PyPI | Preprocesado geométrico/fotométrico, métricas de foco. | Distribuye binarios nativos; aconsejable firma de paquete y ejecución en entorno aislado. |

**Tabla 2 – Extras de ejecución**

| Extra | Paquetes | Uso principal | Observaciones |
| --- | --- | --- | --- |
| `[api]` | `fastapi`, `uvicorn`, `python-multipart`, `httpx` (`pyproject.toml:22-28`) | Expone API REST y servidor ASGI. | Solo escucha en host/puerto configurados; sin dependencias externas más allá de PyPI. |
| `[ui]` | `gradio`, `httpx` (`pyproject.toml:29-32`) | UI web opcional para demos. | Gradio sirve activos estáticos locales; revisar antes de exponer a redes no confiables. |

**Tabla 3 – Extras de desarrollo (`pyproject.toml:33-38`)**

| Paquete | Propósito |
| --- | --- |
| `pytest`, `pytest-cov` | Suite unitaria + cobertura. |
| `ruff` | Linter/formatter. |
| `mypy` | Comprobador de tipos estático. |

### 3.1 Dependencias transitivas relevantes
- `numpy`, `scipy`, `tokenizers`, `safetensors`, `huggingface-hub`, `onnxruntime` (vía docTR) y librerías Rust/C/CUDA adicionales. Todas se obtienen de PyPI/Hugging Face y deben validarse mediante hashes o un mirror corporativo.
- `tokenizers` y `safetensors` usan extensiones nativas; requieren entornos confiables para su compilación/carga.

## 4. Modelos y artefactos externos

### 4.1 docTR
- Pesos `det` y `reco` (por defecto `db_resnet50` + `crnn_vgg16_bn`) se descargan automáticamente al ejecutar el OCR (`src/dni_pipeline/core/ocr.py:1-120`).
- Ubicación por defecto: `~/.cache/doctr/models`. Permisos recomendados `700` para evitar accesos de otros usuarios.
- En entornos sin Internet, se pueden pre-sembrar vía `huggingface-cli download mindee/doctr-*` según se documenta en `README.md:111-134`.

### 4.2 Qwen3-VL-8B-Instruct
- Los pesos se buscan localmente (`DEFAULT_QWEN_SEARCH_PATHS`) y, si no existen, se intenta descargar desde Hugging Face (`src/dni_pipeline/core/vlm.py:21-80`).
- La carga usa `transformers.AutoProcessor` y `Qwen3VLForConditionalGeneration` con `trust_remote_code=True` (`src/dni_pipeline/core/vlm.py:53-64`), requisito del repositorio oficial de Qwen. Esto implica ejecutar código Python incluido en la tarjeta del modelo, por lo que:
  - Debe fijarse un commit/tag concreto al clonar los pesos.
  - Se recomienda revisar el contenido del repositorio y almacenarlo en un artefacto interno firmado antes de su distribución.
- Ubicaciones soportadas: `--model-path`, `Qwen3-VL-8B-Instruct/` en la raíz o `models/Qwen3-VL-8B-Instruct/` (`README.md:119-134`).

### 4.3 Herramientas auxiliares
- `huggingface-cli` no forma parte de las dependencias directas, pero se utiliza para descargar modelos de forma autenticada y puede instalarse temporalmente en el mismo entorno virtual.

## 5. Flujo de instalación controlada
1. Crear y activar entorno virtual aislado.
2. `python -m pip install --upgrade pip setuptools wheel` (`README.md:95-109`).
3. Instalar el paquete en editable junto con los extras deseados (por ejemplo `pip install -e .[api,ui]`).
4. Descargar los pesos de docTR y Qwen desde una máquina con Internet y moverlos a la red interna (ver sección 4).
5. Registrar hashes/firmas de todos los artefactos para futuras verificaciones.

## 6. Consideraciones de seguridad y mitigaciones
- **Ejecución offline**: el pipeline no realiza llamadas externas durante la inferencia; cualquier acceso a red proviene únicamente de la descarga inicial de modelos o de la UI si se publica en Internet (`README.md:90-134`, `195-224`). Puede operar completamente aislado tras cachear los pesos.
- **trust_remote_code**: obligatorio para Qwen; mitigar fijando commit, revisando código y almacenándolo en un repositorio interno verificado (`src/dni_pipeline/core/vlm.py:53-64`).
- **Binarios nativos**: PyTorch, OpenCV y Pillow incluyen extensiones compiladas. Mantenerlos en sus LTS soportados y monitorizar CVEs (CVE-2023-4863 para libwebp, etc.).
- **Permisos sobre modelos**: guardar pesos en ubicaciones dedicadas con permisos restringidos (por ejemplo `/opt/dni-vlm/models`, modo 750) y registrar el origen.
- **Servidor opcional**: si se usa `[api]` o `[ui]`, exponer únicamente en redes internas, detrás de TLS y controles de autenticación (FastAPI no incluye auth por defecto).
- **Datos temporales**: las subidas se almacenan en `/tmp` salvo que se configure lo contrario (`README.md:222-224`). Limpiar directorios temporales y cifrar discos cuando sea necesario.
- **Logs**: el pipeline genera trazas con el bloque `[OCR]` y JSONs resultantes; se recomienda rotación y acceso restringido.

## 7. Checklist previo a producción
- [ ] Verificar versión de Python (≥3.10) y de CUDA/driver compatibles con la build de PyTorch instalada.
- [ ] Clonar/descargar los repositorios de docTR y Qwen en un entorno con Internet, escanearlos y firmarlos antes de moverlos a la red corporativa.
- [ ] Emitir SBOM o inventario de paquetes a partir de `pip freeze` del entorno aprobado.
- [ ] Ejecutar `pytest`, `ruff` y `mypy` para validar que el paquete quedó instalado íntegro (`pyproject.toml:33-38`).
- [ ] Configurar variables (`DNI_PIPELINE_QWEN_MODEL_PATH`, `DNI_PIPELINE_KEEP_UPLOADS`) según las políticas internas y documentar su valor.
- [ ] Revisar periódicamente boletines de seguridad de PyTorch, Hugging Face Transformers y OpenCV; planificar ventanas de actualización controladas.

Con estas medidas, el script puede desplegarse en entornos corporativos manteniendo un inventario claro de los componentes instalados y sin exponer datos a servicios de terceros.
