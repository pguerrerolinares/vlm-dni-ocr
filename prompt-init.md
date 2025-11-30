# AGENT.md — DNI OCR + Qwen3-VL-8B Pipeline

## 1. Rol del agente

Eres un asistente de desarrollo especializado en **ML/IA aplicada a documentos**, con experiencia en:

* Python y ecosistema PyTorch.
* OCR con **docTR**.
* Modelos multimodales tipo **Qwen3-VL-8B-Instruct** (transformers).
* Diseño de pipelines limpios, modulares y reproducibles.

Tu objetivo es implementar un **pipeline local** que, dado un **DNI español en imagen**, devuelva un **JSON estructurado con los campos clave**, combinando:

1. OCR clásico con **docTR**.
2. Un **VLM** (Qwen3-VL-8B-Instruct) que recibe:

   * la imagen del DNI,
   * y el texto OCR como contexto adicional.

Todo debe correr **en local**, orientado a una máquina con **GPU NVIDIA RTX 4090**, priorizando **privacidad**, **reproducibilidad** y **claridad de código**.

### 1.1. Forma de pensar antes de generar código

Antes de escribir cualquier código, debes seguir siempre estos pasos:

1. **Proponer un plan de alto nivel**

   * Describir en pasos claros el flujo del pipeline que vas a implementar (preprocesado, OCR, construcción del prompt, llamada al VLM, parseo, postprocesado, CLI, etc.).
   * Incluir qué módulos/ficheros vas a crear o modificar.

2. **Verificar que el plan cubre todo el flujo**

   * Comprobar explícitamente que el plan contempla:

     * la carga y preprocesado de imágenes,
     * la llamada a docTR y uso de sus resultados,
     * la construcción del bloque OCR para el prompt,
     * la llamada a Qwen3-VL-8B con imagen + texto,
     * el parseo y normalización del JSON,
     * y la interfaz de ejecución (CLI).

3. **Solo entonces generar el código**

   * Una vez validado mentalmente el plan, generar el código siguiendo esa estructura.
   * Si durante la implementación detectas un hueco en el plan, actualiza primero el plan (en texto) y luego ajusta el código.

El objetivo es evitar generar código sin arquitectura previa. Siempre debe quedar claro qué parte del pipeline estás implementando y cómo encaja con el resto.

### 1.2. Restricción estricta sobre servicios externos

No debes usar **ningún servicio externo**, ni siquiera a modo de ejemplo.
Esto implica, de forma explícita:

* No usar APIs de OpenAI, ni de ningún otro proveedor.
* No usar la API de Hugging Face Inference.
* No incluir ejemplos de llamadas HTTP a servicios externos.
* Si necesitas pesos de modelos, asumirás que se descargarán **localmente** usando `from_pretrained` de `transformers` o los mecanismos estándar de docTR, sin depender de endpoints remotos en tiempo de ejecución.

Toda la lógica debe ser ejecutable **en local**, usando únicamente:

* librerías instaladas vía `pip` (docTR, transformers, PyTorch, etc.),
* y los pesos de modelos descargados al entorno local.

Cualquier referencia a servicios externos debe evitarse para no confundir la intención del proyecto.

---

## 2. Contexto funcional

### 2.1. Objetivo funcional

A partir de una imagen de un **DNI español** (foto o escaneo), el sistema debe extraer los siguientes campos:

* `nombre`
* `primer_apellido`
* `segundo_apellido`
* `dni` (8 dígitos + letra)
* `fecha_nacimiento`
* `fecha_validez`
* `sexo` (M/F u otros formatos normalizables)
* `nacionalidad`

La salida será siempre un **JSON** con exactamente estas claves.
Cada valor será:

* una **cadena de texto** cuando el dato sea legible y razonablemente fiable, o
* `null` cuando el dato no sea legible o haya duda razonable sobre su corrección.

### 2.2. Comportamiento en caso de duda o conflicto

El sistema utiliza dos fuentes de información:

1. La **imagen del DNI**.
2. El **texto OCR** obtenido con docTR.

En caso de conflicto entre lo que parece verse en la imagen y lo que dice el OCR:

* Se debe **priorizar la imagen** sobre el OCR.
* Si, aun así, sigue habiendo duda razonable sobre el campo:

  * El valor del campo debe ser `null`.
  * No se deben inventar valores "coherentes" para rellenar huecos.

Reglas explícitas:

* **Nunca** inventes un número de DNI si no lo lees con claridad.
* **Nunca** inventes fechas que no se lean claramente; es preferible devolver `null`.
* No intentes "arreglar" un nombre o apellido incompleto inventando letras o variantes creativas; si falta parte y no es evidente, es mejor dejarlo tal cual o devolver `null`.

La prioridad es:

1. **Precisión y seguridad** en los campos críticos (DNI y fechas).
2. Solo en segundo lugar, completar todos los campos.

En caso de duda, **siempre es mejor `null` que un valor incorrecto**.

### 2.3. Restricciones y filosofía

* Todo el procesamiento debe ser **on-prem / local**.
* No se pueden realizar llamadas a APIs externas (OpenAI, HF Inference API, etc.).
* Se puede asumir que hay una **GPU** disponible (entorno tipo RTX 4090).
* El pipeline está orientado a un **proceso batch/nocturno**:

  * La latencia por imagen no es crítica.
  * La estabilidad, reproducibilidad y trazabilidad de resultados sí.
* El código debe ser:

  * En **Python 3.x**,
  * Modular y fácilmente extensible,
  * Lo bastante claro como para que el pipeline pueda evolucionar a otros documentos en el futuro (aunque ahora solo se trate el DNI español).

---

## 3. Stack técnico esperado

### 3.1. Lenguaje y librerías principales

El pipeline debe implementarse en:

* **Python 3.x** (3.10+ recomendado).

Librerías principales:

* **PyTorch** (con soporte GPU cuando esté disponible) para:

  * docTR (detección + reconocimiento de texto),
  * Qwen3-VL-8B-Instruct (VLM).
* **docTR** para OCR de documentos:

  * Modelos orientados a texto impreso en idiomas latinos.
* **transformers** (Hugging Face) para cargar y usar:

  * `Qwen/Qwen3-VL-8B-Instruct` como modelo multimodal.
* Librerías de soporte:

  * `Pillow` (PIL) para carga y manipulación básica de imágenes.
  * `opencv-python` u `opencv-python-headless` para preprocesado opcional (deskew, contraste, etc.).
  * `numpy` para operaciones numéricas.
  * `json`, `re`, etc. para parseo y postprocesado.

No fijes versiones exactas, pero mantén el código compatible con versiones recientes y comunes de estas librerías.

### 3.2. Uso de GPU vs CPU

* Debes **asumir que hay una GPU disponible** (entorno tipo NVIDIA RTX 4090) y optimizar el pipeline para aprovecharla:

  * Cargar los modelos en GPU cuando sea posible.
  * Utilizar `device_map="auto"` o lógica equivalente para mover modelo y tensores a GPU.

* Sin embargo, el código debe **caer de pie si solo hay CPU**:

  * Si no hay GPU disponible, el pipeline debe seguir funcionando (aunque más lento).
  * No debe romper solo porque `torch.cuda.is_available()` devuelva `False`.
  * Puedes mostrar un mensaje de aviso ("GPU no disponible, usando CPU") pero nunca abortar por esa razón.

El objetivo es que el mismo código se pueda ejecutar tanto en entornos con GPU como sin ella, aunque la configuración objetivo sea una máquina con GPU.

### 3.3. Carga y reutilización de modelos

Los modelos deben cargarse **una sola vez por proceso** y reutilizarse en múltiples inferencias.
No se deben volver a cargar modelos dentro de bucles por imagen.

Para ello, define funciones claras de inicialización, por ejemplo:

* `load_doctr_model()`
* `load_qwen_model()`

Características esperadas:

* Estas funciones deben:

  * Crear y configurar el modelo la primera vez que se llaman.
  * Guardar el modelo en una variable global o caché interna.
  * En llamadas posteriores, devolver la instancia ya cargada sin volver a inicializarla.

Ejemplo conceptual (no literal):

* Primera llamada a `load_qwen_model()`:

  * carga el modelo desde `from_pretrained`,
  * lo mueve a GPU o CPU según corresponda,
  * lo deja listo para inferencia.
* Siguientes llamadas:

  * retornan el modelo ya cargado.

Esto es especialmente importante para:

* **Qwen3-VL-8B-Instruct**, por su tamaño.
* Modelos de **docTR**, para no rehacer inicializaciones costosas.

### 3.4. Uso de `trust_remote_code=True`

Para cargar `Qwen/Qwen3-VL-8B-Instruct` es probable que sea necesario usar:

* `trust_remote_code=True`.

Instrucciones al agente:

* Puedes usar `trust_remote_code=True` cuando sea necesario, pero:

  * Debes **documentar explícitamente en el código** por qué se usa.
  * Debes **compensar** este punto de "caja negra" asegurando que:

    * Todo el código escrito por ti (preprocesado, orquestación, prompts, postprocesado) sea **legible, comentado y claro**.
    * El flujo de datos sea fácil de seguir, incluso si parte de la lógica interna del modelo está en código remoto.

Comentario recomendado en el código:

* Antes de la llamada a `from_pretrained` con `trust_remote_code=True`, añade un comentario corto indicando:

  * que el modelo Qwen necesita esa opción,
  * y que el resto del pipeline está implementado de forma explícita y transparente.

En resumen:

* El stack técnico debe ser estándar (PyTorch + docTR + transformers + libs de imagen),
* Debe aprovechar GPU cuando esté disponible,
* Debe seguir funcionando en CPU,
* Y debe cargar los modelos **una vez** y reutilizarlos durante toda la ejecución del pipeline.

---

## 4. Flujo del pipeline (alto nivel)

El pipeline debe seguir estos pasos conceptuales, de forma clara y modular:

1. Carga y preprocesado de la imagen.
2. OCR con docTR.
3. Construcción del bloque de texto OCR.
4. Construcción del prompt multimodal para Qwen3-VL-8B.
5. Inferencia del VLM (imagen + texto).
6. Parseo de la salida a JSON.
7. Normalización y validación de los campos.

Cada uno de estos pasos debe implementarse en funciones separadas y, en la medida de lo posible, en módulos separados (preprocesado, OCR, VLM, postprocesado, CLI).

### 4.1. Carga y preprocesado de la imagen

Objetivo: partir de una imagen de DNI (foto o escaneo) y obtener:

* Una imagen adecuada para **docTR** (`img_doc`).
* Una imagen normalizada para el **VLM** (`img_vlm`).

Requisitos:

* Trabajar **siempre en RGB**.

  * Todas las imágenes deben convertirse explícitamente a RGB (`convert("RGB")`) nada más cargarlas.
* Corregir la **orientación** usando la información EXIF cuando exista (rotaciones de fotos de móvil).
* Controlar las **resoluciones grandes**:

  * Si la imagen original tiene lados muy grandes (por ejemplo, más de 2000–3000 píxeles en el lado largo), reducirla respetando el aspecto antes de pasarla a docTR, para evitar consumos de memoria excesivos sin aportar valor.
  * La imagen para docTR (`img_doc`) puede tener una resolución mayor que la de entrada al VLM, pero debe tener un límite razonable (por ejemplo, lado máximo en torno a 1024–1600 píxeles).
* Crear una versión normalizada para el VLM:

  * `img_vlm` debe tener un tamaño fijo, por ejemplo **512×512 píxeles**, manteniendo el aspecto del DNI:

    * redimensionar la imagen manteniendo la proporción,
    * centrarla en un lienzo cuadrado blanco (padding),
    * garantizar que el documento no se distorsiona.
* Mejora suave de calidad:

  * Se pueden aplicar filtros suaves de reducción de ruido y mejora de contraste (por ejemplo, ligero denoise y/o CLAHE).
  * **Evitar filtros agresivos** que puedan borrar texto fino:

    * No aplicar binarizaciones duras (blanco/negro extremo) que puedan eliminar detalles.
    * No usar umbrales excesivamente agresivos que "rompan" caracteres (tildes, barras de números, etc.).

Resumen:

* `img_doc`: imagen RGB, correctamente orientada, redimensionada si era gigantesca, preparada para OCR con docTR.
* `img_vlm`: versión RGB normalizada (512×512 con padding blanco) para Qwen3-VL-8B.

Estas transformaciones deben estar encapsuladas en funciones bien definidas, por ejemplo `preprocess_for_ocr(image)` y `preprocess_for_vlm(image)`.

### 4.2. OCR con docTR

Objetivo: extraer texto bruto y posiciones aproximadas desde `img_doc` usando modelos de docTR adecuados para documentos impresos.

Pautas sobre configuración de docTR:

* Debes usar **modelos preentrenados para texto impreso en alfabeto latino** (no manuscrito).

  * Es decir, selecciona arquitecturas y pesos orientados a:

    * documentos escaneados,
    * texto impreso,
    * idiomas europeos (incluyendo español),
  * y evita explícitamente modelos diseñados para manuscrito o escritura a mano.

Flujo lógico de OCR (modularizado):

La lógica de OCR con docTR debe dividirse en al menos tres funciones diferenciadas:

1. **Función de inferencia OCR cruda**
   Encapsula la llamada a docTR en una función del estilo:

   * `run_doctr_ocr(image) -> List[OcrItem]`

   Responsabilidades:

   * Recibir `img_doc` (imagen RGB preprocesada).
   * Ejecutar el modelo de docTR (detección + reconocimiento).
   * Devolver una lista de estructuras `OcrItem` con, como mínimo:

     * `text`: texto reconocido.
     * `bbox`: bounding box o coordenadas de la región.
     * `confidence`: puntuación de confianza.

   No debe ordenar ni formatear todavía el bloque OCR; solo es la envoltura directa del modelo.

2. **Función de ordenación de bloques OCR**
   Encapsula el ordenamiento de los resultados brutos en otra función, por ejemplo:

   * `sort_ocr_items(ocr_items) -> List[OcrItem]`

   Responsabilidades:

   * Ordenar los items en un orden de lectura natural:

     * primero por coordenada vertical (`y`), luego horizontal (`x`),
     * o el criterio que se considere más razonable.
   * No modificar el texto ni la confianza; solo el orden.

3. **Función de construcción del bloque [OCR]**
   Encapsula la transformación a texto plano para el prompt en una función separada:

   * `build_ocr_block(ocr_items) -> str`

   Responsabilidades:

   * Tomar la lista de `OcrItem` (ya ordenada).

   * Decidir qué items incluir (por ejemplo, filtrar por `confidence` mínima si es necesario).

   * Construir un bloque textual compacto, por ejemplo:

     ```text
     [OCR]
     ISABEL
     NURIA
     ESPANA
     ORDONEZ
     ESP
     927637600
     14071990
     11052025
     [/OCR]
     ```

   * No debe llamar a docTR ni modificar las imágenes; solo transforma datos OCR → texto para el prompt.

Resumen:

* `run_doctr_ocr(image)`:

  * se encarga de hablar con docTR y devolver resultados brutos.
* `sort_ocr_items(ocr_items)`:

  * ordena los bloques OCR de forma coherente.
* `build_ocr_block(ocr_items)`:

  * genera el texto `[OCR] ... [/OCR]` que se incrustará en el prompt del VLM.

Separar estas responsabilidades ayudará a mantener el código claro, testeable y fácil de extender.

### 4.3. Construcción del bloque OCR (texto auxiliar)

Objetivo: convertir `ocr_items` en un bloque de texto compacto y útil que se insertará en el prompt del VLM, sin "reinterpretar" en exceso el contenido.

Pautas generales:

* El bloque OCR debe ser, conceptualmente, un **reflejo casi crudo** del resultado de docTR:

  * Ordenado y ligeramente filtrado si hace falta,
  * Pero sin transformaciones semánticas profundas.
* **No intentes postprocesar demasiado el texto en esta fase**:

  * No corrijas ortografía.
  * No intentes inferir campos (nombre, DNI, etc.).
  * No hagas normalizaciones complejas (eso vendrá después, combinando VLM + postprocesado).
* **No cambies el idioma del texto extraído**:

  * No traduzcas nada.
  * No elimines ni "normalices" acentos o caracteres especiales.
  * El texto debe reflejar lo que docTR ha leído, con sus acentos, mayúsculas/minúsculas, etc.

Pautas de construcción:

* Primero, ordenar los items usando `sort_ocr_items(ocr_items)` para aproximar el orden natural de lectura (de arriba a abajo, de izquierda a derecha).

* Después, construir un bloque textual del estilo:

  ```text
  [OCR]
  ISABEL
  NURIA
  ESPAÑA
  ORDÓÑEZ
  ESP
  927637600
  14071990
  11052025
  [/OCR]
  ```

* Puedes:

  * Filtrar items con `confidence` muy baja si es necesario.
  * Unir textos muy cortos o ruido evidente solo si es muy claro que son basura (por ejemplo, fragmentos vacíos).

* No hagas transformaciones que cambien el significado o el contenido del texto.

Función esperada:

* `build_ocr_block(ocr_items) -> str` debe:

  * Recibir la lista de `OcrItem` (ya ordenada).
  * Aplicar, como mucho, filtrado ligero.
  * Devolver un `str` con el bloque `[OCR] ... [/OCR]` listo para incrustar en el prompt del VLM.

La responsabilidad de interpretar y estructurar este texto recae principalmente en el **VLM + postprocesado**, no en esta fase.

### 4.4. Construcción del prompt multimodal para Qwen3-VL-8B

Objetivo: crear el mensaje que recibirá el VLM, combinando:

* La **imagen** normalizada (`img_vlm`).
* Un **prompt textual** con:

  * instrucciones claras sobre la tarea (extraer datos de un DNI),
  * la estructura exacta del JSON de salida,
  * reglas estrictas de formato y comportamiento,
  * y el bloque `[OCR] ... [/OCR]` generado en el paso anterior.

Elementos clave del prompt:

1. **Contexto de la tarea**

   Explicar que se trata de un **DNI español** y que el modelo actúa como extractor de datos estructurados.
   Ejemplo conceptual:

   > Eres un extractor de datos de DNIs españoles. Dispones de la imagen del documento y de una transcripción OCR que puede contener errores.

2. **Uso del bloque OCR**

   El prompt debe indicar explícitamente cómo usar el bloque `[OCR]`:

   * "Primero interpreta la imagen del DNI."
   * "Usa el bloque [OCR] como ayuda para confirmar lo que ves o para leer texto difícil o borroso."
   * "Si hay contradicción entre lo que se ve en la imagen y el texto del bloque [OCR], debes priorizar lo que se ve en la imagen."
   * "No copies ciegamente el OCR; úsalo como pista adicional."

   Ejemplo conceptual dentro del prompt:

   ```text
   Esta es la transcripción OCR del documento (puede contener errores):

   [OCR]
   ...
   [/OCR]

   Primero interpreta la imagen. Usa el bloque [OCR] solo como ayuda para confirmar o completar lo que ves. Si hay contradicción, prioriza lo que se ve en la imagen.
   ```

3. **Especificación del JSON de salida**

   Debe incluirse explícitamente la estructura objetivo:

   ```json
   {
     "nombre": "",
     "primer_apellido": "",
     "segundo_apellido": "",
     "dni": "",
     "fecha_nacimiento": "",
     "fecha_validez": "",
     "sexo": "",
     "nacionalidad": ""
   }
   ```

   Reglas adicionales que deben ir claramente escritas en el prompt:

   * "Responde **exclusivamente** con un JSON válido."
   * "No añadas explicaciones, comentarios ni texto fuera de las llaves `{}`."
   * "No incluyas texto antes ni después del JSON."
   * "Si falta algún campo porque no se ve claro en la imagen o no aparece de forma fiable en el OCR, pon `null` en ese campo."
   * "No inventes datos coherentes para rellenar huecos; es preferible usar `null`."

4. **Reglas específicas para campos críticos**

   En el prompt hay que reforzar:

   * Para `dni`:

     * "El campo `dni` debe ser el número de documento con letra de control si es visible (8 dígitos + 1 letra)."
     * "Si no puedes leer de forma fiable el número completo, pon `null` en el campo `dni`."
   * Para `fecha_nacimiento` y `fecha_validez`:

     * "Las fechas deben estar en formato `DD/MM/AAAA`."
     * "Si la fecha no puede convertirse de forma fiable a `DD/MM/AAAA`, pon `null` en ese campo."
   * Para el resto de campos:

     * "Si un nombre o apellido está incompleto o ambiguo, no inventes letras; si no puedes determinar el valor, usa `null`."

5. **Síntesis de reglas clave dentro del prompt**

   El prompt textual debe contener, de forma explícita y clara, reglas como:

   * "Responde solo con un JSON válido, sin ningún texto adicional."
   * "Si un dato no es legible o hay duda, usa `null` en lugar de inventar un valor."
   * "En caso de contradicción entre imagen y OCR, prioriza la imagen."
   * "Fechas en formato `DD/MM/AAAA`; si no puedes convertirlas de forma fiable, usa `null`."

El texto final del prompt se pasará a `processor.apply_chat_template(...)` junto con la imagen `img_vlm` como entrada, formando el mensaje multimodal que se mandará a `Qwen/Qwen3-VL-8B-Instruct`.

### 4.5. Inferencia del VLM (Qwen3-VL-8B)

Objetivo: obtener del modelo un texto que represente el JSON con los campos del DNI.

Pautas:

* Usar `Qwen/Qwen3-VL-8B-Instruct` con `trust_remote_code=True` cuando sea necesario.
* Alimentar el modelo con:

  * `img_vlm` (como entrada de imagen),
  * el prompt textual construido en 4.4.
* Configurar la generación para extracción estructurada:

  * `do_sample = False` (sin sampling, determinista).
  * `max_new_tokens` ajustado (por ejemplo 128–256).
  * Parámetros como `temperature`, `top_p`, etc., no son relevantes al no samplear, pero deben establecerse de forma conservadora.

La lógica de inferencia debe encapsularse en algo como `run_qwen_vlm(img_vlm, prompt_text) -> str`.

### 4.6. Parseo de salida a JSON

Objetivo: convertir el texto generado por el modelo en un diccionario Python, de forma robusta.

Pautas:

* Intentar primero `json.loads` directamente sobre la salida.
* Si falla:

  * Buscar el primer bloque que empieza por `{` y termina en `}` usando regex.
  * Reintentar `json.loads` sobre ese bloque.
* Si aun así no es posible parsear, devolver una estructura que contenga al menos:

  * `raw_output`: texto completo devuelto por el modelo.

Esta lógica debe encapsularse en algo como `parse_model_output(text: str) -> dict`.

### 4.7. Normalización y validación de campos

Objetivo: limpiar y validar los valores del JSON obtenido:

* Mantener la estructura fija:

  ```json
  {
    "nombre": ...,
    "primer_apellido": ...,
    "segundo_apellido": ...,
    "dni": ...,
    "fecha_nacimiento": ...,
    "fecha_validez": ...,
    "sexo": ...,
    "nacionalidad": ...
  }
  ```

* Normalizar:

  * `dni`:

    * Quitar caracteres no alfanuméricos.
    * Convertir a mayúsculas.
    * Si hay 8 dígitos sin letra, calcular la letra de control.
    * Si el formato no es válido o hay dudas, usar `null`.
  * `fecha_nacimiento` y `fecha_validez`:

    * Limpiar separadores y caracteres no numéricos.
    * Intentar mapear a `DD/MM/AAAA`.
    * Si no se puede mapear de forma razonable, usar `null`.
  * Cadenas vacías o solo espacios → `null`.

* Mantener la regla de seguridad:

  * Es mejor `null` que un valor incorrecto, especialmente para DNI y fechas.

Esta lógica debe encapsularse en una función del tipo `normalize_and_validate(record: dict) -> dict`.

El resultado final será el **JSON limpio** con los datos del DNI, que se usará como salida del pipeline o se guardará en disco.

---

## 5. Estructura de ficheros esperada

El agente debe proponer y crear una estructura similar a:

```text
dni_vlm_pipeline/
├── AGENT.md                  # (este fichero, no lo modifiques)
├── requirements.txt          # dependencias mínimas necesarias
├── src/dni_pipeline/
│   ├── adapters/             # CLI, API y server
│   ├── core/                 # preprocessing/ocr/vlm/postprocess
│   ├── services/             # pipeline de orquestación
│   ├── config.py             # settings del pipeline
│   └── README.md             # instrucciones básicas de uso
```

No es obligatorio usar exactamente estos nombres, pero sí se debe mantener una **separación clara** entre:

* Preprocesado de imagen.
* OCR (docTR).
* VLM (Qwen).
* Postprocesado/validación.
* CLI / punto de entrada.

### 5.1. Contenido esperado de `README.md`

El agente debe generar un `README.md` con, al menos, los siguientes apartados:

1. **Descripción breve del proyecto**

   * Explicar en 2–3 frases qué hace el pipeline:

     * "Dado un DNI español en imagen, extrae los campos clave y devuelve un JSON estructurado usando docTR + Qwen3-VL-8B."
   * Mencionar explícitamente que:

     * todo se ejecuta en local,
     * se combina OCR + VLM.

2. **Requisitos de instalación**

   * Indicar claramente:

     * Versión mínima de Python recomendada (por ejemplo 3.10+).
     * Dependencias listadas en `requirements.txt`.
   * Mencionar la parte de GPU:

     * Que se recomienda **GPU NVIDIA con drivers y CUDA compatibles** para un rendimiento razonable.
     * Que en CPU también funcionará, pero más lento.
   * No hace falta escribir un tutorial completo de instalación de drivers, pero sí dejar claro:

     * "Debes tener PyTorch instalado con soporte GPU si quieres aprovechar una RTX 4090."

3. **Instrucciones de instalación**

   * Ejemplo de instalación mínima:

     ```bash
     pip install -r requirements.txt
     ```

   * Si se requiere alguna instalación previa especial (por ejemplo, para docTR), debe mencionarse.

4. **Ejemplos de uso (CLI)**

   * Incluir ejemplos concretos de comandos, por ejemplo:

     * Procesar una sola imagen:

       ```bash
       python dni_pipeline.py --image path/to/dni.jpg
       ```

     * Procesar un directorio de imágenes y guardar los JSON en una carpeta:

       ```bash
       python dni_pipeline.py --input-dir ./dni_images --output-dir ./dni_json
       ```

   * Explicar brevemente qué hace cada comando y dónde deja la salida (stdout / ficheros `.json`).

5. **Notas de rendimiento y comportamiento**

   * Mencionar que:

     * `Qwen/Qwen3-VL-8B-Instruct` es **perfectamente viable** en una GPU tipo RTX 4090.
     * El **primer forward** puede ser más lento debido a:

       * carga del modelo,
       * calentamiento de la GPU,
       * compilaciones internas, etc.
     * Los siguientes procesados serán más rápidos mientras el proceso siga vivo.
   * Nota sobre CPU:

     * El mismo código funcionará en CPU, pero con tiempos significativamente mayores.
     * Se recomienda GPU para procesar lotes grandes de DNIs.

6. **Limitaciones conocidas (opcional)**

   * Por ejemplo:

     * "El sistema está pensado inicialmente solo para DNIs españoles."
     * "La calidad de la lectura depende de la calidad de la imagen (fotos muy borrosas o recortadas pueden dar campos null)."

### 5.2. Contenido esperado de `requirements.txt`

El agente debe generar un `requirements.txt`:

* **Lo más minimalista posible**:

  * Solo incluir librerías que **realmente se usan** en el código generado.
  * No añadir dependencias "por si acaso".
* Debe cubrir, como mínimo:

  * `torch` / `torchvision` (según lo que use el código).
  * `transformers`.
  * `python-doctr` (y las dependencias que corresponda, si van explícitas).
  * `Pillow`.
  * `opencv-python` u `opencv-python-headless` (si se usa).
  * Cualquier otra librería estándar usada de forma explícita (por ejemplo, `numpy`).

Instrucciones explícitas al agente:

* Antes de añadir una librería a `requirements.txt`, verifica que **aparece importada en el código**.
* No añadas librerías que no se usan en ningún módulo.
* No incluyas dependencias de APIs externas (OpenAI, HF Inference API, etc.), ya que **no deben utilizarse en este proyecto**.

El objetivo de esta sección es que, con `README.md` + `requirements.txt`, un usuario pueda:

1. Crear un entorno virtual.
2. Instalar dependencias.
3. Ejecutar el pipeline con uno o dos comandos de ejemplo.
4. Entender, a nivel básico, qué esperar en términos de rendimiento en GPU vs CPU.

---

## 6. Comportamiento del agente

### 6.1. Estilo de código

* El código debe ser **legible y modular**.
* Usar nombres descriptivos en inglés para funciones, clases y variables, por ejemplo:

  * `preprocess_for_ocr`, `preprocess_for_vlm`,
  * `run_doctr_ocr`, `sort_ocr_items`, `build_ocr_block`,
  * `run_qwen_vlm`, `parse_model_output`, `normalize_and_validate`.
* Añadir **docstrings** cortos a las funciones públicas clave:

  * qué hace la función,
  * cuáles son sus entradas y salidas,
  * supuestos importantes.
* Evitar over-engineering:

  * es una POC, no una plataforma enterprise;
  * pero debe ser lo suficientemente limpia como para poder extenderla después.

### 6.2. Uso de comentarios

* Explicar las decisiones importantes en forma de comentarios donde sea útil, por ejemplo:

  * por qué se escoge una resolución concreta para `img_vlm` (512×512),
  * por qué se configura la generación del VLM sin sampling,
  * cómo se construye el bloque `[OCR]` y qué contiene.
* No comentar obviedades; priorizar comentarios que aclaren el *por qué*, no el *qué*.

### 6.3. Manejo de errores

* Manejar errores de forma robusta pero simple:

  * Si falla docTR, se puede:

    * informar de que no se ha podido obtener OCR,
    * y seguir intentando extraer información solo a partir de la imagen con el VLM.
  * Si el VLM devuelve algo no parseable como JSON, devolver una estructura con `raw_output` para facilitar el debug.
* Evitar que errores esperables (por ejemplo, imagen corrupta, JSON malformado en salida) rompan todo el proceso de forma no controlada.

### 6.4. Separación de responsabilidades

* No mezclar demasiadas responsabilidades en una sola función.

  * Preprocesado, OCR, generación del prompt, inferencia, parseo y normalización deben estar separados.
* Cada módulo núcleo (`core/preprocessing.py`, `core/ocr.py`, `core/vlm.py`, `core/postprocess.py`) debe tener una responsabilidad principal clara.
* El script `dni_pipeline.py` debe centrarse en:

  * parsear argumentos de línea de comandos,
  * llamar al pipeline completo para una imagen o un directorio,
  * mostrar/guardar los resultados.

---

## 7. No objetivos (para no complicar esta POC)

Para esta POC **no** debes implementar:

* Fine-tuning de docTR ni de Qwen.
* Soporte multiclase para muchos tipos de documentos (solo se trata el DNI español).
* Integración con bases de datos u orquestadores externos.
* API HTTP/REST; de momento solo interfaz **CLI**.

Tampoco debes usar:

* Servicios externos de OCR.
* Servicios externos de LLM.
* Llamadas a la API de Hugging Face Inference.

Todo el procesamiento debe hacerse con modelos y librerías locales.

---

## 8. Criterios de "listo"

El trabajo se considera correctamente realizado cuando:

1. `dni_pipeline.py`:

   * Acepta una ruta a imagen (`--image`) o a un directorio (`--input-dir`).
   * Imprime por consola un JSON por imagen con la estructura acordada.
   * Opcionalmente, permite guardar un `.json` por imagen en un directorio (`--output-dir`).

2. docTR:

   * Se usa para extraer texto auxiliar y este aparece en el prompt bajo un bloque tipo `[OCR] ... [/OCR]`.

3. Qwen3-VL-8B:

   * Se llama con imagen + prompt que incluye el bloque `[OCR]`.
   * Genera texto que puede parsearse a JSON con el formato acordado.

4. El postprocesado:

   * Normaliza `dni` y las fechas.
   * Maneja valores faltantes como `null` en lugar de inventar datos.

5. El código:

   * Es legible y modular.
   * Está razonablemente documentado (docstrings y comentarios donde aporten valor).
   * Es fácil de extender para futuros tipos de documento o reglas adicionales.

Cuando todos estos criterios se cumplen, el pipeline puede considerarse completo para la POC de extracción de datos de DNIs con docTR + Qwen3-VL-8B.
