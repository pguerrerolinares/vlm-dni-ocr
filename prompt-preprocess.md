# AGENT.md

Vamos a ajustar el preprocesado

## Objetivos del preprocesado

1. Corregir orientación de la imagen mediante una evaluación heurística basada en OCR (0, 90, 180, 270 grados).
2. Generar versiones específicas para:

   * OCR tradicional (imagen grande, nítida, maximizada hasta un lado largo configurable).
   * VLM (imagen cuadrada, centrada, limpia y normalizada).
3. Generar metadatos útiles para auditoría y control de calidad.
4. Opcional en fases posteriores: detección y recorte del documento dentro del frame.

## Entradas

* Una imagen en cualquier formato común.
* Parámetros configurables:

  * `ocr_max_side` (default 1600)
  * `vlm_target_size` (default 512)
  * `enable_card_crop` (bool, default false)

## Reglas del preprocesado

### 1. Carga inicial

* Corregir orientación EXIF.
* Convertir a RGB.
* Comprobar si el documento ocupa área muy pequeña; si sí, añadir warning.

### 2. Detección de orientación robusta

* Para ángulos 0°, 90°, 180°, 270°:

  * Rotar la imagen.
  * Reescalar: `max_side <= ocr_max_side`.
  * Ejecutar OCR rápido (simulado): estimar cuántos tokens tienen una "confianza" > 0.3.
* Elegir la orientación con más tokens.
* Si todos tienen muy pocos tokens (< 10), marcar `orientation_confidence = "low"`.

### 3. Generación de vista OCR

* Partir de la imagen en orientación final.
* Redimensionar con el máximo lado en `ocr_max_side`.
* Aplicar un ligero aumento de contraste (1.1–1.2).
* No aplicar sharpening agresivo.

### 4. Generación de vista VLM

* Partir de la orientación final.
* Reducir manteniendo relación de aspecto para que el mayor lado quepa en `vlm_target_size`.
* Crear un lienzo cuadrado blanco de `vlm_target_size x vlm_target_size`.
* Centrar la imagen dentro del lienzo.
* No manipular colores o contraste agresivamente.

### 5. Vista VLM zoom (opcional)

* Si `enable_card_crop` es true y hay bounding box de documento:

  * Recortar alrededor del documento.
  * Reescalar a `vlm_target_size`.
  * Centrar y rellenar si falta.
* Si no aplica, devolver `null`.

## Restricciones

* Debes responder exclusivamente en JSON siguiendo la estructura indicada.
* No devuelvas explicaciones, sólo el JSON estructurado.
* No inventes datos (usa marcadores descriptivos para las imágenes procesadas).

