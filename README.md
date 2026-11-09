# AccentLab Web v2.0 — Estación de Recolección y Análisis Acústico-Lingüístico

    [![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www
    .python.org/)
    [![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
    [![SQLite](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
    [![Praat
    Parselmouth](https://img.shields.io/badge/Praat-Parselmouth%200.4+-red.svg)](https://parselmouth.readthedocs.io/)
    [![CUDA](https://img.shields.io/badge/CUDA-VRAM%20%3C%202GB-76B900.svg)](https://developer.nvidia.com/cuda-toolki
    t)
    [![Norma
    1984](https://img.shields.io/badge/Maya-Norma%20Agosto%201984-purple.svg)](#motor-lingüístico-maya-yucateco-norma
    -de-agosto-de-1984)

    > **Sistema Full-Stack de Recolección, Segmentación Inteligente (Silero VAD), Transcripción Neuronal
    (Faster-Whisper), Motor Fonológico G2P Maya 1984, Glosador Morfosintáctico Leipzig y Análisis Acústico-Prosódico
    (Praat) para Construcción de Datasets TTS / VC Regionales.**

    ---

    ## 📑 Tabla de Contenidos

    - [1. Visión General](#1-visión-general)
      - [El Problema que Resuelve](#el-problema-que-resuelve)
      - [Estándar de Audio del Sistema](#estándar-de-audio-del-sistema)
    - [2. Arquitectura del Sistema](#2-arquitectura-del-sistema)
      - [Diagrama de Flujo de Datos](#diagrama-de-flujo-de-datos)
      - [Stack Tecnológico](#stack-tecnológico)
      - [Estructura del Proyecto](#estructura-del-proyecto)
    - [3. Instalación y Puesta en Marcha](#3-instalación-y-puesta-en-marcha)
      - [Requisitos Previos](#requisitos-previos)
      - [Encendido del Sistema](#encendido-del-sistema)
      - [Apagado Seguro](#apagado-seguro)
    - [4. Manual de Operación y Uso (Paso a Paso)](#4-manual-de-operación-y-uso-paso-a-paso)
      - [Sección 01: Perfil de Hablante y Guion Fonético](#sección-01-perfil-de-hablante-y-guion-fonético)
      - [Sección 02: Captura en Vivo y Carga de Audios Largos](#sección-02-captura-en-vivo-y-carga-de-audios-largos)
      - [Sección 03: Revisión, Transcripción y Laboratorio G2P en
      Vivo](#sección-03-revisión-transcripción-y-laboratorio-g2p-en-vivo)
      - [Sección 04: Explorador del Corpus y Exportación ZIP](#sección-04-explorador-del-corpus-y-exportación-zip)
    - [5. Motor Lingüístico Maya Yucateco (Norma 1984)](#5-motor-lingüístico-maya-yucateco-norma-1984)
      - [Normalización Ortográfica](#normalización-ortográfica)
      - [Regla Crítica del Apóstrofo: 4 Tipos Fonológicos](#regla-crítica-del-apóstrofo-4-tipos-fonológicos)
      - [Los 5 Núcleos Vocálicos y Consonantes Especiales](#los-5-núcleos-vocálicos-y-consonantes-especiales)
      - [Analizador Morfosintáctico Leipzig](#analizador-morfosintáctico-leipzig)
    - [6. Motor DSP y Acústica Experimental](#6-motor-dsp-y-acústica-experimental)
      - [Decodificación y Acondicionamiento de Señal](#decodificación-y-acondicionamiento-de-señal)
      - [F0, Curva en Semitonos y Formantes (F1/F2)](#f0-curva-en-semitonos-y-formantes-f1f2)
      - [Detección Heurística de Cortes Glotales [ʔ]](#detección-heurística-de-cortes-glotales-)
      - [Índice de Alineación Fonético-Acústica (0–100%)](#índice-de-alineación-fonético-acústica-0100)
    - [7. Pipeline de Machine Learning y Segmentación](#7-pipeline-de-machine-learning-y-segmentación)
    - [8. Formatos de Exportación para Modelos TTS](#8-formatos-de-exportación-para-modelos-tts)
    - [9. Referencia de la API REST](#9-referencia-de-la-api-rest)
    - [10. Variables de Entorno y Configuración](#10-variables-de-entorno-y-configuración)
    - [11. Solución de Problemas Frecuentes (FAQ)](#11-solución-de-problemas-frecuentes-faq)

    ---

    ## 1. Visión General

    **AccentLab Web v2.0** es una estación de trabajo de ingeniería acústico-lingüística diseñada para solventar la
    carencia de corpus de audio anotados y curados fonológicamente para **Lengua Maya Yucateca (`yua`)** bajo la
    **Norma Oficial de Agosto de 1984**, **Español Yucateco (`es_yuc`)** y patrones de alternancia de
    código/bilingüismo (`yua_es`).

    ### El Problema que Resuelve

    Los pipelines tradicionales de NLP, ASR y TTS cometen fallas críticas ante las variantes yucatecas:
    1. **Destrucción del apóstrofo (`'`):** Los tokenizadores estándar lo eliminan o lo sustituyen por
    silencios/espacios, fracturando morfemas y confundiendo consonantes eyectivas (`k'`, `ch'`, `ts'`, `p'`, `t'`),
    vocales rearticuladas (`ma'alob`, `ts'o'ok`), cierres glotales en coda (`ja'`, `ma'`) y ataques glotales (`in
    'otoch`).
    2. **Colapso tonal y vocálico:** Ignoran la oposición fonémica entre los 5 núcleos vocálicos mayas (cortas,
    largas bajas, largas altas con tono, rearticuladas y glotalizadas) y la implosiva bilabial `/ɓ/`.
    3. **Supresión DSP por WebRTC:** Los filtros de cancelación de ruido y control de ganancia del navegador borran
    los cortes glotales `[ʔ]` de corta duración (< 20 ms).
    4. **Alucinaciones ASR:** Whisper sin anclaje léxico regional traduce voces mayas a alfabetos arbitrarios o
    castellaniza modismos peninsulares (*mare*, *boxito*, *xix*, *tuch*).

    ### Estándar de Audio del Sistema
    - **Contenedor y Codificación:** WAV Mono, 16-bit Signed PCM.
    - **Tasa de Muestreo:** 24,000 Hz.
    - **Tope de Normalización:** -1.0 dBFS (ganancia controlada con tope de +26 dB para no inflar piso de ruido).

    ---

    ## 2. Arquitectura del Sistema

    ### Diagrama de Flujo de Datos


  +------------------------------------------------------------------------------+
  |                         CLIENTE WEB SPA (Navegador)                          |
  |  - Panel 01: Perfil Hablante + Selector Idioma + 29 Frases Trampa            |
  |  - Panel 02: MediaRecorder (Captura Pura DSP) + Subida de Audio Silero VAD   |
  |  - Panel 03: Curva Semitonos (30 pts) + Formantes + Probador G2P y Glosa     |
  |  - Panel 04: Explorador SQLite + Curación + Exportador en 4 Formatos TTS     |
  +--------------------------------------+---------------------------------------+
  | HTTP REST / Multipart Form
  v
  +------------------------------------------------------------------------------+
  |                         CAPA API BACKEND (FastAPI)                           |
  |      Validación Pydantic v2 · Ciclo Lifespan · Safe Migrations SQLite        |
  +-------------------+--------------------------------------+-------------------+
  |                                      |
  v                                      v
  +---------------------------------------+  +-----------------------------------+
  |     ORQUESTADOR ML (processor.py)     |  | MOTOR LINGÜÍSTICO MAYA 1984       |
  | - Silero VAD (Segmentación 3 a 10s)   |  | - MayaG2P1984: Regla del Apóstrofo|
  | - Faster-Whisper (CUDA int8_float16)  |  |   AFI (IPA) + Tokens TTS Atómicos |
  | - Prompt conditioning especializado   |  | - MayaGrammaticalAnalyzer: Glosa  |
  | - VRAM total < 2 GB                   |  |   Leipzig, Aspecto-Modo, Juegos A/B|
  +-------------------+-------------------+  +-----------------+-----------------+
  |                                        |
  +-------------------+--------------------+
  v
  +------------------------------------------------------------------------------+
  |              MOTOR DSP Y FONÉTICA ACÚSTICA (Praat & NumPy)                   |
  |  - Decodificación en 4 etapas (FFmpeg -> PyAV -> SoundFile/SciPy -> Pydub)   |
  |  - Extracción F0 Boersma (ST respecto a mediana) + LPC Burg Formantes        |
  |  - HNR (dB) + SNR energético + Detección física de Saltillos [ʔ]             |
  |  - Cálculo de Cumplimiento Fonético-Acústico G2P vs. Praat (0 a 100%)        |
  +--------------------------------------+---------------------------------------+
  | SQLAlchemy 2.0 ORM
  v
  +------------------------------------------------------------------------------+
  |                     ALMACENAMIENTO Y BASE DE DATOS LOCAL                     |
  |  - Base de Datos : data/accentlab.db (SQLite modo WAL, Foreign Keys activas) |
  |  - Audios WAV    : data/wavs/{SPEAKER_CODE}{TIMESTAMP}{UID}.wav            |
  |  - Exportaciones : data/exports/{DATASET}_{TIMESTAMP}/ (WAVs + CSV + ZIP)    |
  +------------------------------------------------------------------------------+

    ### Stack Tecnológico

    - **Backend:** FastAPI 0.115+, Uvicorn, Pydantic v2.
    - **Persistencia:** SQLite 3 (WAL mode), SQLAlchemy 2.0 (migraciones en caliente mediante `inspect`).
    - **DSP & Acústica:** Praat Parselmouth 0.4.4, NumPy 1.26+, SciPy 1.13+, PyAV, SoundFile, FFmpeg.
    - **Machine Learning / ASR:** Silero VAD 5.1+, Faster-Whisper 1.1+ (CTranslate2, CUDA cuBLAS/cuDNN v12).
    - **Frontend:** SPA nativa (HTML5, Bootstrap 5 Dark, Chart.js, Web Audio API).

    ### Estructura del Proyecto


  VoiceProject/
  ├── run.py                              # Entrada del servidor ASGI
  ├── requirements.txt                    # Dependencias Python / CUDA
  ├── .venv/                              # Entorno virtual de Python
  ├── app/
  │   ├── database.py                     # SQLite WAL y migraciones seguras
  │   ├── models.py                       # Modelos ORM (Speaker, AudioSegment, AcousticMetrics)
  │   ├── main.py                         # Endpoints FastAPI y catálogo de 29 frases trampa
  │   ├── linguistics/                    # Capa lingüística pura (latencia < 1ms, 0 MB VRAM)
  │   │   ├── maya_g2p_1984.py            # Normalizador 1984, clasificador de apóstrofos y G2P
  │   │   └── maya_grammar.py             # Analizador morfosintáctico Leipzig y diccionario léxico
  │   ├── analyzers/
  │   │   └── prosody.py                  # Extracción acústica Praat y validación de alineación
  │   ├── pipeline/
  │   │   └── processor.py                # Singleton Silero VAD + Faster-Whisper
  │   └── static/                         # Frontend Web SPA (HTML / CSS / JS)
  └── data/                               # Almacén de datos persistente
  ├── accentlab.db                    # Base de datos relacional
  ├── wavs/                           # Corpus de clips WAV 24kHz
  └── exports/                        # Paquetes ZIP exportados

    ---

    ## 3. Instalación y Puesta en Marcha

    ### Requisitos Previos

    - Windows 10/11 o Linux x86_64.
    - Python 3.10 a 3.13+.
    - GPU NVIDIA con drivers actualizados (Opcional; si no se detecta CUDA, conmuta automáticamente a CPU).

    ### Encendido del Sistema

    1. Abra una terminal PowerShell en el directorio raíz del proyecto:
       ```powershell
       cd C:\Users\gersonaam\Documents\VoiceProject
  2. Inicie el servidor con el entorno virtual:
    .\.venv\Scripts\python.exe run.py

  3. Espere a que Uvicorn indique que el servicio está activo:
    INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)

  4. Abra su navegador web (Chrome o Edge) e ingrese a:
    http://127.0.0.1:8000
  (Al pulsar "Grabar" por primera vez, otorgue permisos de acceso al micrófono).
  ### Apagado Seguro
  1. Cierre la pestaña del navegador.
  2. En la terminal donde corre Uvicorn, presione Ctrl + C para detener el proceso de forma limpia sin corromper la
  base de datos SQLite en modo WAL.
  ──────
  ## 4. Manual de Operación y Uso (Paso a Paso)
  La pantalla de trabajo está distribuida en cuatro paneles secuenciales:

    +------------------------------------------+------------------------------------------+
    |  [01] Perfil de Hablante y Guion         |  [03] Analizador Visual, G2P Maya & Glosa|
    +------------------------------------------+------------------------------------------+
    |  [02] Estudio de Grabación / VAD         |  [04] Explorador del Dataset & ZIP       |
    +------------------------------------------+------------------------------------------+
  ### Sección 01: Perfil de Hablante y Guion Fonético
  1. Registrar un Hablante:
      • Clic en el botón [+ Nuevo Hablante].
      • Capture: Nombre o Alias, Género (M, F, Otro), Edad, Subregión geográfica (Centro, Oriente, Sur, Poniente,
      Costa, Otro) e Intensidad del acento percibida (Suave, Medio, Marcado).
      • Guarde: el sistema generará automáticamente un código como YUC_ORI_001.
  2. Seleccionar Idioma de la Sesión:
      • Maya Yucateco [Norma 1984]: Filtra el carrusel de frases a maya y condiciona a Whisper con el alfabeto de
      1984.
      • Español Yucateco: Carga el catálogo de modismos y fenómenos fonéticos peninsulares.
      • Detección Automática: Útil en sesiones con informantes bilingües.
  3. Modo de Estímulo:
      • Frases Trampa: Carrusel con 29 oraciones controladas para levantar fenómenos fonológicos específicos. Incluye
      botón [Ver G2P / Glosa].
      • Habla Espontánea: Grabación libre de respuestas cortas (de 4 a 9 segundos) a preguntas abiertas.

  ### Sección 02: Captura en Vivo y Carga de Audios Largos
  #### Método A: Grabación con Micrófono
  1. Verifique que la "Captura Pura DSP" permanezca activa (inhabilita filtros WebRTC que mitigan saltillos [ʔ]).
  2. Presione [Grabar], espere medio segundo e indique al hablante que comience.
  3. Al terminar, pulse [Detener Grabación].
  4. En la escucha previa:
      • Si hubo ruido, equivocación o golpe de aire, pulse [Descartar].
      • Si la toma es limpia, pulse [Analizar Prosodia y Transcribir (Whisper)].

  #### Método B: Segmentación Inteligente VAD (Archivos Largos)

  1. Pase a la pestaña [Audio Largo (VAD)].
  2. Arrastre una entrevista o monólogo (.wav, .mp3, .m4a, .flac, .ogg) de un solo hablante.
  3. Parámetros predeterminados: Mínimo: 3.0s, Máximo: 10.0s, Padding: 180ms.
  4. Haga clic en [Segmentar con Silero VAD y Analizar Lote]. El motor cortará en los silencios naturales sin partir
  morfemas y analizará acústicamente cada clip.

  ### Sección 03: Revisión, Transcripción y Laboratorio G2P en Vivo
  1. Inspección Rápida:
      • Cortes Glotales [ʔ]: Muestra el conteo de saltillos. Al hacer clic sobre los chips de tiempo (ej. 0.65s), el
      reproductor salta a ese instante exacto.
      • HNR / SNR: Valores de SNR > 15 dB señalan una relación señal-ruido óptima.
      • Alineación G2P (%): Calificación cuantitativa del cumplimiento de los rasgos articulatorios esperados.
  2. Curación del Texto:
      • Revise la caja de transcripción. En frases trampa mayas, el sistema asigna y protege la norma ortográfica de
      1984.
      • Corrija errores si la persona usó una variante dialectal, active la casilla "Validado para exportación TTS" y
      pulse [Guardar Transcripción] (esto recalcula al vuelo el G2P, la glosa y la alineación acústica).
  3. Laboratorio Fonológico G2P en Vivo:
      • Ingrese cualquier frase maya en el probador interactivo para obtener su desglose cromático de fonemas,
      desglose de apóstrofos y glosa gramatical Leipzig.
      • El botón [Usar para Grabar] envía la frase escrita directamente al Panel 01 como guion de lectura activo.

  ### Sección 04: Explorador del Corpus y Exportación ZIP
  1. Tabla de Datos: Búsqueda textual (incluso en la representación fonética AFI), filtros por hablante, lengua o
  subregión, y edición en línea de transcripciones.
  2. Exportar Dataset:
      • Defina el nombre del lote.
      • Elija el formato de salida (orthographic_1984, ipa_phonemes, tts_tokens o dual_orth_ipa).
      • Opcionalmente fije un umbral de SNR mín. (ej. 15 dB).
      • Clic en [Exportar Dataset] y descargue el .zip resultante.

  ──────
  ## 5. Motor Lingüístico Maya Yucateco (Norma 1984)
  Ubicado en app/linguistics/, implementado en Python puro con latencia inferior a 1 ms por frase y 0 MB de consumo
  de VRAM.
  ### Normalización Ortográfica
  Normaliza a la Norma Oficial de Agosto de 1984:

  • Formato Unicode NFC para vocales acentuadas (á, é, í, ó, ú).
  • Unificación tipográfica de cualquier variante de apóstrofo (’, ‘, ʼ, ´, `) a ' ASCII.
  • Conversión determinista de grafías coloniales y pre-1984:
      • tz' y dz → ts' (ej. Tz'o'ok → Ts'o'ok).
      • tz → ts (ej. utz → uts).
      • c' → k' (ej. c'iin → k'iin).
      • c velar → k, preservando préstamos léxicos del castellano mediante lista de excepciones (Dios, camión,
      mercado).
  ### Regla Crítica del Apóstrofo: 4 Tipos Fonológicos
  │ Regla de Oro: Está estrictamente prohibido reemplazar apóstrofos por espacios o signos de puntuación. El parser
  │ asigna precedencia contextual en 4 categorías:

   Tipo   │ Denominación Fonológica │ Patrón Grafémico  │ Ejemplos          │ Símbolo AFI       │ Token TTS
  ────────┼─────────────────────────┼───────────────────┼───────────────────┼───────────────────┼────────────────────
   Tipo 1 │ Consonante Eyectiva /   │ C' (k', ch', ts', │ k'iin, ch'íich',  │ /kʼ, tʃʼ, tsʼ,    │ k_ej, ch_ej,
          │ Glotalizada             │ p', t')           │ ts'íib            │ pʼ, tʼ/           │ ts_ej, p_ej, t_ej
   Tipo 2 │ Vocal Rearticulada /    │ V₁'V₁ (misma      │ ma'alob, ts'o'ok, │ /a̰ʔa, ḛʔe, ḭʔi,   │ a_reart, e_reart,
          │ Laringalizada           │ vocal base)       │ bu'ul             │ o̰ʔo, ṵʔu/         │ i_reart, o_reart,
          │                         │                   │                   │                   │ u_reart
   Tipo 3 │ Cierre Glotal Coda /    │ V' (ante          │ ma', ja', te'elo' │ /aʔ, eʔ, iʔ, oʔ,  │ a_glot, e_glot,
          │ Posvocálico             │ consonante o      │                   │ uʔ/               │ i_glot, o_glot,
          │                         │ final)            │                   │                   │ u_glot
   Tipo 4 │ Ataque Glotal /         │ 'V o C'V          │ in 'otoch,        │ /ʔ/               │ glot_ons
          │ Frontera Morfológica    │ (consonante no    │ y'éetel           │                   │
          │                         │ eyectivable)      │                   │                   │
  ### Los 5 Núcleos Vocálicos y Consonantes Especiales

  1. Corta simple: a, e, i, o, u → /a, e, i, o, u/
  2. Larga tono bajo: aa, ee, ii, oo, uu → /àː, èː, ìː, òː, ùː/ (Tokens: aa_low, ee_low...)
  3. Larga tono alto: áa, ée, íi, óo, úu → /áː, éː, íː, óː, úː/ (Tokens: aa_high, ee_high...)
  4. Rearticulada: a'a, e'e, i'i, o'o, u'u → /a̰ʔa, ḛʔe, ḭʔi, o̰ʔo, ṵʔu/ (Tokens: a_reart...)
  5. Glotalizada corta: a', e', i', o', u' → /aʔ, eʔ, iʔ, oʔ, uʔ/ (Tokens: a_glot...)

  Consonantes representativas:
  • b → Implosiva bilabial sonora /ɓ/ (b_imp).
  • x → Fricativa postalveolar sorda /ʃ/ (x_sh).
  • j → Fricativa glotal sorda /h/ (j_h).

  ### Analizador Morfosintáctico Leipzig

  El módulo MayaGrammaticalAnalyzer realiza glosa interlineal detallada:
  • Aspecto-Modo (AM): k-/ku (incompletivo), t-/tu (completivo), táan (progresivo), ts'o'ok (terminativo), yan
  (obligativo), ko'ox (exhortativo).
  • Pronombres Ergativos / Posesivos (Juego A): in/inw-, a/aw-, u/uy-, k con soporte para prefijos epentéticos ante
  vocal.
  • Sufijos Absolutivos (Juego B): -en, -ech, -o'on, -e'ex, -o'ob.
  • Marcos Deícticos: le ...-o' (distal), le ...-a' (proximal), -e' (tópico).
  ──────
  ## 6. Motor DSP y Acústica Experimental

  La clase ProsodyAnalyzer (app/analyzers/prosody.py) combina Praat (Parselmouth) y NumPy para parametrizar la señal:

  ### Decodificación y Acondicionamiento de Señal

  • Fallback de decodificación en 4 niveles: FFmpeg CLI → PyAV → SoundFile/SciPy → Pydub.
  • Normalización física: Eliminación de componente directa DC (mono - mean(mono)), normalización a -1.0 dBFS y
  aplicación de techo de amplificación de +26 dB (evita amplificar el ruido ambiente en tomas mudas).
  ### F0, Curva en Semitonos y Formantes (F1/F2)
  • Calibración fisiológica de F0 por género:
      • Masculino (M): 65 Hz – 300 Hz.
      • Femenino (F): 100 Hz – 500 Hz.
      • Otro: 75 Hz – 450 Hz.
  • Curva normalizada de 30 puntos en Semitonos:

                     ⎛  F₀(t)   ⎞
    ST(t) = 12 · log₂⎜──────────⎟
                     ⎜F         ⎟
                     ⎝ 0,mediana⎠

  Permite comparar la entonación independientemente del tono base absoluto del hablante.

  • Formantes vocálicos: Algoritmo LPC de Burg (5 formantes, ventana de 25 ms) acotado a zonas sonoras estables (200
  ≤ F₁ ≤ 1100 Hz, 700 ≤ F₂ ≤ 3200 Hz).
  ### Detección Heurística de Cortes Glotales [ʔ]
  Detecta saltillos físicos evaluando la curva de intensidad Praat (paso de 2 ms):

  • Caída de intensidad ≥ 10.5 dB en un lapso ≤ 18 ms.
  • Concurrencia de sonoridad previa (F₀ > 0).
  • Filtro anti-fricativas con tasa de cruces por cero (ZCR ≤ 2800 Hz).
  • Periodo refractario fisiológico de 120 ms entre detecciones consecutivas.

  ### Índice de Alineación Fonético-Acústica (0–100%)

  Evalúa el grado de concordancia entre la señal acústica y las expectativas fonológicas del G2P:
  1. Sub-puntaje de Glotalización (45%): Confronta saltillos detectados y valores de HNR propios de voz laringalizada
  con las eyectivas y vocales rearticuladas esperadas.
  2. Sub-puntaje Tonal (35%): Verifica excursiones tonales positivas (≥ +1.5 ST) si la oración contiene vocales de
  tono alto (áa), o estabilidad si son bajas.
  3. Sub-puntaje de Fonación y Calidad (20%): Pondera la fracción sonora continua y la relación señal-ruido (SNR).
  ──────
  ## 7. Pipeline de Machine Learning y Segmentación

  • Silero VAD v5: Ejecución en CPU (0 MB VRAM). Segmentación adaptativa orientada a TTS (cortes entre 3.0s y 10.0s
  en valles de energía inter-léxica con silencio > 280 ms).
  • Faster-Whisper: Ejecución sobre CUDA en int8_float16 con footprint en VRAM de ~500 MB (modelo small). Si CUDA no
  está disponible, conmuta a CPU int8.
  • Inyección de Prompts Regionales:
      • Modo Maya (yua): Inyecta DEFAULT_MAYA_1984_PROMPT con léxico ortográfico de 1984, impidiendo que Whisper
      alucine en otros alfabetos.
      • Modo Español (es_yuc): Inyecta DEFAULT_REGIONAL_PROMPT con modismos peninsulares.
  • Protección de Transcripción: En oraciones de lectura dirigida (Frases Trampa), el texto se homologa a la
  ortografía oficial 1984 mientras que la salida cruda de Whisper se conserva en raw_transcript para fines
  comparativos.
  ──────
  ## 8. Formatos de Exportación para Modelos TTS

  El endpoint POST /api/export-dataset genera un archivo .zip que contiene la carpeta wavs/ y el archivo metadata.csv
  (UTF-8 sin cabecera), adaptado a distintas arquitecturas neuronales:
  ### 1. orthographic_1984 (Recomendado para XTTS-v2 / F5-TTS)

  Texto estandarizado en ortografía oficial de 1984 / español regional:

    wavs/YUC_ORI_001_20261109_001.wav|Ma'alob k'iin|YUC_ORI_001
    wavs/YUC_ORI_001_20261109_002.wav|Jach k'a'ana'an in kanik maaya t'aan.|YUC_ORI_001

  ### 2. ipa_phonemes (Recomendado para VITS / Coqui IPA)

  Transcripción pura en Alfabeto Fonético Internacional:
    wavs/YUC_ORI_001_20261109_001.wav|ma̰ʔaloɓ kʼìːn|YUC_ORI_001
    wavs/YUC_ORI_001_20261109_002.wav|hatʃ kʼa̰ʔanaʔan in kanik maːja tʼàːn|YUC_ORI_001
  ### 3. tts_tokens (Recomendado para vocabularios fonéticos custom en Piper)
  Secuencia fonémica desambiguada:
    wavs/YUC_ORI_001_20261109_001.wav|m a_reart l o b_imp | k_ej ii_low n|YUC_ORI_001
    wavs/YUC_ORI_001_20261109_002.wav|j_h a ch | k_ej a_reart n a_glot a n | i n | k a n i k | m aa_low y a | t_ej
    aa_low n|YUC_ORI_001
  ### 4. dual_orth_ipa (Alineación Dual)
    wavs/YUC_ORI_001_20261109_001.wav|Ma'alob k'iin|ma̰ʔaloɓ kʼìːn|YUC_ORI_001

  Adicionalmente, se incluye manifest.json con todos los metadatos acústicos y lingüísticos de cada clip (incluyendo
  puntajes de alineación fonética y glosa Leipzig completa).
  ──────
  ## 9. Referencia de la API REST
   Método │ Endpoint                      │ Descripción
  ────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────
   GET    │ /                             │ Retorna la aplicación web SPA.
   GET    │ /api/prompts                  │ Catálogo de frases trampa con filtros ?language=yua|es_yuc y categoría.
   POST   │ /api/linguistics/analyze-maya │ Analiza un texto en maya: devuelve ortografía 1984, AFI, tokens TTS,
          │                               │ clasificación de apóstrofos y glosa Leipzig.
   GET    │ /api/speakers                 │ Lista de hablantes registrados con métricas acumuladas.
   POST   │ /api/speakers                 │ Registra un nuevo informante y autogenera su speaker_code.
   DELETE │ /api/speakers/{id}            │ Borrado en cascada del hablante, sus clips en BD y los archivos WAV en
          │                               │ disco.
   GET    │ /api/stats                    │ KPIs globales del corpus (clips, duración limpia, distribución por
          │                               │ lengua/región).
   POST   │ /api/record-clip              │ Recepción de audio en vivo vía multipart/form-data, análisis DSP y
          │                               │ guardado.
   POST   │ /api/upload-long-audio        │ Recepción de archivo de larga duración y segmentación inteligente Silero
          │                               │ VAD.
   GET    │ /api/segments                 │ Consulta de clips con filtros por hablante, subregión, idioma y búsqueda
          │                               │ AFI.
   GET    │ /api/segments/{id}            │ Detalle completo de métricas acústicas y glosa de un segmento.
   PUT    │ /api/segments/{id}/transcript │ Modifica la transcripción y recalcula dinámicamente G2P, glosa y métrica
          │                               │ de alineación.
   DELETE │ /api/segments/{id}            │ Elimina el registro del clip y remueve el archivo WAV de almacenamiento.
   POST   │ /api/export-dataset           │ Empaqueta y exporta el corpus filtrado en archivo .zip en cualquiera de
          │                               │ los 4 formatos.
  ──────
  ## 10. Variables de Entorno y Configuración

  Configurables mediante archivo .env o variables de entorno del sistema:

   Variable                      │ Valor por Defecto        │ Descripción
  ───────────────────────────────┼──────────────────────────┼────────────────────────────────────────────────────────
   ACCENTLAB_HOST                │ 127.0.0.1                │ Dirección IP de escucha de Uvicorn.
   ACCENTLAB_PORT                │ 8000                     │ Puerto HTTP del servicio.
   ACCENTLAB_RELOAD              │ 1                        │ Activa la recarga automática en cambios de código (1 o
                                 │                          │ 0).
   ACCENTLAB_DATABASE_URL        │ sqlite:///data/accentlab │ URL de conexión de SQLAlchemy.
                                 │ .db                      │
   ACCENTLAB_WHISPER_MODEL       │ small                    │ Modelo de Whisper (tiny, base, small, medium).
   ACCENTLAB_DEVICE              │ auto                     │ Dispositivo de inferencia (auto, cuda, cpu).
   ACCENTLAB_INITIAL_PROMPT      │ (Léxico regional)        │ Prompt contextual para ASR en español yucateco.
   ACCENTLAB_MAYA_INITIAL_PROMPT │ (Léxico 1984)            │ Prompt contextual para ASR en lengua maya.
  ──────
  ## 11. Solución de Problemas Frecuentes (FAQ)

  #### 1. El navegador indica "No se pudo acceder al micrófono"

  • Compruebe la conexión del dispositivo de captura.
  • En la barra de navegación (icono de candado o configuración junto a 127.0.0.1:8000), asegúrese de que el permiso
  de Micrófono esté configurado en "Permitir" y recargue la página (F5).

  #### 2. La primera grabación tarda varios segundos en analizar

  • Es el comportamiento esperado: en el primer clip, el orquestador realiza el encendido en frío (lazy loading) de
  los pesos de Whisper y Silero VAD en la memoria GPU/CPU. Los procesamientos posteriores se ejecutan en tiempo casi
  real.

  #### 3. Error al exportar: "No existen clips que cumplan los criterios"

  • Verifique en el Panel 04 que existan segmentos con la casilla Validado activa y con transcripción no vacía.
  • Asegúrese de que los selectores de filtro no apunten a una combinación sin registros guardados.

  #### 4. Pérdida de conexión o sitio inaccesible

  • Si la consola muestra que Uvicorn se cerró, vuelva a ejecutar .\.venv\Scripts\python.exe run.py. Las grabaciones
  y métricas previas están a salvo en la base de datos relacional.
  ──────
  ## Licencia y Citación

  Desarrollado para la investigación, preservación y síntesis neuronal de la Lengua Maya Yucateca y variantes
  dialectales del Español Peninsular de México.