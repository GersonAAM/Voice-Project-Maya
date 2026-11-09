"""
Analizador Morfosintáctico, Glosador Interlineal (Estilo Leipzig) y Traductor
Semántico-Pragmático para Lengua Maya Yucateca (`yua`, Norma 1984) y Español Yucateco.

Capacidades principales:
1. Detección automática de variedad lingüística:
   - `"yua"`   : Maya Yucateco (Norma Ortográfica de Agosto de 1984).
   - `"es_yuc"`: Español Yucateco / Mexicano (con identificación de sustrato maya).
   - `"yua_es"`: Bilingüe / Alternancia de código (*code-switching* Maya-Español).
2. Segmentación morfológica productiva del Maya Yucateco:
   - Marcadores de Aspecto-Modo (AM): `k-` (INC), `t-` (CMP), `ts'o'ok` (TERM),
     `táan` (PROG), `yan` (OBLIG), `taak` (DESID), `je'el ...-e'` (ASEV), `ko'ox` (EXHORT).
   - Pronombres Ergativos (Juego A): `in / inw- / w-` (1SG), `a / aw- / w-` (2SG),
     `u / uy- / y-` (3SG), `k` (1PL), y marcos discontinuos `a ...-e'ex` (2PL), `u ...-o'ob` (3PL).
   - Sufijos Absolutivos (Juego B): `-en` (1SG), `-ech` (2SG), `-∅` (3SG),
     `-o'on` (1PL), `-e'ex` (2PL), `-o'ob` (3PL).
   - Sufijos derivativos, de estatus verbal y clíticos deícticos/tópico:
     `-tal` (INCH incoativo), `-ik` (INC.TR), `-aj` (CMP.TR), `-ej` (SUBJ.TR),
     `-il` (REL relacional/abstracto), `-a'` (PROX), `-o'` (DIST), `-e'` (TOP/MARCO).
3. Diccionario de construcciones idiomáticas y contrastes pragmáticos entre variantes
   equivalentes (ej. `Ma'alob k'iin` vs. `Yutsil ja'atskab` vs. `Ma'alob u sáastal k'iin`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.linguistics.maya_g2p_1984 import MayaG2P1984


# ==============================================================================
# 1. Morfemas Funcionales: Aspecto-Modo, Juego A (Ergativo) y Juego B (Absolutivo)
# ==============================================================================
ASPECT_MOOD_MARKERS: dict[str, dict[str, str]] = {
    "k": {
        "gloss": "INC",
        "category": "AM (Aspecto-Modo)",
        "meaning": "Aspecto incompletivo (habitual / presente / general)",
    },
    "t": {
        "gloss": "CMP",
        "category": "AM (Aspecto-Modo)",
        "meaning": "Aspecto completivo perfectivo transitivo",
    },
    "ts'o'ok": {
        "gloss": "TERM",
        "category": "AM (Aspecto-Modo)",
        "meaning": "Aspecto terminativo ('ya terminó de / ya...')",
    },
    "táan": {
        "gloss": "PROG",
        "category": "AM (Aspecto-Modo)",
        "meaning": "Aspecto progresivo o durativo ('estar haciendo')",
    },
    "tun": {
        "gloss": "PROG.3SG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Contracción oral de `táan u` ('él/ella está...')",
    },
    "tin": {
        "gloss": "CMP/PROG-1SG.ERG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Contracción de `t-in` (completivo 1SG) o `táan in` (progresivo 1SG)",
    },
    "ta": {
        "gloss": "CMP-2SG.ERG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Contracción de `t-a` (completivo + 2SG ergativo)",
    },
    "tu": {
        "gloss": "CMP-3SG.ERG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Contracción de `t-u` (completivo + 3SG ergativo; o prep. `ti' u`)",
    },
    "kin": {
        "gloss": "INC-1SG.ERG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Fusión de `k-in` (incompletivo + 1SG ergativo: 'yo suelo/hago')",
    },
    "ka": {
        "gloss": "INC-2SG.ERG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Fusión de `k-a` (incompletivo + 2SG ergativo: 'tú haces/vas')",
    },
    "ku": {
        "gloss": "INC-3SG.ERG",
        "category": "AM+ERG (Contracción)",
        "meaning": "Fusión de `k-u` (incompletivo + 3SG ergativo: 'él/ella hace')",
    },
    "yan": {
        "gloss": "OBLIG/EXIST",
        "category": "AM / Predicado Existencial",
        "meaning": "Modo obligativo ('tener que / haber de') o existencial ('hay')",
    },
    "yaan": {
        "gloss": "EXIST/POS",
        "category": "Predicado Existencial",
        "meaning": "Existencial o posesivo ('hay / existe / tener')",
    },
    "taak": {
        "gloss": "DESID",
        "category": "AM (Modal Desiderativo)",
        "meaning": "Deseo o voluntad ('querer / tener ganas de')",
    },
    "je'el": {
        "gloss": "ASEV",
        "category": "AM (Modo Aseverativo)",
        "meaning": "Compromiso aseverativo ('seguro que sí...'), exige clítico final `-e'`",
    },
    "je'ele'": {
        "gloss": "ASEV-TOP",
        "category": "Respuesta Aseverativa",
        "meaning": "'Sí, claro que sí / por supuesto'",
    },
    "ko'ox": {
        "gloss": "EXHORT.1PL",
        "category": "AM (Exhortativo)",
        "meaning": "¡Vamos! (1ª persona plural inclusiva)",
    },
    "ko'one'ex": {
        "gloss": "EXHORT-1PL.INCL",
        "category": "AM (Exhortativo Plural)",
        "meaning": "¡Vámonos todos!",
    },
    "sáam": {
        "gloss": "PAS.REC",
        "category": "AM (Pasado Reciente)",
        "meaning": "Hace rato (hoy mismo)",
    },
    "úuch": {
        "gloss": "PAS.REM",
        "category": "AM (Pasado Remoto)",
        "meaning": "Hace mucho tiempo",
    },
}

ERGATIVE_SET_A: dict[str, dict[str, str]] = {
    "in": {
        "gloss": "1SG.ERG",
        "category": "Pronombre Juego A (Ergativo/Posesivo)",
        "meaning": "1ª persona singular ('yo' sujeto / 'mi' poseedor)",
    },
    "inw": {
        "gloss": "1SG.ERG.PREV",
        "category": "Pronombre Juego A Prevocálico",
        "meaning": "1ª persona singular ante raíz iniciada en vocal ('yo / mi')",
    },
    "a": {
        "gloss": "2SG.ERG",
        "category": "Pronombre Juego A (Ergativo/Posesivo)",
        "meaning": "2ª persona singular ('tú' sujeto / 'tu' poseedor)",
    },
    "aw": {
        "gloss": "2SG.ERG.PREV",
        "category": "Pronombre Juego A Prevocálico",
        "meaning": "2ª persona singular ante raíz iniciada en vocal ('tú / tu')",
    },
    "u": {
        "gloss": "3SG.ERG",
        "category": "Pronombre Juego A (Ergativo/Posesivo)",
        "meaning": "3ª persona singular ('él/ella' sujeto / 'su' poseedor)",
    },
    "uy": {
        "gloss": "3SG.ERG.PREV",
        "category": "Pronombre Juego A Prevocálico",
        "meaning": "3ª persona singular ante raíz iniciada en vocal ('él/ella / su')",
    },
}

ABSOLUTIVE_SET_B_SUFFIXES: list[tuple[str, str, str]] = [
    # (sufijo, glosa_leipzig, descripcion)
    ("o'on", "1PL.ABS", "Sufijo Juego B 1ª plural ('nosotros / nos')"),
    ("e'ex", "2PL.ABS/ERG", "Sufijo Juego B / pluralizador 2ª plural ('ustedes')"),
    ("o'ob", "3PL.ABS/PL", "Sufijo Juego B 3ª plural o pluralizador nominal ('ellos / -s')"),
    ("ech", "2SG.ABS", "Sufijo Juego B 2ª singular ('tú eres / a ti')"),
    ("en", "1SG.ABS", "Sufijo Juego B 1ª singular ('yo soy / a mí')"),
]

DEICTIC_AND_FRAME_CLITICS: list[tuple[str, str, str]] = [
    ("a'", "PROX", "Clítico deíctico proximal ('este / aquí', requiere determinante `le`)"),
    ("o'", "DIST", "Clítico deíctico distal ('ese / aquel / ahí', requiere `le`)"),
    ("e'", "TOP/MARCO", "Clítico de marco tópico o cierre aseverativo (`le ...-e'`, `je'el ...-e'`)"),
    ("i'", "LOC.ANA", "Clítico locativo anafórico ('allí / de eso')"),
]

DERIVATIONAL_AND_STATUS_SUFFIXES: list[tuple[str, str, str]] = [
    ("chajal", "INCH.INTR", "Sufijo incoativo intransitivo ('convertirse en')"),
    ("tal", "INCH.INC", "Sufijo incoativo incompletivo ('volverse / clarear / ponerse')"),
    ("ik", "INC.TR", "Sufijo de estatus incompletivo transitivo"),
    ("aj", "CMP.TR", "Sufijo de estatus completivo transitivo"),
    ("ej", "SUBJ.TR", "Sufijo de estatus subjuntivo/imperativo transitivo"),
    ("il", "REL", "Sufijo relacional / abstractivo / pertenencia intrínseca"),
]


# ==============================================================================
# 2. Léxico Nuclear Maya Yucateco (Norma 1984) + Sustrato en Español Yucateco
# ==============================================================================
MAYA_LEXICON: dict[str, dict[str, str]] = {
    # --- Saludos, Cortesía y Estado Anímico ---
    "ma'alob": {
        "gloss": "bueno/bien",
        "category": "ADJ / ADV",
        "meaning": "Bueno, bien, correcto (etim. `ma'` NEG + `-alob` de `lob` 'malo': 'no-malo')",
        "decomposition": "ma' (NEG) + -alob (de lob 'malo')",
    },
    "uts": {
        "gloss": "bueno/bondad",
        "category": "ADJ / N",
        "meaning": "Bueno, bondadoso, virtud, bien moral o agradable",
    },
    "yutsil": {
        "gloss": "3SG-bondad-REL",
        "category": "N.REL / ADJ",
        "meaning": "Bondad, hermosura, cualidad buena o bendecida (`y-uts-il`)",
        "decomposition": "y- (3SG.PREV) + uts ('bueno') + -il (REL)",
    },
    "k'iin": {
        "gloss": "sol/día",
        "category": "N",
        "meaning": "Sol, día, tiempo, época o festividad",
    },
    "ja'atskab": {
        "gloss": "mañana.temprano",
        "category": "ADV / N",
        "meaning": "Mañana temprana, madrugada, primeras horas del día",
    },
    "sáas": {
        "gloss": "claro/luz",
        "category": "RAÍZ.ADJ/N",
        "meaning": "Claridad, luz diurna, transparente",
    },
    "sáastal": {
        "gloss": "clarear-INCH.INC",
        "category": "V.INTR.INCH",
        "meaning": "Amanecer, clarear el alba (`sáas` 'luz/claro' + `-tal` incoativo)",
        "decomposition": "sáas ('luz/claro') + -tal (INCH.INC 'volverse')",
    },
    "chi'in": {
        "gloss": "inclinarse.sol",
        "category": "RAÍZ.V",
        "meaning": "Declinar o inclinarse el sol pasado el mediodía",
    },
    "chi'inil": {
        "gloss": "declive.solar-REL",
        "category": "N.REL",
        "meaning": "Tarde / atardecer (inclinación vespertina del sol: `chi'in-il`)",
        "decomposition": "chi'in ('declinar') + -il (REL)",
    },
    "áak'ab": {
        "gloss": "noche",
        "category": "N",
        "meaning": "Noche, oscuridad nocturna",
    },
    "sáamal": {
        "gloss": "mañana.futuro",
        "category": "ADV.TEMP",
        "meaning": "Mañana (día siguiente)",
    },
    "bejla'e'": {
        "gloss": "hoy-TOP",
        "category": "ADV.TEMP",
        "meaning": "Hoy, en este día (`bejla'` + `-e'`)",
        "decomposition": "bejla' ('hoy') + -e' (TOP)",
    },
    "toj": {
        "gloss": "recto/sano",
        "category": "ADJ.EST",
        "meaning": "Recto, derecho, cabal, en buen estado de salud",
    },
    "óol": {
        "gloss": "ánimo/espíritu",
        "category": "N.INAL",
        "meaning": "Centro anímico, voluntad, aliento vital, estado de ánimo",
    },
    "óolal": {
        "gloss": "ánimo-REL",
        "category": "N.ABSTR",
        "meaning": "Voluntad, intención o sentimiento cordial (`óol-al`)",
        "decomposition": "óol ('ánimo') + -al (ABSTR)",
    },
    "nib": {
        "gloss": "agradecer",
        "category": "RAÍZ.V/N",
        "meaning": "Agradecimiento, reconocer un favor (maya clásico/literario)",
    },
    "ki'imak": {
        "gloss": "alegre/dulce",
        "category": "ADJ.AFECT",
        "meaning": "Alegre, contento, grato (en `ki'imak in wóol`: 'estoy feliz')",
    },
    "bo'ol": {
        "gloss": "pagar/pago",
        "category": "N / V",
        "meaning": "Pago, retribución",
    },
    "bo'ot": {
        "gloss": "pagar",
        "category": "RAÍZ.V.TR",
        "meaning": "Pagar, recompensar, retribuir",
    },
    "bo'otik": {
        "gloss": "pagar-INC.TR",
        "category": "V.TR.INC",
        "meaning": "Paga / recompensa (en `Dios bo'otik`: 'Dios lo pague / gracias')",
        "decomposition": "bo'ot ('pagar') + -ik (INC.TR)",
    },
    "dios": {
        "gloss": "Dios(ESP)",
        "category": "N.PRÉSTAMO",
        "meaning": "Dios (préstamo hispano integrado desde el periodo colonial)",
    },
    # --- Interrogativos, Negación y Partículas ---
    "bix": {
        "gloss": "cómo",
        "category": "PART.INT",
        "meaning": "¿Cómo? / de qué manera",
    },
    "ba'ax": {
        "gloss": "qué",
        "category": "PRON.INT",
        "meaning": "¿Qué? / ¿qué cosa? / lo que",
    },
    "máax": {
        "gloss": "quién",
        "category": "PRON.INT",
        "meaning": "¿Quién? / la persona que",
    },
    "tu'ux": {
        "gloss": "dónde",
        "category": "ADV.INT",
        "meaning": "¿Dónde? / ¿a dónde? / lugar donde",
    },
    "bajux": {
        "gloss": "cuánto.precio",
        "category": "PART.INT",
        "meaning": "¿Cuánto cuesta?",
    },
    "buka'aj": {
        "gloss": "cuánto.cantidad",
        "category": "PART.INT",
        "meaning": "¿Cuánto? / ¿de qué tamaño?",
    },
    "ba'axten": {
        "gloss": "por.qué",
        "category": "ADV.INT",
        "meaning": "¿Por qué? (`ba'ax` 'qué' + `ti'al/ten`)",
    },
    "ma'": {
        "gloss": "NEG",
        "category": "PART.NEG",
        "meaning": "No (negación proposicional o respuesta negativa con cierre glotal final)",
    },
    "mix": {
        "gloss": "tampoco/ni",
        "category": "PART.NEG.ENF",
        "meaning": "Ni, tampoco, ninguno (ej. `mix ba'al`: 'nada / de nada')",
    },
    "jach": {
        "gloss": "muy/mucho",
        "category": "ADV.INTENS",
        "meaning": "Muy, sumamente, de verdad",
    },
    "chén": {
        "gloss": "solo/nomás",
        "category": "ADV.FOC",
        "meaning": "Solamente, nada más (`chéen` / `chén`)",
    },
    "chéen": {
        "gloss": "solo/nomás",
        "category": "ADV.FOC",
        "meaning": "Solamente, únicamente",
    },
    " xan": {
        "gloss": "también",
        "category": "ADV",
        "meaning": "También, igualmente",
    },
    "xan": {
        "gloss": "también",
        "category": "ADV",
        "meaning": "También, igualmente",
    },
    "le": {
        "gloss": "DET",
        "category": "DET",
        "meaning": "Determinante demostrativo/definido ('el/la/este/ese', exige `-a'/-o'/-e'`)",
    },
    "ti'": {
        "gloss": "PREP.LOC/DAT",
        "category": "PREP",
        "meaning": "A, en, para, hacia, de (preposición general maya)",
    },
    "ich": {
        "gloss": "dentro/ojo",
        "category": "PREP / N",
        "meaning": "Dentro de, en (`ichil`), o sustantivo 'ojo/rostro/fruto'",
    },
    "ichil": {
        "gloss": "dentro-REL",
        "category": "PREP.REL",
        "meaning": "Dentro de, en medio de",
        "decomposition": "ich ('interior') + -il (REL)",
    },
    "yéetel": {
        "gloss": "con/y",
        "category": "CONJ / PREP.REL",
        "meaning": "Y, con, junto con (`y-éet-el`)",
    },
    "y'éetel": {
        "gloss": "con/y",
        "category": "CONJ / PREP.REL",
        "meaning": "Y, con (variante con frontera glotal explícita `y'éetel`)",
    },
    "k'a'ana'an": {
        "gloss": "importante/necesario",
        "category": "ADJ.EST",
        "meaning": "Necesario, importante, urgente (doble vocal rearticulada `/a̰ʔa/`)",
    },
    # --- Sustantivos Cotidianos, Naturaleza, Casa y Cultura ---
    "bej": {
        "gloss": "camino",
        "category": "N",
        "meaning": "Camino, sendero, rumbo",
    },
    "beel": {
        "gloss": "camino.POS",
        "category": "N.POS",
        "meaning": "Camino poseído, curso de vida o salud (`a beel`: 'tu camino/estado')",
        "decomposition": "bej ('camino') + -el (POS.INAL)",
    },
    "ja'": {
        "gloss": "agua",
        "category": "N",
        "meaning": "Agua, lluvia",
    },
    "sa'": {
        "gloss": "atole",
        "category": "N",
        "meaning": "Atole de maíz nuevo",
    },
    "k'áak'": {
        "gloss": "fuego",
        "category": "N",
        "meaning": "Fuego, lumbre (vocal larga tono alto + doble eyectiva `k'`)",
    },
    "iik'": {
        "gloss": "viento/aire",
        "category": "N",
        "meaning": "Viento, aire, aliento, espíritu",
    },
    "lu'um": {
        "gloss": "tierra",
        "category": "N",
        "meaning": "Tierra, suelo, terreno",
    },
    "ka'an": {
        "gloss": "cielo",
        "category": "N",
        "meaning": "Cielo, firmamento (contrasta con `kaan` 'serpiente' y `káan`)",
    },
    "kaan": {
        "gloss": "serpiente/cuatro",
        "category": "N / NUM",
        "meaning": "Serpiente (o variante de número cuatro `kan`)",
    },
    "naj": {
        "gloss": "casa",
        "category": "N",
        "meaning": "Casa (construcción física)",
    },
    "na'": {
        "gloss": "madre",
        "category": "N",
        "meaning": "Madre, mamá",
    },
    "otoch": {
        "gloss": "hogar",
        "category": "N.INAL",
        "meaning": "Hogar, morada propia (siempre poseído: `in wotoch`, `a wotoch`, `u yotoch`)",
    },
    "kaaj": {
        "gloss": "pueblo",
        "category": "N",
        "meaning": "Pueblo, localidad, comunidad",
    },
    "kool": {
        "gloss": "milpa",
        "category": "N",
        "meaning": "Milpa tradicional agrícola",
    },
    "waaj": {
        "gloss": "tortilla/pan",
        "category": "N",
        "meaning": "Tortilla de maíz, tamal o pan",
    },
    "bu'ul": {
        "gloss": "frijol",
        "category": "N",
        "meaning": "Frijol",
    },
    "píib": {
        "gloss": "horno.subterráneo",
        "category": "N",
        "meaning": "Horno subterráneo tradicional maya o alimento horneado bajo tierra",
    },
    "k'éek'en": {
        "gloss": "cerdo",
        "category": "N",
        "meaning": "Cerdo, puerco",
    },
    "peek'": {
        "gloss": "perro",
        "category": "N",
        "meaning": "Perro",
    },
    "miis": {
        "gloss": "gato/barrer",
        "category": "N / V",
        "meaning": "Gato (o acción de barrer)",
    },
    "ch'íich'": {
        "gloss": "pájaro",
        "category": "N",
        "meaning": "Ave, pájaro (doble africada eyectiva `ch'` + tono alto `íi`)",
    },
    "paal": {
        "gloss": "niño/hijo",
        "category": "N",
        "meaning": "Niño, muchacho, hijo/hija",
    },
    "winik": {
        "gloss": "hombre/persona",
        "category": "N",
        "meaning": "Ser humano, persona, hombre",
    },
    "ko'olel": {
        "gloss": "mujer/señora",
        "category": "N",
        "meaning": "Mujer, dama, señora",
    },
    "maaya": {
        "gloss": "maya",
        "category": "N / ADJ",
        "meaning": "Pueblo, cultura o lengua maya (`maaya t'aan`)",
    },
    "t'aan": {
        "gloss": "palabra/lengua/hablar",
        "category": "N / V.INTR",
        "meaning": "Palabra, idioma, voz o acto de hablar",
    },
    "k'aaba'": {
        "gloss": "nombre",
        "category": "N",
        "meaning": "Nombre (`bix a k'aaba'`: '¿cómo te llamas?')",
    },
    # --- Raíces Verbales Frecuentes ---
    "bin": {
        "gloss": "ir",
        "category": "V.INTR",
        "meaning": "Ir, dirigirse hacia un lugar",
    },
    "taal": {
        "gloss": "venir",
        "category": "V.INTR",
        "meaning": "Venir, llegar aquí",
    },
    "janal": {
        "gloss": "comer/comida",
        "category": "V.INTR / N",
        "meaning": "Comer o alimento/comida",
    },
    "uk'ul": {
        "gloss": "beber/bebida",
        "category": "V.INTR / N",
        "meaning": "Beber o bebida matutina",
    },
    "wenel": {
        "gloss": "dormir",
        "category": "V.INTR",
        "meaning": "Dormir",
    },
    "meyaj": {
        "gloss": "trabajar/trabajo",
        "category": "V.INTR / N",
        "meaning": "Trabajar, laborar, oficio",
    },
    "báaxal": {
        "gloss": "jugar",
        "category": "V.INTR",
        "meaning": "Jugar, bromear",
    },
    "a'al": {
        "gloss": "decir",
        "category": "RAÍZ.V.TR",
        "meaning": "Decir, contar, expresar (`w-a'al-ik`)",
    },
    "beet": {
        "gloss": "hacer",
        "category": "RAÍZ.V.TR",
        "meaning": "Hacer, realizar, construir (`beet-ik`, `beet-aj`)",
    },
    "il": {
        "gloss": "ver",
        "category": "RAÍZ.V.TR",
        "meaning": "Ver, mirar, visitar (`w-il-ik`, `y-il-aj`)",
    },
    "kan": {
        "gloss": "aprender",
        "category": "RAÍZ.V.TR",
        "meaning": "Aprender, adquirir conocimiento (`kan-ik`)",
    },
    "kaanbal": {
        "gloss": "aprender.INTR",
        "category": "V.INTR.ANTIPAS",
        "meaning": "Estudiar, aprender",
    },
    "k'áat": {
        "gloss": "querer/preguntar",
        "category": "N.MODAL / RAÍZ.V",
        "meaning": "Deseo o petición (`in k'áat`: 'yo quiero'; `ba'ax a k'áat`: '¿qué deseas?')",
    },
    "ojel": {
        "gloss": "saber",
        "category": "N.COGN / V.EST",
        "meaning": "Saber, tener conocimiento (`in wojel`: 'lo sé')",
    },
    "ts'íib": {
        "gloss": "escribir/escritura",
        "category": "N / V",
        "meaning": "Escritura, escribir, trazo",
    },
    "xook": {
        "gloss": "leer/contar/estudio",
        "category": "N / V",
        "meaning": "Leer, contar, estudiar",
    },
    "maan": {
        "gloss": "comprar/pasar",
        "category": "V",
        "meaning": "Comprar (`maan`) o pasar/transitar (`máan`)",
    },
    "máan": {
        "gloss": "pasar/andar",
        "category": "V.INTR",
        "meaning": "Pasar, caminar por un sitio",
    },
    "kuxtal": {
        "gloss": "vida/vivir",
        "category": "N / V.INTR",
        "meaning": "Vida, existencia, vivir",
    },
    "sáansamal": {
        "gloss": "cada.día",
        "category": "ADV.TEMP",
        "meaning": "Todos los días, diariamente (`sáan-samal`)",
    },
    "xíimbal": {
        "gloss": "caminar/pasear",
        "category": "V.INTR / N",
        "meaning": "Caminar, pasear, hacer visita",
    },
    "te'elo'": {
        "gloss": "allá-DIST",
        "category": "ADV.DEM",
        "meaning": "Allá, por aquel lugar (`te'el-o'`)",
    },
    "te'ela'": {
        "gloss": "aquí-PROX",
        "category": "ADV.DEM",
        "meaning": "Aquí, en este lugar (`te'el-a'`)",
    },
    "te'": {
        "gloss": "LOC.DEM",
        "category": "PART.LOC",
        "meaning": "Ahí/aquí en (demostrativo locativo que exige clítico `-a'` o `-o'`)",
    },
    "k'aay": {
        "gloss": "cantar/canto",
        "category": "V.INTR / N",
        "meaning": "Cantar o canción/canto",
    },
    "yóok'ol": {
        "gloss": "3SG-sobre-REL",
        "category": "N.REL.LOC",
        "meaning": "Sobre, encima de (`y-óok'-ol`)",
        "decomposition": "y- (3SG.PREV) + óok' ('encima') + -ol (REL)",
    },
    "ts'íibt": {
        "gloss": "escribir.TR",
        "category": "RAÍZ.V.TR",
        "meaning": "Escribir algo transitivamente (`ts'íib-t-ik`)",
    },
    "kaajal": {
        "gloss": "pueblo-REL",
        "category": "N.POS",
        "meaning": "Pueblo natal o comunidad poseída (`in kaajal`: 'mi pueblo')",
        "decomposition": "kaaj ('pueblo') + -al (REL.POS)",
    },
    "p'o'": {
        "gloss": "lavar",
        "category": "V.TR / IMP",
        "meaning": "Lavar (`p'o' a k'ab`: 'lávate las manos')",
    },
    "k'ab": {
        "gloss": "mano/brazo",
        "category": "N",
        "meaning": "Mano, brazo o rama",
    },
    "táanil": {
        "gloss": "delante/antes-REL",
        "category": "ADV / PREP.REL",
        "meaning": "Antes, primero, adelante (`táan-il`)",
    },
    "láak'": {
        "gloss": "pariente/compañero",
        "category": "N",
        "meaning": "Familiar, hermano, prójimo o compañero (`a láak'-o'ob`)",
    },
    "yuum": {
        "gloss": "padre/señor",
        "category": "N",
        "meaning": "Padre, señor respetado o dueño (`in yuum`)",
    },
    "síis": {
        "gloss": "frío/fresco",
        "category": "ADJ",
        "meaning": "Fresco, frío agradable (`síis iik'`: 'viento fresco')",
    },
    "jáal": {
        "gloss": "orilla",
        "category": "N.LOC",
        "meaning": "Orilla, borde, ribera (`tu jáal`: 'a la orilla de')",
    },
}

# Voces de sustrato maya integradas dentro del Español Yucateco cotidiano
YUCATEC_SPANISH_SUBSTRATE_LOANS: dict[str, str] = {
    "mare": "Interjección exclamativa regional yucateca de asombro o énfasis.",
    "puchis": "Interjección coloquial de sorpresa o contrariedad leve.",
    "boxito": "Vocativo afectivo ('negrito/querido'), de maya `box` ('negro') + dim. `-ito`.",
    "xix": "Resto, migaja o asiento de comida/bebida (`/ʃiʃ/`, maya `xix`).",
    "xux": "Avispa (`/ʃuʃ/`, maya `xux`); en modismo 'ponerse xux' = ponerse alerta/vivo.",
    "tuch": "Ombligo o molleja (maya `tuuch`).",
    "chichí": "Abuela (maya infantil/afectivo `chich`).",
    "huiro": "Persona del interior o trato coloquial local (asociado a güiro/jícara).",
    "fo": "Interjección de desagrado olfativo realizada con cierre glotal final `[foʔ]`.",
    "pib": "Tamal grande horneado bajo tierra o el horno subterráneo (maya `píib`).",
}

# Palabras funcionales inequívocamente españolas para distinguir `es_yuc` vs `yua_es` vs `yua`
_SPANISH_FUNCTION_WORDS: set[str] = {
    "el",
    "la",
    "los",
    "las",
    "un",
    "una",
    "unos",
    "unas",
    "de",
    "del",
    "al",
    "en",
    "con",
    "por",
    "para",
    "pero",
    "como",
    "cuando",
    "donde",
    "dónde",
    "quien",
    "quién",
    "que",
    "qué",
    "no",
    "sí",
    "ya",
    "muy",
    "más",
    "este",
    "esta",
    "ese",
    "esa",
    "mi",
    "tu",
    "su",
    "está",
    "estuvo",
    "están",
    "vas",
    "vamos",
    "iba",
    "ir",
    "hacer",
    "haciendo",
    "dejaste",
    "dije",
    "compró",
    "hacen",
    "dejó",
    "pasaron",
    "busqué",
    "préstame",
    "ahorita",
    "encima",
    "debajo",
    "abajo",
}


# ==============================================================================
# 3. Base de Conocimiento de Construcciones y Variantes Pragmáticas Equivalentes
# ==============================================================================
CANONICAL_CONSTRUCTIONS: dict[str, dict[str, Any]] = {
    "ma'alob k'iin": {
        "semantic_translation": "Buen día / Buenos días.",
        "literal_translation": "Bueno [no-malo] día/sol.",
        "syntactic_structure": "Sintagma Nominal Salutatorio [ADJ + N]",
        "pragmatic_notes": (
            "Saludo matutino y diurno estándar pan-peninsular en el maya contemporáneo. "
            "Funciona en cualquier registro social desde el amanecer hasta el mediodía. "
            "Comparte equivalencia pragmática con 'Yutsil ja'atskab' (más tradicional y "
            "acotado a la mañana temprana) y con 'Ma'alob u sáastal k'iin' (deseo verbal sobre el amanecer)."
        ),
        "equivalent_variants": [
            "Yutsil ja'atskab (Buena mañana temprana — registro tradicional)",
            "Ma'alob u sáastal k'iin (Que amanezca bien el día — construcción verbal incoativa)",
        ],
    },
    "yutsil ja'atskab": {
        "semantic_translation": "Buena mañana / Muy buenos días (temprano).",
        "literal_translation": "Su bondad/hermosura de la mañana temprana.",
        "syntactic_structure": "Construcción Relacional Poseída [y-ADJ-il + N.TEMP]",
        "pragmatic_notes": (
            "Saludo tradicional y deferente propio de las primeras horas de la mañana (`ja'atskab`). "
            "A diferencia de 'Ma'alob k'iin', emplea la raíz positiva `uts` ('bondad/bien') con "
            "morfología relacional `-il` en lugar del compuesto negativo lexicalizado `ma'alob`."
        ),
        "equivalent_variants": [
            "Ma'alob k'iin (Buenos días — saludo general diurno)",
            "Ma'alob u sáastal k'iin (Buen clarear del día — desiderativo-descriptivo)",
        ],
    },
    "yutsil ja'atskab k'iin": {
        "semantic_translation": "Buena mañana de este día / Excelentes días.",
        "literal_translation": "Bondad de la mañana temprana del día.",
        "syntactic_structure": "Construcción Relacional Extendida [y-ADJ-il + N.TEMP + N]",
        "pragmatic_notes": (
            "Variante solemne y cortés de 'Yutsil ja'atskab' que explicita el núcleo temporal `k'iin` ('día/sol')."
        ),
        "equivalent_variants": [
            "Yutsil ja'atskab",
            "Ma'alob k'iin",
            "Ma'alob u sáastal k'iin",
        ],
    },
    "ma'alob u sáastal k'iin": {
        "semantic_translation": "Que amanezca bien el día / Buen amanecer.",
        "literal_translation": "Bueno es su clarear/amanecer del día.",
        "syntactic_structure": "Predicación Estativa sobre Cláusula Incompletiva Nominalizada [ADJ + 3SG.ERG + V.INCH + N]",
        "pragmatic_notes": (
            "Expresión genuinamente verbal-aspectual maya al despuntar el alba: utiliza el pronombre "
            "ergativo de Juego A `u` poseyendo al verbo incoativo `sáas-tal` ('volverse claro / amanecer'). "
            "Transmite un matiz augural y contemplativo frente al saludo breve 'Ma'alob k'iin'."
        ),
        "equivalent_variants": [
            "Ma'alob k'iin (Buenos días — fórmula breve estándar)",
            "Yutsil ja'atskab (Buena mañana — fórmula relacional tradicional)",
        ],
    },
    "ma'alob u sáastal": {
        "semantic_translation": "Buen amanecer / Que claree bien.",
        "literal_translation": "Bueno su clarear.",
        "syntactic_structure": "Predicación Estativa + Verbo Incoativo [ADJ + 3SG.ERG + V.INCH]",
        "pragmatic_notes": "Forma elíptica natural de 'Ma'alob u sáastal k'iin' empleada al alba.",
        "equivalent_variants": ["Ma'alob u sáastal k'iin", "Yutsil ja'atskab", "Ma'alob k'iin"],
    },
    "ma'alob chi'inil k'iin": {
        "semantic_translation": "Buenas tardes.",
        "literal_translation": "Buena inclinación/caída del sol.",
        "syntactic_structure": "Sintagma Nominal Relacional [ADJ + N-REL + N]",
        "pragmatic_notes": (
            "Saludo vespertino desde pasado el mediodía hasta la puesta del sol; describe astronómicamente "
            "el declive solar (`chi'in-il k'iin`)."
        ),
        "equivalent_variants": ["Yutsil chi'inil k'iin"],
    },
    "ma'alob áak'ab": {
        "semantic_translation": "Buenas noches.",
        "literal_translation": "Buena noche.",
        "syntactic_structure": "Sintagma Nominal Salutatorio [ADJ + N]",
        "pragmatic_notes": "Saludo o despedida nocturna estándar bajo la Norma de 1984.",
        "equivalent_variants": ["Yutsil áak'ab"],
    },
    "bix a beel": {
        "semantic_translation": "¿Cómo estás? / ¿Qué tal te va?",
        "literal_translation": "¿Cómo [está] tu camino?",
        "syntactic_structure": "Cláusula Interrogativa Estativa [INT + 2SG.ERG + N.POS]",
        "pragmatic_notes": (
            "Fórmula clásica de saludo dialógico maya; concibe el bienestar vital como el estado "
            "del camino personal (`a beel`, forma poseída de `bej`). Se responde típicamente 'Ma'alob' o 'Toj in wóol'."
        ),
        "equivalent_variants": [
            "Bix yanikech (¿Cómo te encuentras? — estado situacional)",
            "Ba'ax ka wa'alik (¿Qué dices / qué cuentas? — coloquial cercano)",
        ],
    },
    "toj in wóol": {
        "semantic_translation": "Estoy bien (de salud y ánimo).",
        "literal_translation": "Recto/sano [está] mi ánimo-espíritu.",
        "syntactic_structure": "Predicación Estativa Anímica [ADJ.EST + 1SG.ERG + w-N.INAL]",
        "pragmatic_notes": (
            "Respuesta canónica de bienestar integral en maya: `w-` es la semiconsonante epentética "
            "exigida por `in` (1SG Juego A) ante la raíz vocal `óol` ('centro anímico')."
        ),
        "equivalent_variants": [
            "Ma'alob, Dios bo'otik (Bien, gracias)",
            "Ki'imak in wóol (Estoy contento / alegre)",
        ],
    },
    "ki'imak in wóol": {
        "semantic_translation": "Estoy muy contento / Me da alegría.",
        "literal_translation": "Dulce/regozijado [está] mi ánimo.",
        "syntactic_structure": "Predicación Estativa Afectiva [ADJ + 1SG.ERG + w-N.INAL]",
        "pragmatic_notes": "Expresa felicidad activa o gusto al saludar a alguien.",
        "equivalent_variants": ["Toj in wóol (Estoy sano/bien)"],
    },
    "dios bo'otik": {
        "semantic_translation": "Gracias / Muchas gracias.",
        "literal_translation": "Dios lo paga / Dios lo recompense.",
        "syntactic_structure": "Cláusula Transitiva Habitual [N.SUJ + V.TR.INC]",
        "pragmatic_notes": (
            "Fórmula cotidiana mayoritaria de agradecimiento oral en las comunidades mayahablantes; "
            "integra el préstamo colonial `Dios` con el verbo transitivo `bo'ot-ik`."
        ),
        "equivalent_variants": ["Nib óolal (Gracias / agradecimiento — registro purista/institucional)"],
    },
    "nib óolal": {
        "semantic_translation": "Gracias / Agradecimiento.",
        "literal_translation": "Reconocimiento del ánimo/voluntad.",
        "syntactic_structure": "Compuesto Nominal-Verbal Clásico [V/N + N.ABSTR]",
        "pragmatic_notes": (
            "Expresión patrimonial recuperada en ámbitos educativos, literarios y de radio bilingüe "
            "como alternativa sin préstamos hispanos frente a 'Dios bo'otik'."
        ),
        "equivalent_variants": ["Dios bo'otik (Gracias — uso coloquial comunitario)"],
    },
    "tu'ux ka bin": {
        "semantic_translation": "¿A dónde vas?",
        "literal_translation": "¿Dónde INC-2SG.ERG ir?",
        "syntactic_structure": "Interrogativa de Movimiento con Foco Locativo [INT.LOC + AM.ERG + V.INTR]",
        "pragmatic_notes": (
            "Además de pregunta literal de destino, funciona pragmáticamente como saludo de paso "
            "al cruzarse con un vecino en el camino comunitario."
        ),
        "equivalent_variants": ["Tu'ux ka máan (¿Por dónde andas/pasas?)"],
    },
    "ba'ax ka wa'alik": {
        "semantic_translation": "¿Qué dices? / ¿Qué cuentas?",
        "literal_translation": "¿Qué INC-2SG.ERG EPEN-decir-INC.TR?",
        "syntactic_structure": "Interrogativa Transitiva de Foco de Objeto [PRON.INT + AM.ERG + w-V-ik]",
        "pragmatic_notes": (
            "Saludo conversacional amistoso equivalente al español mexicano '¿Qué cuentas? / ¿Qué onda?'. "
            "Muestra claramente el alomorfo prevocálico `w-` antes de la raíz `a'al` ('decir')."
        ),
        "equivalent_variants": ["Bix a beel (¿Cómo estás?)", "Ba'ax ka beetik (¿Qué haces?)"],
    },
    "ba'ax ka beetik": {
        "semantic_translation": "¿Qué estás haciendo? / ¿Qué haces?",
        "literal_translation": "¿Qué INC-2SG.ERG hacer-INC.TR?",
        "syntactic_structure": "Interrogativa Transitiva [PRON.INT + AM.ERG + V-ik]",
        "pragmatic_notes": "Pregunta cotidiana de actividad en curso o habitual.",
        "equivalent_variants": ["Ba'ax táan a beetik (¿Qué estás haciendo justo ahora? — progresivo explícito)"],
    },
    "ts'o'ok in janal": {
        "semantic_translation": "Ya comí / Ya terminé de comer.",
        "literal_translation": "TERM 1SG.ERG comer.",
        "syntactic_structure": "Cláusula Incompletiva con Aspecto Terminativo [AM.TERM + 1SG.ERG + V.INTR]",
        "pragmatic_notes": (
            "Ejemplifica la ergatividad escindida (*split ergativity*) del maya yucateco: con el marcador "
            "aspectual `ts'o'ok`, incluso un verbo intransitivo como `janal` toma sujeto de Juego A (`in`)."
        ),
        "equivalent_variants": ["Jaanen (Comí — completivo simple con sufijo Juego B `-en`)"],
    },
    "bix a beel toj in wóol dios bo'otik": {
        "semantic_translation": "¿Cómo estás? Estoy bien, muchas gracias.",
        "literal_translation": "¿Cómo tu camino? Sano mi ánimo, Dios lo paga.",
        "syntactic_structure": "Par Adyacente Salutatorio [Interrogativa Estativa + Respuesta Afectiva + Agradecimiento]",
        "pragmatic_notes": (
            "Intercambio dialógico completo de cortesía comunitaria en Maya Yucateco 1984."
        ),
        "equivalent_variants": ["¿Bix a beel? Ma'alob, nib óolal."],
    },
    "jach k'a'ana'an in kanik maaya t'aan sáansamal": {
        "semantic_translation": "Es muy importante que yo aprenda lengua maya todos los días.",
        "literal_translation": "Muy necesario 1SG.ERG aprender-INC.TR maya palabra cada.día.",
        "syntactic_structure": "Predicación Estativa Modal con Cláusula Subordinada [ADV + ADJ.EST + 1SG.ERG + V-ik + SN + ADV]",
        "pragmatic_notes": (
            "Concentra doble vocal rearticulada en `k'a'ana'an` (`/kʼa̰ʔana̰ʔan/`) y cierre glotal intervocálico "
            "junto con la eyectiva alveolar `t'` en `t'aan`."
        ),
        "equivalent_variants": ["K'a'ana'an k kanik maaya t'aan (Es necesario que aprendamos maya)"],
    },
    "ts'o'ok in janal bu'ul yéetel waaj ichil in wotoch": {
        "semantic_translation": "Ya terminé de comer frijol con tortilla dentro de mi hogar.",
        "literal_translation": "TERM 1SG.ERG comer frijol con tortilla dentro-REL 1SG.ERG EPEN-hogar.",
        "syntactic_structure": "Cláusula Terminativa Extendida [AM.TERM + 1SG.ERG + V + SN.COORD + SP.LOC]",
        "pragmatic_notes": (
            "Combina vocales rearticuladas (`ts'o'ok`, `bu'ul`), implosiva bilabial `/ɓ/`, tono alto (`yéetel`) "
            "y posesión inalienable con prefijo epentético `w-` en `in wotoch`."
        ),
        "equivalent_variants": [],
    },
    "ba'ax ka wa'alik ko'ox xíimbal te'elo'": {
        "semantic_translation": "¿Qué cuentas? ¡Vamos a caminar por allá!",
        "literal_translation": "¿Qué INC-2SG.ERG EPEN-decir-INC.TR? EXHORT.1PL caminar allá-DIST.",
        "syntactic_structure": "Secuencia Conversacional [Interrogativa Transitiva + Cláusula Exhortativa 1PL]",
        "pragmatic_notes": (
            "Ejemplifica cuatro vocales rearticuladas/glotalizadas (`ba'ax`, `wa'alik`, `ko'ox`, `te'elo'`) "
            "y el contraste con la vocal larga de tono alto en `xíimbal`."
        ),
        "equivalent_variants": ["Ko'one'ex xíimbal (¡Vámonos todos a pasear!)"],
    },
    "le ch'íich'o' táan u k'aay yóok'ol le k'áak'o'": {
        "semantic_translation": "Ese pájaro está cantando sobre el fuego.",
        "literal_translation": "DET pájaro-DIST PROG 3SG.ERG cantar 3SG-sobre-REL DET fuego-DIST.",
        "syntactic_structure": "Cláusula Progresiva Topicalizada con Doble Marco Demostrativo `le ... -o'`",
        "pragmatic_notes": (
            "Oración de alta densidad de consonantes eyectivas (`ch'`, `k'`) combinadas con núcleos "
            "largos de tono alto (`ch'íich'`, `táan`, `yóok'ol`, `k'áak'`) y cierres glotales deícticos `-o'`."
        ),
        "equivalent_variants": [],
    },
    "taak in ts'íibtik u k'aaba' in kaajal ich maaya t'aan": {
        "semantic_translation": "Quiero escribir el nombre de mi pueblo en lengua maya.",
        "literal_translation": "DESID 1SG.ERG escribir-INC.TR 3SG.ERG nombre 1SG.ERG pueblo-REL en maya lengua.",
        "syntactic_structure": "Construcción Modal Desiderativa Transitiva [AM.DESID + 1SG.ERG + V-ik + SN.POS + SP]",
        "pragmatic_notes": (
            "Reúne las eyectivas `ts'`, `k'` y `t'` junto con cierre glotal léxico final en `k'aaba'` (`/kʼàːɓaʔ/`)."
        ),
        "equivalent_variants": ["In k'áat in ts'íibtej u k'aaba' in kaajal"],
    },
    "p'o' a k'ab táanil ti' a janal yéetel a láak'o'ob": {
        "semantic_translation": "Lávate las manos antes de comer con tus familiares.",
        "literal_translation": "Lavar.IMP 2SG.ERG mano antes-REL PREP 2SG.ERG comer con 2SG.ERG pariente-PL.",
        "syntactic_structure": "Cláusula Imperativa Transitiva + Subordinada Temporal [V.IMP + SN + PREP.TEMP + SN]",
        "pragmatic_notes": (
            "Ilustra la bilabial eyectiva inicial `p'` seguida de cierre glotal en `p'o'` (`/pʼoʔ/`), "
            "la implosiva final `/ɓ/` en `k'ab` y el circunfijo `a ... -o'ob`."
        ),
        "equivalent_variants": [],
    },
    "sáansamal ku sáastal ma'alob k'iin tu kool in yuum": {
        "semantic_translation": "Todos los días amanece un buen día en la milpa de mi padre.",
        "literal_translation": "Cada.día INC-3SG clarear-INCH bueno día PREP-3SG milpa 1SG.ERG padre.",
        "syntactic_structure": "Cláusula Habitual Incoativa con Adjunto Locativo Poseído",
        "pragmatic_notes": (
            "Diseñada para contrastar acústicamente vocales largas de tono alto (`sáansamal`, `sáastal` -> `/áː/`) "
            "frente a vocales largas de tono bajo (`k'iin`, `kool`, `yuum` -> `/ìː, òː, ùː/`)."
        ),
        "equivalent_variants": [],
    },
    "ma'alob áak'ab táan u taal síis iik' te' kaajo'": {
        "semantic_translation": "Buenas noches, está llegando aire fresco aquí en el pueblo.",
        "literal_translation": "Buena noche, PROG 3SG.ERG venir fresco viento LOC.DEM pueblo-DIST.",
        "syntactic_structure": "Saludo Nocturno + Cláusula Progresiva de Movimiento",
        "pragmatic_notes": (
            "Contrasta tres sílabas largas de tono alto (`áak'ab`, `táan`, `síis`) con tres sílabas largas "
            "de tono bajo (`taal`, `iik'`, `kaaj`) para calibración de excursión tonal en Praat."
        ),
        "equivalent_variants": [],
    },
    "in peek' yéetel in miis táan u wenelo'ob tu jáal le bejo'": {
        "semantic_translation": "Mi perro y mi gato están durmiendo a la orilla del camino.",
        "literal_translation": "1SG.ERG perro con 1SG.ERG gato PROG 3SG.ERG dormir-3PL PREP-3SG orilla DET camino-DIST.",
        "syntactic_structure": "Sujeto Coordinado Topicalizado + Cláusula Progresiva Plural",
        "pragmatic_notes": (
            "Oposición directa entre vocales largas bajas (`peek'`, `miis`) y vocales largas altas (`yéetel`, `táan`, `jáal`)."
        ),
        "equivalent_variants": [],
    },
    "tu'ux ka bin bejla'e' yan in bin meyaj ich kool": {
        "semantic_translation": "¿A dónde vas hoy? Tengo que ir a trabajar a la milpa.",
        "literal_translation": "¿Dónde INC-2SG ir hoy-TOP? OBLIG 1SG.ERG ir trabajar en milpa.",
        "syntactic_structure": "Par Dialógico [Interrogativa Locativa + Cláusula Obligativa con `yan`]",
        "pragmatic_notes": (
            "Combina vocal rearticulada (`tu'ux`), clítico tópico final (`bejla'e'`), modo obligativo (`yan`) "
            "y contraste de duración vocálica entre `bin` (corta) y `kool` (larga baja)."
        ),
        "equivalent_variants": [],
    },
}


# ==============================================================================
# Clase Principal: Analizador Morfosintáctico y Semántico
# ==============================================================================
@dataclass
class MorphemeBreakdown:
    """Glosa interlineal estilo Leipzig de un token léxico."""

    surface_word: str
    normalized_word: str
    morpheme_segmentation: str
    leipzig_gloss: str
    lexical_category: str
    meaning_es: str
    is_maya_word: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface_word": self.surface_word,
            "normalized_word": self.normalized_word,
            "morpheme_segmentation": self.morpheme_segmentation,
            "leipzig_gloss": self.leipzig_gloss,
            "lexical_category": self.lexical_category,
            "meaning_es": self.meaning_es,
            "is_maya_word": self.is_maya_word,
        }


class MayaGrammaticalAnalyzer:
    """
    Motor de análisis morfosintáctico, detección de lengua (`yua` / `es_yuc` / `yua_es`),
    glosa interlineal Leipzig y traducción semántico-pragmática para Maya Yucateco 1984.
    """

    _WORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+")

    def __init__(self, g2p: Optional[MayaG2P1984] = None) -> None:
        self.g2p = g2p or MayaG2P1984()

    def detect_language_code(self, text: str) -> str:
        """
        Clasifica automáticamente el texto de entrada en:
        - `"yua"`   : Maya Yucateco (Norma 1984, incluyendo préstamos coloniales aislados como `Dios`).
        - `"es_yuc"`: Español Yucateco / Mexicano (incluyendo regionalismos de sustrato como `boxito`, `xix`).
        - `"yua_es"`: Alternancia de código bilingüe Maya-Español (*code-switching*).
        """
        raw_tokens = [
            t for t in self._WORD_RE.findall(text or "") if any(c.isalnum() for c in t)
        ]
        if not raw_tokens:
            return "es_yuc"

        norm_text = self.g2p.normalize_orthography_1984(text)
        norm_tokens = [
            t.lower()
            for t in self._WORD_RE.findall(norm_text)
            if any(c.isalnum() for c in t)
        ]

        maya_score = 0
        spanish_score = 0

        for idx, tok in enumerate(norm_tokens):
            stripped = tok.strip("'")

            # "Dios" junto a "bo'otik" es parte de locución maya consolidada
            if stripped == "dios" and "bo'otik" in norm_tokens:
                maya_score += 1
                continue

            if stripped in YUCATEC_SPANISH_SUBSTRATE_LOANS:
                # Modismos del español yucateco (`boxito`, `puchis`, `mare`, `huiro`)
                spanish_score += 2
                continue

            if stripped in _SPANISH_FUNCTION_WORDS:
                spanish_score += 2
                continue

            is_known_maya = (
                tok in MAYA_LEXICON
                or stripped in MAYA_LEXICON
                or tok in ASPECT_MOOD_MARKERS
                or tok in ERGATIVE_SET_A
            )
            has_maya_phonotactics = (
                "'" in tok
                or "ts" in tok
                or "k'" in tok
                or "ch'" in tok
                or "p'" in tok
                or "t'" in tok
                or bool(re.search(r"([aeiou])\1", tok))
                or bool(re.search(r"[áéíóú][aeiou]", tok))
                or tok.startswith(("w", "x"))
                or tok.endswith(("ik", "aj", "o'ob", "e'ex", "tal"))
            )

            if is_known_maya or has_maya_phonotactics:
                # Evitar falsos positivos con 'u' conjunción española o 'a' preposición española
                if tok in ("a", "u") and idx > 0 and norm_tokens[idx - 1] in _SPANISH_FUNCTION_WORDS:
                    spanish_score += 1
                else:
                    maya_score += 2
            else:
                spanish_score += 1

        if maya_score >= 2 and spanish_score == 0:
            return "yua"
        if maya_score > 0 and spanish_score > 0:
            total = maya_score + spanish_score
            if (maya_score / total) >= 0.78:
                return "yua"
            if (spanish_score / total) >= 0.80:
                return "es_yuc"
            return "yua_es"
        return "yua" if maya_score > spanish_score else "es_yuc"

    def _analyze_single_word(
        self,
        surface_word: str,
        norm_word: str,
        prev_norm_lower: Optional[str],
    ) -> MorphemeBreakdown:
        """Desglosa morfológicamente una palabra aplicando léxico y reglas de afijación maya."""
        w = norm_word.lower()

        # 1. Marcadores de Aspecto-Modo y contracciones AM+ERG
        if w in ASPECT_MOOD_MARKERS:
            entry = ASPECT_MOOD_MARKERS[w]
            seg = w
            if w in ("kin", "ka", "ku"):
                seg = f"k-{w[1:]}"
            elif w in ("tin", "ta", "tu"):
                seg = f"t-{w[1:]}"
            return MorphemeBreakdown(
                surface_word=surface_word,
                normalized_word=norm_word,
                morpheme_segmentation=seg,
                leipzig_gloss=entry["gloss"],
                lexical_category=entry["category"],
                meaning_es=entry["meaning"],
                is_maya_word=True,
            )

        # 2. Pronombres Ergativos de Juego A
        if w in ERGATIVE_SET_A:
            entry = ERGATIVE_SET_A[w]
            return MorphemeBreakdown(
                surface_word=surface_word,
                normalized_word=norm_word,
                morpheme_segmentation=w,
                leipzig_gloss=entry["gloss"],
                lexical_category=entry["category"],
                meaning_es=entry["meaning"],
                is_maya_word=True,
            )

        # 3. Entrada directa en el léxico maya
        if w in MAYA_LEXICON:
            entry = MAYA_LEXICON[w]
            seg = entry.get("decomposition", w)
            is_loan_es = entry["category"].endswith("PRÉSTAMO")
            return MorphemeBreakdown(
                surface_word=surface_word,
                normalized_word=norm_word,
                morpheme_segmentation=seg,
                leipzig_gloss=entry["gloss"],
                lexical_category=entry["category"],
                meaning_es=entry["meaning"],
                is_maya_word=not is_loan_es,
            )

        # 4. Préstamos de sustrato maya en español yucateco
        if w in YUCATEC_SPANISH_SUBSTRATE_LOANS:
            return MorphemeBreakdown(
                surface_word=surface_word,
                normalized_word=norm_word,
                morpheme_segmentation=w,
                leipzig_gloss="SUSTRATO.MAYA",
                lexical_category="Regionalismo Yucateco",
                meaning_es=YUCATEC_SPANISH_SUBSTRATE_LOANS[w],
                is_maya_word=True,
            )

        # 5. Segmentador Morfológico Productivo (Prefijos epentéticos w-/y- + Raíz + Sufijos)
        prefix_morphs: list[str] = []
        prefix_glosses: list[str] = []
        stem = w

        # Alomorfos prevocálicos de Juego A (`w-` tras `in, a, kin, ka, tin, ta`; `y-` tras `u, ku, tu`)
        if (
            len(stem) >= 3
            and stem.startswith("w")
            and stem[1] in "aeiouáéíóú"
            and (prev_norm_lower in ("in", "a", "kin", "ka", "tin", "ta") or stem[1:] in MAYA_LEXICON)
        ):
            prefix_morphs.append("w")
            prefix_glosses.append("EPEN.1/2SG")
            stem = stem[1:]
        elif (
            len(stem) >= 3
            and stem.startswith("y")
            and stem[1] in "aeiouáéíóú"
            and (prev_norm_lower in ("u", "ku", "tu") or stem[1:] in MAYA_LEXICON)
        ):
            prefix_morphs.append("y")
            prefix_glosses.append("EPEN.3SG")
            stem = stem[1:]

        # Desprendimiento de clíticos finales de marco/deixis (`-a'`, `-o'`, `-e'`, `-i'`)
        suffix_morphs: list[str] = []
        suffix_glosses: list[str] = []
        suffix_notes: list[str] = []

        for clit, clit_gloss, clit_desc in DEICTIC_AND_FRAME_CLITICS:
            if len(stem) > (len(clit) + 1) and stem.endswith(clit) and stem not in MAYA_LEXICON:
                # Evitar cortar vocales cortas glotalizadas léxicas como `ja'`, `k'aaba'`
                candidate = stem[: -len(clit)]
                if candidate in MAYA_LEXICON or any(
                    candidate.endswith(s[0]) for s in ABSOLUTIVE_SET_B_SUFFIXES + DERIVATIONAL_AND_STATUS_SUFFIXES
                ):
                    stem = candidate
                    suffix_morphs.insert(0, f"-{clit}")
                    suffix_glosses.insert(0, clit_gloss)
                    suffix_notes.insert(0, clit_desc)
                    break

        # Desprendimiento de sufijos Juego B / pluralizadores (`-o'ob`, `-e'ex`, `-o'on`, `-ech`, `-en`)
        for suf, suf_gloss, suf_desc in ABSOLUTIVE_SET_B_SUFFIXES:
            if len(stem) > (len(suf) + 1) and stem.endswith(suf) and stem not in MAYA_LEXICON:
                candidate = stem[: -len(suf)]
                stem = candidate
                suffix_morphs.insert(0, f"-{suf}")
                suffix_glosses.insert(0, suf_gloss)
                suffix_notes.insert(0, suf_desc)
                break

        # Desprendimiento de sufijos derivativos / estatus verbal (`-tal`, `-ik`, `-aj`, `-ej`, `-il`)
        for d_suf, d_gloss, d_desc in DERIVATIONAL_AND_STATUS_SUFFIXES:
            if len(stem) >= (len(d_suf) + 2) and stem.endswith(d_suf) and stem not in MAYA_LEXICON:
                candidate = stem[: -len(d_suf)]
                stem = candidate
                suffix_morphs.insert(0, f"-{d_suf}")
                suffix_glosses.insert(0, d_gloss)
                suffix_notes.insert(0, d_desc)
                break

        if stem in MAYA_LEXICON or prefix_morphs or suffix_morphs:
            root_entry = MAYA_LEXICON.get(stem)
            root_gloss = root_entry["gloss"] if root_entry else stem
            root_cat = root_entry["category"] if root_entry else "RAÍZ.MAYA"
            root_meaning = root_entry["meaning"] if root_entry else f"Raíz maya '{stem}'"

            seg_parts = [f"{p}-" for p in prefix_morphs] + [stem] + suffix_morphs
            gloss_parts = prefix_glosses + [root_gloss] + suffix_glosses

            extra = f" [{' + '.join(suffix_notes)}]" if suffix_notes else ""
            return MorphemeBreakdown(
                surface_word=surface_word,
                normalized_word=norm_word,
                morpheme_segmentation="".join(seg_parts),
                leipzig_gloss="-".join(gloss_parts),
                lexical_category=root_cat if not suffix_morphs else f"{root_cat} (Flexionado)",
                meaning_es=f"{root_meaning}{extra}",
                is_maya_word=True,
            )

        # 6. Heurística para palabras desconocidas: distinguir raíz maya potencial vs palabra española
        looks_maya = (
            "'" in w
            or "ts" in w
            or w.startswith(("x", "w", "k"))
            or bool(re.search(r"([aeiou])\1", w))
            or bool(re.search(r"[áéíóú][aeiou]", w))
        )
        if looks_maya:
            return MorphemeBreakdown(
                surface_word=surface_word,
                normalized_word=norm_word,
                morpheme_segmentation=w,
                leipzig_gloss=f"RAÍZ({w})",
                lexical_category="Vocablo Maya (Abierto)",
                meaning_es="Forma léxica maya analizada fonológicamente bajo Norma 1984",
                is_maya_word=True,
            )

        return MorphemeBreakdown(
            surface_word=surface_word,
            normalized_word=norm_word,
            morpheme_segmentation=w,
            leipzig_gloss="ESP",
            lexical_category="Vocablo Español",
            meaning_es=f"Palabra del español ('{surface_word}')",
            is_maya_word=False,
        )

    @staticmethod
    def _detect_discontinuous_frames(norm_words_lower: list[str]) -> list[str]:
        """Detecta construcciones discontinuas típicas de la sintaxis maya yucateca."""
        frames: list[str] = []
        if not norm_words_lower:
            return frames

        last_word = norm_words_lower[-1]
        if "je'el" in norm_words_lower and last_word.endswith("e'"):
            frames.append(
                "Marco Aseverativo Discontinuo `je'el ... -e'`: compromete al hablante con la certeza del evento."
            )
        if "le" in norm_words_lower:
            if any(w.endswith("a'") for w in norm_words_lower):
                frames.append(
                    "Marco Demostrativo Proximal `le ... -a'` ('este / aquí cerca del hablante')."
                )
            if any(w.endswith("o'") for w in norm_words_lower):
                frames.append(
                    "Marco Demostrativo Distal `le ... -o'` ('ese / aquel conocido por los interlocutores')."
                )
            if any(w.endswith("e'") and w != "je'ele'" for w in norm_words_lower):
                frames.append(
                    "Marco de Tópico Definido `le ... -e'` ('el/la mencionado')."
                )
        if "a" in norm_words_lower and any(w.endswith("e'ex") for w in norm_words_lower):
            frames.append(
                "Circunfijo Ergativo de 2ª Persona Plural `a ... -e'ex` ('ustedes / de ustedes')."
            )
        if "u" in norm_words_lower and any(w.endswith("o'ob") for w in norm_words_lower):
            frames.append(
                "Circunfijo Ergativo de 3ª Persona Plural `u ... -o'ob` ('ellos / de ellos')."
            )
        if "ma'" in norm_words_lower and any(w.endswith("i'") for w in norm_words_lower):
            frames.append(
                "Marco Negativo Enfático `ma' ... -i'` (negación de foco o predicado estativo)."
            )
        return frames

    def _compose_compositional_translation(
        self,
        lang_code: str,
        morphemes: list[MorphemeBreakdown],
        norm_clean_key: str,
    ) -> tuple[str, str, str, str, list[str]]:
        """
        Obtiene la traducción semántica, glosa literal, estructura sintáctica,
        notas pragmáticas y variantes equivalentes de la oración.
        """
        # 1) Búsqueda exacta en construcciones canónicas
        if norm_clean_key in CANONICAL_CONSTRUCTIONS:
            c = CANONICAL_CONSTRUCTIONS[norm_clean_key]
            return (
                c["semantic_translation"],
                c["literal_translation"],
                c["syntactic_structure"],
                c["pragmatic_notes"],
                list(c.get("equivalent_variants", [])),
            )

        # 2) Búsqueda de sub-construcciones contenidas en enunciados compuestos
        matched_sub_translations: list[str] = []
        matched_notes: list[str] = []
        matched_variants: list[str] = []
        # Ordenar claves de mayor a menor longitud para priorizar frases más específicas
        for key in sorted(CANONICAL_CONSTRUCTIONS.keys(), key=len, reverse=True):
            if re.search(rf"(?:^|\b){re.escape(key)}(?:$|\b)", norm_clean_key):
                c = CANONICAL_CONSTRUCTIONS[key]
                matched_sub_translations.append(c["semantic_translation"].rstrip("."))
                matched_notes.append(f"[{key}] {c['pragmatic_notes']}")
                for v in c.get("equivalent_variants", []):
                    if v not in matched_variants:
                        matched_variants.append(v)

        literal_parts = [m.leipzig_gloss for m in morphemes]
        literal_str = " ".join(literal_parts)

        if lang_code == "es_yuc":
            substrate_hits = [
                f"`{m.normalized_word}` ({m.meaning_es})"
                for m in morphemes
                if m.leipzig_gloss == "SUSTRATO.MAYA"
            ]
            prag_note = (
                "Enunciado en español regional con préstamos léxicos de sustrato maya: "
                + "; ".join(substrate_hits)
                if substrate_hits
                else "Enunciado en español regional yucateco/mexicano sin morfología verbal maya activa."
            )
            return (
                "Oración en español regional (ver desglose de regionalismos/sustrato).",
                literal_str,
                "Oración en Español Yucateco / Mexicano",
                prag_note,
                [],
            )

        if matched_sub_translations:
            sem_trans = " / ".join(matched_sub_translations) + " (construcción extendida)."
            synt_struct = "Oración Compuesta / Cláusula Extendida en Maya Yucateco"
            prag_str = " ".join(matched_notes)
            return sem_trans, literal_str, synt_struct, prag_str, matched_variants

        # 3) Síntesis composicional guiada por glosas morfológicas
        gloss_readout = [
            m.leipzig_gloss if m.is_maya_word else m.surface_word for m in morphemes
        ]
        sem_approx = "Glosa composicional: " + " · ".join(gloss_readout)
        synt_struct = (
            "Cláusula Bilingüe Maya-Español (Code-Switching)"
            if lang_code == "yua_es"
            else "Cláusula en Maya Yucateco (Norma Ortográfica de 1984)"
        )
        prag_default = (
            "Analizada productivamente mediante segmentación de marcadores Aspecto-Modo, "
            "pronombres ergativos (Juego A), sufijos absolutivos (Juego B) y raíces léxicas."
        )
        return sem_approx, literal_str, synt_struct, prag_default, []

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Ejecuta el análisis lingüístico integral (G2P 1984 + Morfosintaxis + Semántica):
        - Detecta `language_code` (`yua`, `es_yuc`, `yua_es`).
        - Genera desglose morfema por morfema (glosa interlineal Leipzig).
        - Traduce el sentido semántico y documenta equivalencias pragmáticas regionales.
        """
        raw_text = (text or "").strip()
        g2p_res = self.g2p.phonemize_sentence(raw_text)
        lang_code = self.detect_language_code(raw_text)

        lang_labels = {
            "yua": "Maya Yucateco (Norma de Agosto de 1984)",
            "es_yuc": "Español Yucateco / Mexicano",
            "yua_es": "Bilingüe Maya Yucateco – Español (Code-Switching)",
        }

        surface_words = [
            t for t in self._WORD_RE.findall(raw_text) if any(c.isalnum() for c in t)
        ]
        norm_words = [w.normalized_word for w in g2p_res.words]

        # Emparejar palabras superficiales y normalizadas con seguridad
        paired_words: list[tuple[str, str]] = []
        for idx, nw in enumerate(norm_words):
            sw = surface_words[idx] if idx < len(surface_words) else nw
            paired_words.append((sw, nw))

        morpheme_list: list[MorphemeBreakdown] = []
        prev_lower: Optional[str] = None
        for sw, nw in paired_words:
            mb = self._analyze_single_word(sw, nw, prev_lower)
            morpheme_list.append(mb)
            prev_lower = nw.lower()

        norm_clean_key = " ".join(nw.lower() for _, nw in paired_words).strip()
        (
            semantic_translation,
            literal_translation,
            syntactic_structure,
            pragmatic_notes,
            equivalent_variants,
        ) = self._compose_compositional_translation(
            lang_code=lang_code,
            morphemes=morpheme_list,
            norm_clean_key=norm_clean_key,
        )

        discontinuous_frames = self._detect_discontinuous_frames(
            [nw.lower() for _, nw in paired_words]
        )

        interlinear_tokens = "   ".join(m.morpheme_segmentation for m in morpheme_list)
        interlinear_gloss = "   ".join(m.leipzig_gloss for m in morpheme_list)

        return {
            "original_text": raw_text,
            "normalized_1984": g2p_res.normalized_1984,
            "language_code": lang_code,
            "language_label": lang_labels.get(lang_code, lang_code),
            "syntactic_structure": syntactic_structure,
            "semantic_translation_es": semantic_translation,
            "literal_gloss_es": literal_translation,
            "interlinear_segmentation": interlinear_tokens,
            "interlinear_leipzig_gloss": interlinear_gloss,
            "pragmatic_notes": pragmatic_notes,
            "equivalent_variants": equivalent_variants,
            "discontinuous_frames": discontinuous_frames,
            "morphemes": [m.to_dict() for m in morpheme_list],
            "g2p": g2p_res.to_dict(),
        }
