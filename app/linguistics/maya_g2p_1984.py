"""
Motor Fonológico Grafema-a-Fonema (G2P) para Lengua Maya Yucateca (`yua`)
bajo el Alfabeto Práctico de la Norma Ortográfica de Agosto de 1984.

Principios lingüísticos y de ingeniería TTS/DSP:
1. Separación estricta entre representación fonológica (este módulo) y medición
   acústica física en Praat (`app/analyzers/prosody.py`).
2. Normalización automática hacia la ortografía de Agosto de 1984:
   - Unificación de apóstrofos tipográficos (`’`, `‘`, `ʼ`, `ʾ`, `` ` ``, `´`) -> `'`.
   - Conversión de grafías pre-1984 / coloniales (`tz'`/`dz` -> `ts'`, `tz` -> `ts`,
     `c'` -> `k'`, `c` velar en raíces mayas -> `k`) respetando préstamos del español.
3. REGLA CRÍTICA DEL APÓSTROFO (`'`): PROHIBIDO convertir apóstrofos en espacios o pausas.
   Cada apóstrofo se clasifica contextualmente con precedencia estricta en 4 tipos:
   - Tipo 1 (`TYPE_1_EJECTIVE`): Consonante eyectiva/glotalizada indivisible
     (`ch'`, `ts'`, `k'`, `p'`, `t'`) -> `/tʃʼ, tsʼ, kʼ, pʼ, tʼ/`
     (`CH_EJ, TS_EJ, K_EJ, P_EJ, T_EJ`).
   - Tipo 2 (`TYPE_2_REARTICULATED`): Vocal rearticulada/laringalizada intervocálica
     homorgánica (`V₁'V₁`, `V́₁'V₁`, `V₁'V́₁`) -> núcleo único `/a̰ʔa, ḛʔe, ḭʔi, o̰ʔo, ṵʔu/`
     (`A_REART, E_REART, I_REART, O_REART, U_REART`).
   - Tipo 3 (`TYPE_3_CODA_GLOTTAL`): Cierre glotal posvocálico/final (`V'` ante
     consonante, frontera heterorgánica o fin de palabra) -> `/aʔ, eʔ, iʔ, oʔ, uʔ/`
     (`A_GLOT, E_GLOT, I_GLOT, O_GLOT, U_GLOT`).
   - Tipo 4 (`TYPE_4_BOUNDARY_GLOTTAL`): Ataque glotal prevocálico inicial (`'V`,
     ej. `in 'otoch`) o frontera morfológica tras consonante no eyectivable (`C'V`,
     ej. `y'éetel`) -> `/ʔ/` (`GLOT_ONS`).
4. Distinción exacta de los 5 núcleos vocálicos del maya yucateco:
   - Corta (`a, e, i, o, u`)                -> `/a, e, i, o, u/`
   - Larga tono bajo (`aa, ee, ii, oo, uu`) -> `/àː, èː, ìː, òː, ùː/`
   - Larga tono alto (`áa, ée, íi, óo, úu`) -> `/áː, éː, íː, óː, úː/`
   - Rearticulada (`a'a, e'e, i'i, o'o, u'u`) -> `/a̰ʔa, ḛʔe, ḭʔi, o̰ʔo, ṵʔu/`
   - Glotalizada corta (`a', e', i', o', u'`) -> `/aʔ, eʔ, iʔ, oʔ, uʔ/`
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

# ==============================================================================
# Tablas Fonológicas del Alfabeto Práctico de Agosto de 1984
# ==============================================================================
_BASE_VOWEL_MAP: dict[str, str] = {
    "a": "a",
    "á": "a",
    "e": "e",
    "é": "e",
    "i": "i",
    "í": "i",
    "o": "o",
    "ó": "o",
    "u": "u",
    "ú": "u",
}

_ACUTE_VOWELS: set[str] = {"á", "é", "í", "ó", "ú"}
_PLAIN_VOWELS: set[str] = {"a", "e", "i", "o", "u"}
_ALL_VOWELS: set[str] = _PLAIN_VOWELS | _ACUTE_VOWELS

# 1. Consonantes Eyectivas / Glotalizadas (Tipo 1)
_EJECTIVE_MAP: dict[str, tuple[str, str, str, str]] = {
    # grafema: (AFI, tts_token, codigo_simbolico, lectura_aprox_es)
    "ts'": ("tsʼ", "ts_ej", "TS_EJ", "tsʼ"),
    "ch'": ("tʃʼ", "ch_ej", "CH_EJ", "chʼ"),
    "k'": ("kʼ", "k_ej", "K_EJ", "kʼ"),
    "p'": ("pʼ", "p_ej", "P_EJ", "pʼ"),
    "t'": ("tʼ", "t_ej", "T_EJ", "tʼ"),
}

# 2. Vocales Rearticuladas / Laringalizadas (Tipo 2: V₁'V₁)
_REARTICULATED_VOWEL_MAP: dict[str, tuple[str, str, str, str]] = {
    "a": ("a̰ʔa", "a_reart", "A_REART", "a·ʔa"),
    "e": ("ḛʔe", "e_reart", "E_REART", "e·ʔe"),
    "i": ("ḭʔi", "i_reart", "I_REART", "i·ʔi"),
    "o": ("o̰ʔo", "o_reart", "O_REART", "o·ʔo"),
    "u": ("ṵʔu", "u_reart", "U_REART", "u·ʔu"),
}

# 3. Vocales Largas con Tono Alto (V́V)
_HIGH_LONG_VOWEL_MAP: dict[str, tuple[str, str, str, str]] = {
    "a": ("áː", "aa_high", "AA_HIGH", "áa"),
    "e": ("éː", "ee_high", "EE_HIGH", "ée"),
    "i": ("íː", "ii_high", "II_HIGH", "íi"),
    "o": ("óː", "oo_high", "OO_HIGH", "óo"),
    "u": ("úː", "uu_high", "UU_HIGH", "úu"),
}

# 4. Vocales Largas con Tono Bajo (VV)
_LOW_LONG_VOWEL_MAP: dict[str, tuple[str, str, str, str]] = {
    "a": ("àː", "aa_low", "AA_LOW", "aa"),
    "e": ("èː", "ee_low", "EE_LOW", "ee"),
    "i": ("ìː", "ii_low", "II_LOW", "ii"),
    "o": ("òː", "oo_low", "OO_LOW", "oo"),
    "u": ("ùː", "uu_low", "UU_LOW", "uu"),
}

# 5. Vocales Cortas Glotalizadas / Cierre Glotal Posvocálico (Tipo 3: V')
_GLOTTAL_VOWEL_MAP: dict[str, tuple[str, str, str, str]] = {
    "a": ("aʔ", "a_glot", "A_GLOT", "aʔ"),
    "e": ("eʔ", "e_glot", "E_GLOT", "eʔ"),
    "i": ("iʔ", "i_glot", "I_GLOT", "iʔ"),
    "o": ("oʔ", "o_glot", "O_GLOT", "oʔ"),
    "u": ("uʔ", "u_glot", "U_GLOT", "uʔ"),
}

# 6. Vocales Cortas Simples y Tónicas Cortas
_SHORT_VOWEL_MAP: dict[str, tuple[str, str, str, str]] = {
    "a": ("a", "a", "A", "a"),
    "e": ("e", "e", "E", "e"),
    "i": ("i", "i", "I", "i"),
    "o": ("o", "o", "O", "o"),
    "u": ("u", "u", "U", "u"),
    "á": ("á", "a_high", "A_HIGH", "á"),
    "é": ("é", "e_high", "E_HIGH", "é"),
    "í": ("í", "i_high", "I_HIGH", "í"),
    "ó": ("ó", "o_high", "O_HIGH", "ó"),
    "ú": ("ú", "u_high", "U_HIGH", "ú"),
}

# 7. Dígrafos Consonánticos Simples (Norma 1984 + Préstamos)
_DIGRAPH_CONSONANT_MAP: dict[str, tuple[str, str, str, str, str]] = {
    # grafema: (AFI, tts_token, codigo_simbolico, lectura_es, categoria)
    "ch": ("tʃ", "ch", "CH", "ch", "plain_consonant"),
    "ts": ("ts", "ts", "TS", "ts", "plain_consonant"),
    "rr": ("r", "rr", "RR_LOAN", "rr", "loan_consonant"),
    "ll": ("j", "y", "LL_LOAN", "y", "loan_consonant"),
    "qu": ("k", "k", "K_LOAN", "k", "loan_consonant"),
}

# 8. Consonantes Simples de 1 Grafema (Norma 1984 + Préstamos Hispanos)
_SINGLE_CONSONANT_MAP: dict[str, tuple[str, str, str, str, str]] = {
    "b": ("ɓ", "b_imp", "B_IMP", "b", "implosive_consonant"),
    "j": ("h", "h", "J_ASP", "j", "plain_consonant"),
    "k": ("k", "k", "K", "k", "plain_consonant"),
    "l": ("l", "l", "L", "l", "plain_consonant"),
    "m": ("m", "m", "M", "m", "plain_consonant"),
    "n": ("n", "n", "N", "n", "plain_consonant"),
    "p": ("p", "p", "P", "p", "plain_consonant"),
    "r": ("ɾ", "r", "R", "r", "plain_consonant"),
    "s": ("s", "s", "S", "s", "plain_consonant"),
    "t": ("t", "t", "T", "t", "plain_consonant"),
    "w": ("w", "w", "W", "gu", "plain_consonant"),
    "x": ("ʃ", "sh", "X_SH", "sh", "plain_consonant"),
    "y": ("j", "y", "Y", "y", "plain_consonant"),
    # Consonantes frecuentes en préstamos integrados (ej. "Dios bo'otik", "amigo")
    "d": ("d", "d", "D_LOAN", "d", "loan_consonant"),
    "f": ("f", "f", "F_LOAN", "f", "loan_consonant"),
    "g": ("ɡ", "g", "G_LOAN", "g", "loan_consonant"),
    "ñ": ("ɲ", "ny", "NY_LOAN", "ñ", "loan_consonant"),
    "v": ("b", "b", "V_LOAN", "b", "loan_consonant"),
    "z": ("s", "s", "Z_LOAN", "s", "loan_consonant"),
    "h": ("", "", "H_MUTE", "", "loan_consonant"),
}

# Conjunto de préstamos y voces españolas comunes cuya 'c'/'z'/'qu' no debe reescribirse a 'k'/'s'
_SPANISH_LOANWORDS_PRESERVE: set[str] = {
    "dios",
    "amigo",
    "amigos",
    "casa",
    "calor",
    "camión",
    "camiones",
    "carbón",
    "carro",
    "coche",
    "caballo",
    "cuarto",
    "cuando",
    "como",
    "cómo",
    "con",
    "centro",
    "ciudad",
    "cielo",
    "cinco",
    "cocina",
    "comida",
    "compró",
    "corazón",
    "cristiano",
    "escuela",
    "esquina",
    "gracias",
    "hacer",
    "haciendo",
    "hamaca",
    "hijo",
    "hora",
    "horrible",
    "hoy",
    "jamón",
    "lápiz",
    "lejos",
    "maestro",
    "mano",
    "mercado",
    "mesa",
    "mucho",
    "méxico",
    "mérida",
    "nombre",
    "otra",
    "patio",
    "pero",
    "portón",
    "pozole",
    "préstame",
    "qué",
    "quién",
    "quieres",
    "ramón",
    "rincón",
    "san",
    "señor",
    "señora",
    "solar",
    "tacos",
    "tarde",
    "temprano",
    "tienes",
    "todo",
    "tomar",
    "tren",
    "yucatán",
}

# Raíces mayas coloniales frecuentes con 'c' velar (pre-1984) para conversión segura a 'k'
_COLONIAL_C_ROOTS_PATTERN = re.compile(
    r"^(?:ca'an|caan|cab|cah|caj|cal|can|ican|chak?|chac|sac|zac|ek|ba'ac|"
    r"cuxtal|cool|col|co'ox|cox|cuuch|cuch|pek|peec|ciin|c'iin|c'a'ana'an|"
    r"mac|máac|maac|wic|winic|oc|ocol|uuc|uc|tucul|tzic|tsic|xuluc|yacunaj|"
    r"ja'atzcab|ja'atscab|áac'ab|aac'ab)$",
    re.IGNORECASE,
)

# Caracteres tipográficos que representan apóstrofo glotal en fuentes externas
_APOSTROPHE_VARIANTS_RE = re.compile(r"[’‘ʼʾʿˈ`´ʹ]")


# ==============================================================================
# Estructuras de Datos de Salida del Motor G2P
# ==============================================================================
@dataclass
class MayaPhonemeToken:
    """Representa una unidad fonológica desambiguada dentro de una palabra maya."""

    word: str
    grapheme: str
    ipa: str
    tts_token: str
    symbolic_code: str
    category: str
    apostrophe_type: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "grapheme": self.grapheme,
            "ipa": self.ipa,
            "tts_token": self.tts_token,
            "symbolic_code": self.symbolic_code,
            "category": self.category,
            "apostrophe_type": self.apostrophe_type,
        }


@dataclass
class MayaWordPhonemization:
    """Resultado fonológico completo para una palabra individual."""

    original_word: str
    normalized_word: str
    ipa: str
    tts_tokens: list[str]
    spanish_approx: str
    tokens: list[MayaPhonemeToken] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_word": self.original_word,
            "normalized_word": self.normalized_word,
            "ipa": self.ipa,
            "tts_tokens": self.tts_tokens,
            "tts_tokens_str": " ".join(self.tts_tokens),
            "spanish_approx": self.spanish_approx,
            "tokens": [t.to_dict() for t in self.tokens],
        }


@dataclass
class MayaG2PResult:
    """
    Contrato completo de salida fonológica de `MayaG2P1984.phonemize_sentence`.
    """

    original_text: str
    normalized_1984: str
    ipa_transcription: str
    tts_phonetic_tokens: str
    Spanish_Approx_Readout: str
    expected_acoustic_profile: dict[str, Any]
    words: list[MayaWordPhonemization] = field(default_factory=list)

    @property
    def spanish_approx_readout(self) -> str:
        """Alias snake_case para acceso uniforme desde APIs Python/JSON."""
        return self.Spanish_Approx_Readout

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_text": self.original_text,
            "normalized_1984": self.normalized_1984,
            "ipa_transcription": self.ipa_transcription,
            "tts_phonetic_tokens": self.tts_phonetic_tokens,
            "Spanish_Approx_Readout": self.Spanish_Approx_Readout,
            "spanish_approx_readout": self.Spanish_Approx_Readout,
            "expected_acoustic_profile": self.expected_acoustic_profile,
            "words": [w.to_dict() for w in self.words],
        }


# ==============================================================================
# Clase Principal: Motor G2P Norma Ortográfica de Agosto de 1984
# ==============================================================================
class MayaG2P1984:
    """
    Conversor Grafema-a-Fonema (G2P) determinista de máxima coincidencia izquierda-derecha
    para Lengua Maya Yucateca bajo la Norma Ortográfica de Agosto de 1984.
    """

    _TOKEN_SPLIT_RE = re.compile(r"([A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+)")

    def normalize_orthography_1984(self, text: str) -> str:
        """
        Normaliza cualquier texto maya hacia el Alfabeto Práctico de Agosto de 1984:
        1. Unifica Unicode a NFC (vocales tónicas precompuestas `á, é, í, ó, ú`).
        2. Convierte apóstrofos tipográficos (`’`, `‘`, `ʼ`, `´`, `` ` ``) a `'` ASCII
           SIN reemplazarlos jamás por espacios ni pausas.
        3. Reescribe grafías pre-1984 / coloniales en palabras mayas:
           `tz'` -> `ts'`, `dz` -> `ts'`, `tz` -> `ts`, `c'` -> `k'`,
           y `c` velar ante `a/o/u` o final -> `k` (preservando préstamos hispanos).
        4. Limpia espacios redundantes y puntuación espuria conservando signos básicos.
        """
        if not text:
            return ""

        norm = unicodedata.normalize("NFC", text)
        norm = _APOSTROPHE_VARIANTS_RE.sub("'", norm)
        # Colapsar múltiples apóstrofos accidentales consecutivos en uno solo
        norm = re.sub(r"'{2,}", "'", norm)

        # Normalizar palabra por palabra preservando capitalización inicial
        def _replace_word(match: re.Match[str]) -> str:
            raw_word = match.group(0)
            return self._normalize_single_word_1984(raw_word)

        norm = self._TOKEN_SPLIT_RE.sub(_replace_word, norm)
        # Limpiar caracteres de control/puntuación exótica dejando puntuación estándar y apóstrofo
        norm = re.sub(r"[^\w\s'.,;:¡!¿?\-()«»\"ÁÉÍÓÚÜÑáéíóúüñ]", " ", norm)
        norm = re.sub(r"[ \t]+", " ", norm).strip()
        return norm

    def _normalize_single_word_1984(self, word: str) -> str:
        """Aplica reglas ortográficas de 1984 a un token léxico individual."""
        if not word:
            return word

        # Separar posibles apóstrofos iniciales/finales legítimos para evaluar préstamos
        core_lower = word.lower()
        stripped_lower = core_lower.strip("'")
        if stripped_lower in _SPANISH_LOANWORDS_PRESERVE:
            return word

        # Detectar si la palabra exhibe rasgos ortográficos claramente mayas o coloniales
        has_maya_traits = (
            "'" in core_lower
            or "tz" in core_lower
            or "dz" in core_lower
            or "ts" in core_lower
            or "k" in core_lower
            or "w" in core_lower
            or "x" in core_lower
            or bool(re.search(r"([aeiouáéíóú])\1", core_lower, re.IGNORECASE))
            or bool(re.search(r"[áéíóú][aeiou]", core_lower, re.IGNORECASE))
            or bool(_COLONIAL_C_ROOTS_PATTERN.match(stripped_lower))
        )

        is_title = word[:1].isupper() and (len(word) == 1 or word[1:].islower())
        is_upper = len(word) > 1 and word.isupper()

        w = core_lower

        # 1) Grafías africadas pre-1984 -> Norma 1984 (`tz'` / `dz` -> `ts'`, `tz` -> `ts`)
        w = w.replace("tz'", "ts'")
        w = w.replace("dz", "ts'")
        w = w.replace("tz", "ts")

        # 2) Oclusiva velar eyectiva colonial `c'` -> `k'` (cuidando no tocar `ch'`)
        w = re.sub(r"(?<![c])c'", "k'", w)

        # 3) `pp` colonial inicial ante vocal -> `p'` (ej. `ppak` -> `p'ak`)
        w = re.sub(r"^pp(?=[aeiouáéíóú])", "p'", w)

        # 4) `c` velar colonial ante `a, o, u` o en coda final -> `k` solo en raíces mayas
        if has_maya_traits:
            # No alterar `ch` ni `ch'`; convertir `c` ante vocal posterior/central o fin de palabra
            w = re.sub(r"c(?=[aouáóú])", "k", w)
            w = re.sub(r"c(?=[ptklmnsxyw])", "k", w)
            w = re.sub(r"c$", "k", w)

        if is_upper:
            return w.upper()
        if is_title and w:
            # Si inicia con apóstrofo prevocálico ('otoch), capitalizar la primera letra real
            if w.startswith("'") and len(w) > 1:
                return "'" + w[1].upper() + w[2:]
            return w[0].upper() + w[1:]
        return w

    def phonemize_word(self, word: str) -> MayaWordPhonemization:
        """
        Analiza y fonemiza una palabra bajo las reglas contextuales de Agosto de 1984.
        Clasifica cada apóstrofo en Tipo 1 (`C'`), Tipo 2 (`V₁'V₁`), Tipo 3 (`V'`)
        o Tipo 4 (`'V` / `C'V` no eyectivable) sin introducir pausas artificiales.
        """
        clean_orig = word.strip(" \t\r\n.,;:¡!¿?()«»\"—–-")
        norm_word = self._normalize_single_word_1984(
            _APOSTROPHE_VARIANTS_RE.sub("'", unicodedata.normalize("NFC", clean_orig))
        )
        w = norm_word.lower()
        n = len(w)

        tokens: list[MayaPhonemeToken] = []
        i = 0

        while i < n:
            rem3 = w[i : i + 3]
            rem2 = w[i : i + 2]
            ch = w[i]

            # ------------------------------------------------------------------
            # PRECEDENCIA 1: Consonantes Eyectivas / Glotalizadas (`C'` - Tipo 1)
            # Dígrafos eyectivos de 3 caracteres (`ts'`, `ch'`) van primero.
            # ------------------------------------------------------------------
            if rem3 in _EJECTIVE_MAP:
                ipa, tts_tok, sym, es_apx = _EJECTIVE_MAP[rem3]
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=rem3,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="ejective_consonant",
                        apostrophe_type="TYPE_1_EJECTIVE",
                    )
                )
                i += 3
                continue

            if rem2 in _EJECTIVE_MAP:
                ipa, tts_tok, sym, es_apx = _EJECTIVE_MAP[rem2]
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=rem2,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="ejective_consonant",
                        apostrophe_type="TYPE_1_EJECTIVE",
                    )
                )
                i += 2
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 2: Dígrafos consonánticos simples (`ch`, `ts`, `rr`, `ll`, `qu`)
            # ------------------------------------------------------------------
            if rem2 in _DIGRAPH_CONSONANT_MAP:
                # Para `qu` en préstamos hispanos ante e/i (ej. "quieres")
                if rem2 == "qu" and (i + 2 >= n or w[i + 2] not in ("e", "i", "é", "í")):
                    pass
                else:
                    ipa, tts_tok, sym, _es_apx, cat = _DIGRAPH_CONSONANT_MAP[rem2]
                    tokens.append(
                        MayaPhonemeToken(
                            word=norm_word,
                            grapheme=rem2,
                            ipa=ipa,
                            tts_token=tts_tok,
                            symbolic_code=sym,
                            category=cat,
                            apostrophe_type=None,
                        )
                    )
                    i += 2
                    continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 3: Vocal Rearticulada Homorgánica (`V₁'V₁` - Tipo 2)
            # Incluye variantes con tilde tonal (`V́₁'V₁` o `V₁'V́₁`, ej. `k'a'ana'an`).
            # ------------------------------------------------------------------
            if (
                i + 2 < n
                and ch in _ALL_VOWELS
                and w[i + 1] == "'"
                and w[i + 2] in _ALL_VOWELS
                and _BASE_VOWEL_MAP[ch] == _BASE_VOWEL_MAP[w[i + 2]]
            ):
                base_v = _BASE_VOWEL_MAP[ch]
                ipa, tts_tok, sym, _es_apx = _REARTICULATED_VOWEL_MAP[base_v]
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=rem3,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="rearticulated_vowel",
                        apostrophe_type="TYPE_2_REARTICULATED",
                    )
                )
                i += 3
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 4: Vocal Larga con Tono Alto (`V́₁V₁` o `V₁V́₁`, ej. `áa`, `óo`)
            # ------------------------------------------------------------------
            if (
                i + 1 < n
                and ch in _ALL_VOWELS
                and w[i + 1] in _ALL_VOWELS
                and _BASE_VOWEL_MAP[ch] == _BASE_VOWEL_MAP[w[i + 1]]
                and (ch in _ACUTE_VOWELS or w[i + 1] in _ACUTE_VOWELS)
            ):
                base_v = _BASE_VOWEL_MAP[ch]
                ipa, tts_tok, sym, _es_apx = _HIGH_LONG_VOWEL_MAP[base_v]
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=rem2,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="high_tone_long_vowel",
                        apostrophe_type=None,
                    )
                )
                i += 2
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 5: Vocal Larga con Tono Bajo (`V₁V₁`, ej. `aa`, `ii`, `oo`)
            # ------------------------------------------------------------------
            if (
                i + 1 < n
                and ch in _PLAIN_VOWELS
                and w[i + 1] == ch
            ):
                ipa, tts_tok, sym, _es_apx = _LOW_LONG_VOWEL_MAP[ch]
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=rem2,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="low_tone_long_vowel",
                        apostrophe_type=None,
                    )
                )
                i += 2
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 6: Cierre Glotal Posvocálico / Final (`V'` - Tipo 3)
            # Ej. `ma'`, `ja'`, `sa'`, `-e'`, `-o'`, o hiato heterorgánico `V₁'V₂`.
            # ------------------------------------------------------------------
            if i + 1 < n and ch in _ALL_VOWELS and w[i + 1] == "'":
                base_v = _BASE_VOWEL_MAP[ch]
                ipa, tts_tok, sym, _es_apx = _GLOTTAL_VOWEL_MAP[base_v]
                if ch in _ACUTE_VOWELS:
                    ipa = f"{ch}ʔ"
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=rem2,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="glottalized_vowel",
                        apostrophe_type="TYPE_3_CODA_GLOTTAL",
                    )
                )
                i += 2
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 7: Ataque Glotal Prevocálico o Frontera Morfológica (Tipo 4)
            # Caso `'V` inicial (ej. `'otoch`) o `C'V` con `C` no eyectivable (`y'éetel`),
            # o cierre glotal tras vocal larga (`áa'`).
            # ------------------------------------------------------------------
            if ch == "'":
                is_after_long_vowel = bool(
                    tokens
                    and tokens[-1].category in ("high_tone_long_vowel", "low_tone_long_vowel")
                    and (i + 1 >= n or w[i + 1] not in _ALL_VOWELS)
                )
                ap_type = (
                    "TYPE_3_CODA_GLOTTAL"
                    if is_after_long_vowel
                    else "TYPE_4_BOUNDARY_GLOTTAL"
                )
                tok_name = "glot_coda" if is_after_long_vowel else "glot_ons"
                sym_code = "GLOT_CODA" if is_after_long_vowel else "GLOT_ONS"
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme="'",
                        ipa="ʔ",
                        tts_token=tok_name,
                        symbolic_code=sym_code,
                        category="glottal_stop",
                        apostrophe_type=ap_type,
                    )
                )
                i += 1
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 8: Vocales Cortas Simples o Tónicas de Préstamo
            # ------------------------------------------------------------------
            if ch in _SHORT_VOWEL_MAP:
                ipa, tts_tok, sym, _es_apx = _SHORT_VOWEL_MAP[ch]
                cat = "high_tone_short_vowel" if ch in _ACUTE_VOWELS else "short_vowel"
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=ch,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category=cat,
                        apostrophe_type=None,
                    )
                )
                i += 1
                continue

            # ------------------------------------------------------------------
            # PRECEDENCIA 9: Consonantes Simples y Respaldo de Préstamos Hispanos
            # ------------------------------------------------------------------
            if ch == "c":
                # En préstamos hispanos conservados (ej. "centro" -> /s/, "casa" -> /k/)
                next_ch = w[i + 1] if (i + 1) < n else ""
                if next_ch in ("e", "i", "é", "í"):
                    ipa, tts_tok, sym = ("s", "s", "C_SOFT_LOAN")
                else:
                    ipa, tts_tok, sym = ("k", "k", "C_HARD_LOAN")
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=ch,
                        ipa=ipa,
                        tts_token=tts_tok,
                        symbolic_code=sym,
                        category="loan_consonant",
                        apostrophe_type=None,
                    )
                )
                i += 1
                continue

            if ch in _SINGLE_CONSONANT_MAP:
                ipa, tts_tok, sym, _es_apx, cat = _SINGLE_CONSONANT_MAP[ch]
                if ipa or tts_tok:
                    tokens.append(
                        MayaPhonemeToken(
                            word=norm_word,
                            grapheme=ch,
                            ipa=ipa,
                            tts_token=tts_tok,
                            symbolic_code=sym,
                            category=cat,
                            apostrophe_type=None,
                        )
                    )
                i += 1
                continue

            # Dígitos o caracteres alfanuméricos residuales
            if ch.isalnum():
                tokens.append(
                    MayaPhonemeToken(
                        word=norm_word,
                        grapheme=ch,
                        ipa=ch,
                        tts_token=ch,
                        symbolic_code=ch.upper(),
                        category="other",
                        apostrophe_type=None,
                    )
                )
            i += 1

        ipa_str = "".join(t.ipa for t in tokens)
        tts_list = [t.tts_token for t in tokens if t.tts_token]
        es_approx = self._build_spanish_approx_word(tokens)

        return MayaWordPhonemization(
            original_word=clean_orig,
            normalized_word=norm_word,
            ipa=ipa_str,
            tts_tokens=tts_list,
            spanish_approx=es_approx,
            tokens=tokens,
        )

    @staticmethod
    def _build_spanish_approx_word(tokens: list[MayaPhonemeToken]) -> str:
        """
        Construye una guía de lectura aproximada para hispanohablantes y motores
        de control sin dividir las vocales rearticuladas ni los cierres glotales en pausas.
        """
        pieces: list[str] = []
        for idx, tok in enumerate(tokens):
            g = tok.grapheme
            if g in _EJECTIVE_MAP:
                pieces.append(_EJECTIVE_MAP[g][3])
            elif tok.category == "rearticulated_vowel":
                base_v = _BASE_VOWEL_MAP.get(g[0], "a")
                pieces.append(_REARTICULATED_VOWEL_MAP[base_v][3])
            elif tok.category == "high_tone_long_vowel":
                base_v = _BASE_VOWEL_MAP.get(g[0], "a")
                pieces.append(_HIGH_LONG_VOWEL_MAP[base_v][3])
            elif tok.category == "low_tone_long_vowel":
                base_v = _BASE_VOWEL_MAP.get(g[0], "a")
                pieces.append(_LOW_LONG_VOWEL_MAP[base_v][3])
            elif tok.category == "glottalized_vowel":
                base_v = _BASE_VOWEL_MAP.get(g[0], "a")
                pieces.append(_GLOTTAL_VOWEL_MAP[base_v][3])
            elif tok.category == "glottal_stop":
                pieces.append("ʔ")
            elif tok.symbolic_code == "X_SH":
                pieces.append("sh")
            elif tok.symbolic_code == "W":
                next_is_vowel = (idx + 1 < len(tokens)) and (
                    "vowel" in tokens[idx + 1].category
                )
                pieces.append("w" if next_is_vowel else "u")
            else:
                pieces.append(g)
        return "".join(pieces)

    def phonemize_sentence(self, text: str) -> MayaG2PResult:
        """
        Ejecuta la normalización ortográfica de Agosto de 1984 y la fonemización
        completa de una oración o párrafo, generando AFI, secuencia de tokens TTS,
        lectura hispana de control y el perfil acústico esperado para validación en Praat.
        """
        normalized = self.normalize_orthography_1984(text)
        if not normalized:
            empty_profile = {
                "expected_glottal_events": 0,
                "expected_vocalic_glottal_drops": 0,
                "rearticulated_vowels": 0,
                "glottalized_vowels": 0,
                "boundary_glottal_stops": 0,
                "ejective_consonants": 0,
                "implosive_consonants": 0,
                "high_tone_syllables": 0,
                "low_tone_long_vowels": 0,
                "long_vowel_syllables": 0,
                "short_vowel_syllables": 0,
                "word_count": 0,
                "apostrophe_breakdown": {
                    "TYPE_1_EJECTIVE": {"count": 0, "items": []},
                    "TYPE_2_REARTICULATED": {"count": 0, "items": []},
                    "TYPE_3_CODA_GLOTTAL": {"count": 0, "items": []},
                    "TYPE_4_BOUNDARY_GLOTTAL": {"count": 0, "items": []},
                },
                "tokens": [],
            }
            return MayaG2PResult(
                original_text=text or "",
                normalized_1984="",
                ipa_transcription="[]",
                tts_phonetic_tokens="",
                Spanish_Approx_Readout="",
                expected_acoustic_profile=empty_profile,
                words=[],
            )

        raw_lexemes = self._TOKEN_SPLIT_RE.findall(normalized)
        word_results: list[MayaWordPhonemization] = []
        all_tokens: list[MayaPhonemeToken] = []

        for lex in raw_lexemes:
            # Ignorar tokens que sean solo un apóstrofo suelto accidental sin letras
            if not any(ch.isalnum() for ch in lex):
                continue
            w_res = self.phonemize_word(lex)
            word_results.append(w_res)
            all_tokens.extend(w_res.tokens)

        ipa_words = [w.ipa for w in word_results if w.ipa]
        ipa_transcription = f"[{' '.join(ipa_words)}]" if ipa_words else "[]"

        # Separador explícito de frontera léxica `|` sin romper apóstrofos internos
        tts_word_Macro = [" ".join(w.tts_tokens) for w in word_results if w.tts_tokens]
        tts_phonetic_tokens = " | ".join(tts_word_Macro)

        es_approx_words = [w.spanish_approx for w in word_results if w.spanish_approx]
        spanish_readout = " ".join(es_approx_words)

        # Contabilización exacta del perfil acústico esperado por Praat (`prosody.py`)
        reart_count = 0
        glot_vowel_count = 0
        boundary_glot_count = 0
        ejective_count = 0
        implosive_count = 0
        high_tone_count = 0
        low_long_count = 0
        high_long_count = 0
        short_vowel_count = 0

        ap_breakdown: dict[str, dict[str, Any]] = {
            "TYPE_1_EJECTIVE": {"count": 0, "items": []},
            "TYPE_2_REARTICULATED": {"count": 0, "items": []},
            "TYPE_3_CODA_GLOTTAL": {"count": 0, "items": []},
            "TYPE_4_BOUNDARY_GLOTTAL": {"count": 0, "items": []},
        }

        for tok in all_tokens:
            if tok.category == "rearticulated_vowel":
                reart_count += 1
            elif tok.category == "glottalized_vowel":
                glot_vowel_count += 1
            elif tok.category == "glottal_stop":
                if tok.apostrophe_type == "TYPE_3_CODA_GLOTTAL":
                    glot_vowel_count += 1
                else:
                    boundary_glot_count += 1
            elif tok.category == "ejective_consonant":
                ejective_count += 1
            elif tok.category == "implosive_consonant":
                implosive_count += 1
            elif tok.category == "high_tone_long_vowel":
                high_tone_count += 1
                high_long_count += 1
            elif tok.category == "high_tone_short_vowel":
                high_tone_count += 1
                short_vowel_count += 1
            elif tok.category == "low_tone_long_vowel":
                low_long_count += 1
            elif tok.category == "short_vowel":
                short_vowel_count += 1

            if tok.apostrophe_type and tok.apostrophe_type in ap_breakdown:
                ap_breakdown[tok.apostrophe_type]["count"] += 1
                ap_breakdown[tok.apostrophe_type]["items"].append(
                    f"{tok.word} ({tok.grapheme} -> /{tok.ipa}/)"
                )

        vocalic_glottal_drops = reart_count + glot_vowel_count + boundary_glot_count
        expected_glottal_events = vocalic_glottal_drops + ejective_count
        long_vowel_syllables = high_long_count + low_long_count

        expected_profile: dict[str, Any] = {
            "expected_glottal_events": expected_glottal_events,
            "expected_vocalic_glottal_drops": vocalic_glottal_drops,
            "rearticulated_vowels": reart_count,
            "glottalized_vowels": glot_vowel_count,
            "boundary_glottal_stops": boundary_glot_count,
            "ejective_consonants": ejective_count,
            "implosive_consonants": implosive_count,
            "high_tone_syllables": high_tone_count,
            "low_tone_long_vowels": low_long_count,
            "long_vowel_syllables": long_vowel_syllables,
            "short_vowel_syllables": short_vowel_count,
            "word_count": len(word_results),
            "apostrophe_breakdown": ap_breakdown,
            "tokens": [t.to_dict() for t in all_tokens],
        }

        return MayaG2PResult(
            original_text=text,
            normalized_1984=normalized,
            ipa_transcription=ipa_transcription,
            tts_phonetic_tokens=tts_phonetic_tokens,
            Spanish_Approx_Readout=spanish_readout,
            expected_acoustic_profile=expected_profile,
            words=word_results,
        )
