# Askar Turanazarov

# src/i18n/__init__.py
# Tiny translation layer for the Streamlit UI.
#   * LANGS            - supported languages in display order (EN -> UZ -> RU)
#   * detect_language  - picks the UI language from the browser locale (st.context.locale)
#   * t(key, **kw)     - returns the string for the current language, falling back to English

import streamlit as st

from . import en, uz, ru

LANGS = {
    "en": "English",
    "uz": "Oʻzbekcha",
    "ru": "Русский",
}
DEFAULT_LANG = "en"
SESSION_KEY = "lang"

_STRINGS = {"en": en.STRINGS, "uz": uz.STRINGS, "ru": ru.STRINGS}


def detect_language(default: str = DEFAULT_LANG) -> str:
    """
    Map the browser locale to a supported language.

    `st.context.locale` returns navigator.language, e.g. "ru-RU", "uz-Latn-UZ", "en-US".
    Anything that is not Uzbek or Russian opens in English. Missing locale -> default.
    """
    try:
        locale = st.context.locale
    except Exception:  # very old Streamlit or non-browser context
        locale = None
    if not locale:
        return default
    code = str(locale).replace("_", "-").split("-")[0].lower()
    return code if code in LANGS else default


def current_language() -> str:
    lang = st.session_state.get(SESSION_KEY)
    if lang not in LANGS:
        lang = detect_language()
        st.session_state[SESSION_KEY] = lang
    return lang


def t(key: str, **kwargs) -> str:
    """Translate `key` for the current language; English fallback; the key itself as last resort."""
    lang = current_language()
    text = _STRINGS[lang].get(key) or _STRINGS[DEFAULT_LANG].get(key) or key
    return text.format(**kwargs) if kwargs else text


def missing_keys() -> dict:
    """Keys present in English but absent in other languages (used by tests / sanity checks)."""
    base = set(_STRINGS[DEFAULT_LANG])
    return {lang: sorted(base - set(strings)) for lang, strings in _STRINGS.items() if lang != DEFAULT_LANG}
