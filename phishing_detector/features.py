"""Інтерпретовані ознаки URL (лише для пояснення результату користувачу).

Нейромережа НЕ використовує ці ознаки — вона працює безпосередньо із символами.
Таблиця потрібна, щоб людині було легше зрозуміти, чому URL виглядає підозріло.
"""
import re
from typing import Dict, List
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "support",
    "online", "site", "icu", "cc", "pw", "live", "buzz", "shop",
}
KEYWORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm", "banking",
    "wallet", "billing", "security", "recovery", "unlock", "password", "validate",
]
BRANDS = [
    "paypal", "google", "apple", "microsoft", "amazon", "facebook", "netflix",
    "instagram", "linkedin", "dropbox", "ebay", "whatsapp", "binance", "coinbase",
    "monobank", "privat24", "novaposhta", "steam", "adobe", "icloud", "outlook",
]


class URLFeatureAnalyzer:
    """Обчислює зрозумілі людині ознаки URL."""

    def analyze(self, url: str) -> List[Dict]:
        raw = url.strip()
        try:
            parsed = urlparse(raw if "://" in raw else "http://" + raw)
            host = (parsed.hostname or "").lower()
            port = parsed.port
        except ValueError:
            parsed, host, port = None, "", None

        path_query = (parsed.path + ("?" + parsed.query if parsed.query else "")) if parsed else ""
        labels = [l for l in host.split(".") if l]
        is_ip = bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))
        tld = labels[-1] if len(labels) > 1 and not is_ip else ""
        n_sub = max(0, len(labels) - 2) if not is_ip else 0
        low = raw.lower()
        found_kw = [k for k in KEYWORDS if k in low]
        sub_part = ".".join(labels[:-2]) if len(labels) > 2 else ""
        brand_in_sub = [b for b in BRANDS if b in sub_part]
        scheme = parsed.scheme if parsed and "://" in raw else None

        def row(name, value, suspicious):
            return {"Ознака": name, "Значення": str(value), "Підозріло": "⚠️ так" if suspicious else "—"}

        return [
            row("Довжина URL", len(raw), len(raw) > 75),
            row("Довжина домену", len(host), len(host) > 30),
            row("Кількість піддоменів", n_sub, n_sub >= 3),
            row("Кількість дефісів у домені", host.count("-"), host.count("-") >= 2),
            row("Кількість цифр у домені", sum(c.isdigit() for c in host), sum(c.isdigit() for c in host) >= 4),
            row("IP-адреса замість домену", "так" if is_ip else "ні", is_ip),
            row("Символ @ в URL", "так" if "@" in raw else "ні", "@" in raw),
            row("Punycode (xn--)", "так" if "xn--" in low else "ні", "xn--" in low),
            row("Домен верхнього рівня (TLD)", tld or "—", tld in SUSPICIOUS_TLDS),
            row("Ключові слова (login, verify…)", ", ".join(found_kw) or "—", len(found_kw) >= 2),
            row("Назва бренду в піддомені", ", ".join(brand_in_sub) or "—", bool(brand_in_sub)),
            row("Схема", scheme or "не вказано", scheme == "http"),
            row("Нестандартний порт", port if port else "—", port not in (None, 80, 443)),
            row("Закодовані символи (%xx)", low.count("%"), low.count("%") >= 3),
            row("Довжина шляху та параметрів", len(path_query), len(path_query) > 80),
        ]
