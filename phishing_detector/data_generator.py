"""Генератор синтетичного датасету URL (легітимні vs фішингові).

Призначення — дати змогу запустити проєкт «з коробки» без завантаження
зовнішніх даних. Для реальних експериментів у курсовій роботі використовуйте
справжні датасети (PhishTank, OpenPhish, Tranco, Kaggle тощо) — див. README.
У генератор навмисно додано «важкі» приклади (легітимні URL зі словами login,
дефісами, підозрілими TLD та фішингові URL зі «звичайним» виглядом),
щоб класи частково перетиналися.
"""
import random
import string

import pandas as pd

LEGIT_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "amazon.com", "wikipedia.org",
    "twitter.com", "instagram.com", "linkedin.com", "microsoft.com", "apple.com",
    "github.com", "stackoverflow.com", "reddit.com", "netflix.com", "paypal.com",
    "ebay.com", "bbc.com", "nytimes.com", "cnn.com", "medium.com", "dropbox.com",
    "adobe.com", "spotify.com", "twitch.tv", "whatsapp.com", "zoom.us", "slack.com",
    "notion.so", "cloudflare.com", "mozilla.org", "python.org", "kaggle.com",
    "coursera.org", "udemy.com", "khanacademy.org", "mit.edu", "stanford.edu",
    "harvard.edu", "nasa.gov", "europa.eu", "who.int", "ukr.net", "rozetka.com.ua",
    "privatbank.ua", "monobank.ua", "novaposhta.ua", "olx.ua", "kyivstar.ua",
    "gov.ua", "kpi.ua", "knu.ua", "bbc.co.uk", "theguardian.com", "walmart.com",
    "target.com", "bestbuy.com", "ikea.com", "booking.com", "airbnb.com",
    "tripadvisor.com", "imdb.com", "quora.com", "yahoo.com", "bing.com",
    "duckduckgo.com", "telegram.org", "discord.com", "steampowered.com",
    "epicgames.com", "tesla.com", "bankofamerica.com", "chase.com", "wellsfargo.com",
]

BRANDS = [
    "paypal", "google", "apple", "microsoft", "amazon", "facebook", "netflix",
    "instagram", "linkedin", "dropbox", "ebay", "whatsapp", "bankofamerica",
    "chase", "wellsfargo", "privat24", "monobank", "novaposhta", "binance",
    "coinbase", "dhl", "fedex", "steam", "adobe", "icloud", "outlook", "office365",
    "telegram", "twitter", "yahoo", "spotify", "docusign", "zoom", "walmart",
    "citibank", "hsbc", "revolut", "metamask", "olx", "kyivstar",
]

SUSPICIOUS_TLDS = [
    "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "link", "work", "support",
    "online", "site", "info", "buzz", "icu", "cc", "pw", "live", "shop",
]
COMMON_TLDS = ["com", "com", "com", "org", "net", "edu", "io", "co", "ua", "de", "uk"]

PHISH_WORDS = [
    "login", "signin", "verify", "secure", "account", "update", "confirm", "banking",
    "support", "wallet", "billing", "security", "recovery", "unlock", "password",
    "auth", "validate", "service", "alert", "helpdesk",
]

WORDS = [
    "green", "valley", "blue", "river", "smart", "home", "garden", "coffee", "bakery",
    "travel", "photo", "studio", "design", "market", "fresh", "city", "north", "sun",
    "moon", "star", "bright", "swift", "alpha", "beta", "delta", "nova", "pixel",
    "cloud", "code", "data", "health", "clinic", "dental", "fitness", "yoga", "music",
    "art", "books", "school", "academy", "learn", "kids", "pets", "auto", "motor",
    "bike", "sport", "team", "club", "event", "tickets", "hotel", "villa", "estate",
    "build", "tools", "metal", "wood", "paper", "print", "media", "news", "daily",
    "world", "global", "local", "union", "trade", "invest", "finance", "legal", "law",
    "tech", "soft", "net", "web", "digital", "lab", "works", "forge", "craft", "mill",
]

FREE_HOSTS = [
    "000webhostapp.com", "weebly.com", "blogspot.com", "firebaseapp.com", "web.app",
    "herokuapp.com", "github.io", "netlify.app", "wixsite.com", "glitch.me",
]

LEGIT_PATH_WORDS = [
    "news", "products", "about", "contact", "blog", "docs", "help", "support", "store",
    "category", "article", "search", "pricing", "careers", "login", "signin", "account",
    "download", "video", "watch", "en", "us", "settings", "profile", "orders",
]

HOMOGLYPHS = {"o": "0", "l": "1", "i": "l", "e": "3", "s": "5", "m": "rn", "w": "vv", "a": "4"}


class SyntheticURLGenerator:
    """Генерує збалансований набір легітимних і фішингових URL."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    # ---------- допоміжні методи ----------
    def _word(self) -> str:
        return self.rng.choice(WORDS)

    def _gib(self, lo: int = 6, hi: int = 14) -> str:
        n = self.rng.randint(lo, hi)
        alphabet = string.ascii_lowercase + string.digits
        return "".join(self.rng.choice(alphabet) for _ in range(n))

    def _hex(self, lo: int = 8, hi: int = 32) -> str:
        n = self.rng.randint(lo, hi)
        return "".join(self.rng.choice("0123456789abcdef") for _ in range(n))

    def _scheme(self, p_https: float) -> str:
        return "https" if self.rng.random() < p_https else "http"

    def _susp_tld(self, p: float = 0.75) -> str:
        if self.rng.random() < p:
            return self.rng.choice(SUSPICIOUS_TLDS)
        return self.rng.choice(["com", "net", "org"])

    # ---------- шляхи ----------
    def _legit_path(self) -> str:
        r = self.rng.random()
        if r < 0.25:
            return ""
        if r < 0.35:
            return "/"
        depth = self.rng.randint(1, 4)
        segs = []
        for _ in range(depth):
            if self.rng.random() < 0.65:
                segs.append(self.rng.choice(LEGIT_PATH_WORDS))
            else:
                seg = self._word()
                if self.rng.random() < 0.4:
                    seg += "-" + self._word()
                segs.append(seg)
        path = "/" + "/".join(segs)
        if self.rng.random() < 0.25:
            path += self.rng.choice([".html", ".php", ".aspx"])
        if self.rng.random() < 0.3:
            key = self.rng.choice(["id", "q", "page", "ref", "utm_source", "lang"])
            val = str(self.rng.randint(1, 9999)) if self.rng.random() < 0.6 else self._word()
            path += f"?{key}={val}"
        return path

    def _phish_path(self) -> str:
        if self.rng.random() < 0.15:
            return ""
        prefixes = ["/", "/wp-content/", "/wp-includes/", "/secure/", "/cgi-bin/", "/includes/", "/auth/"]
        kws = self.rng.sample(PHISH_WORDS, self.rng.randint(1, 3))
        path = self.rng.choice(prefixes) + "-".join(kws)
        path += self.rng.choice([".php", ".html", "", "/", ".aspx"])
        if self.rng.random() < 0.4:
            key = self.rng.choice(["id", "session", "token", "cmd", "key"])
            path += f"?{key}={self._hex()}"
        return path

    # ---------- легітимні ----------
    def _legit_known(self) -> str:
        domain = self.rng.choice(LEGIT_DOMAINS)
        sub = self.rng.choice(["www.", "", "", "en.", "docs.", "mail.", "accounts.",
                               "support.", "shop.", "blog.", "login.", "m."])
        return f"{self._scheme(0.9)}://{sub}{domain}{self._legit_path()}"

    def _legit_generated(self) -> str:
        r = self.rng.random()
        if r < 0.5:
            name = self._word()
        elif r < 0.85:
            name = self._word() + self._word()
        else:
            name = self._word() + "-" + self._word()
        tld = self.rng.choice(COMMON_TLDS)
        sub = "www." if self.rng.random() < 0.5 else ""
        return f"{self._scheme(0.8)}://{sub}{name}.{tld}{self._legit_path()}"

    def _legit_hard(self) -> str:
        """Легітимні, але «підозріло виглядаючі» URL."""
        kind = self.rng.randint(0, 3)
        if kind == 0:  # недорогий TLD
            return f"{self._scheme(0.7)}://{self._word()}{self._word()}.{self.rng.choice(['xyz', 'top', 'online', 'site', 'info', 'live'])}{self._legit_path()}"
        if kind == 1:  # CDN / хеші в шляху
            return f"{self._scheme(0.95)}://cdn.{self.rng.choice(LEGIT_DOMAINS)}/assets/{self._hex(12, 32)}.js"
        if kind == 2:  # внутрішня адреса
            return f"http://192.168.{self.rng.randint(0, 5)}.{self.rng.randint(1, 254)}/admin/{self.rng.choice(['login', 'status', 'settings'])}"
        return f"{self._scheme(0.9)}://{self.rng.choice(['secure', 'login', 'accounts', 'my'])}.{self._word()}-{self._word()}.{self.rng.choice(COMMON_TLDS)}/{self.rng.choice(['login', 'account', 'signin'])}"

    def _legit(self) -> str:
        return self.rng.choices(
            [self._legit_known, self._legit_generated, self._legit_hard], weights=[0.35, 0.45, 0.20]
        )[0]()

    # ---------- фішингові ----------
    def _brand_subdomain(self) -> str:
        brand, kw = self.rng.choice(BRANDS), self.rng.choice(PHISH_WORDS)
        tld = self._susp_tld()
        style = self.rng.randint(0, 2)
        if style == 0:
            host = f"{brand}.com.{kw}-{self._word()}.{tld}"
        elif style == 1:
            host = f"{brand}.{kw}.{self._word()}{self.rng.randint(1, 99)}.{tld}"
        else:
            host = f"{kw}.{brand}.com.{self._gib(4, 8)}.{tld}"
        return f"{self._scheme(0.55)}://{host}{self._phish_path()}"

    def _typo(self, brand: str) -> str:
        for _ in range(10):
            op = self.rng.choice(["homoglyph", "dup", "swap", "drop", "hyphen"])
            b = brand
            if op == "homoglyph":
                idx = [i for i, c in enumerate(b) if c in HOMOGLYPHS]
                if not idx:
                    continue
                i = self.rng.choice(idx)
                b = b[:i] + HOMOGLYPHS[b[i]] + b[i + 1:]
            elif op == "dup":
                i = self.rng.randrange(len(b))
                b = b[:i] + b[i] + b[i:]
            elif op == "swap" and len(b) > 2:
                i = self.rng.randrange(len(b) - 1)
                b = b[:i] + b[i + 1] + b[i] + b[i + 2:]
            elif op == "drop" and len(b) > 3:
                i = self.rng.randrange(len(b))
                b = b[:i] + b[i + 1:]
            elif op == "hyphen":
                i = self.rng.randint(1, len(b) - 1)
                b = b[:i] + "-" + b[i:]
            if b != brand:
                return b
        return brand + "s"

    def _typosquat(self) -> str:
        typo = self._typo(self.rng.choice(BRANDS))
        tld = self.rng.choice(["com", "net", "co", "org"]) if self.rng.random() < 0.6 else self.rng.choice(SUSPICIOUS_TLDS)
        path = self._phish_path() if self.rng.random() < 0.6 else self._legit_path()
        sub = "www." if self.rng.random() < 0.3 else ""
        return f"{self._scheme(0.6)}://{sub}{typo}.{tld}{path}"

    def _ip_host(self) -> str:
        ip = ".".join(str(self.rng.randint(1, 254)) for _ in range(4))
        port = self.rng.choice(["", "", ":8080", ":8443", ":81"])
        path = self._phish_path() or "/index.php"
        return f"{self._scheme(0.15)}://{ip}{port}{path}"

    def _keyword_domain(self) -> str:
        parts = [self.rng.choice(BRANDS)] + self.rng.sample(PHISH_WORDS, self.rng.randint(1, 2))
        self.rng.shuffle(parts)
        return f"{self._scheme(0.6)}://{'-'.join(parts)}.{self._susp_tld(0.7)}{self._phish_path()}"

    def _at_symbol(self) -> str:
        if self.rng.random() < 0.5:
            user = self.rng.choice(LEGIT_DOMAINS)
        else:
            user = f"{self.rng.choice(PHISH_WORDS)}.{self.rng.choice(BRANDS)}.com"
        return f"{self._scheme(0.5)}://{user}@{self._gib(5, 10)}.{self._susp_tld()}{self._phish_path()}"

    def _free_hosting(self) -> str:
        host = f"{self.rng.choice(BRANDS)}-{self.rng.choice(PHISH_WORDS)}-{self._gib(3, 7)}.{self.rng.choice(FREE_HOSTS)}"
        return f"{self._scheme(0.8)}://{host}{self._phish_path()}"

    def _gibberish_domain(self) -> str:
        sub = f"{self._gib(3, 6)}." if self.rng.random() < 0.4 else ""
        return f"{self._scheme(0.5)}://{sub}{self._gib(8, 18)}.{self._susp_tld(0.8)}{self._phish_path()}"

    def _subtle_phish(self) -> str:
        """Фішинг зі «звичайним» виглядом домену — складний випадок."""
        name = self._word() + self._word() + (str(self.rng.randint(1, 99)) if self.rng.random() < 0.3 else "")
        tld = self.rng.choice(["com", "net", "org"])
        return f"{self._scheme(0.7)}://{name}.{tld}{self._phish_path()}"

    def _phish(self) -> str:
        makers = [self._brand_subdomain, self._typosquat, self._ip_host, self._keyword_domain,
                  self._at_symbol, self._free_hosting, self._gibberish_domain, self._subtle_phish]
        weights = [0.15, 0.15, 0.08, 0.17, 0.05, 0.12, 0.13, 0.15]
        return self.rng.choices(makers, weights=weights)[0]()

    # ---------- публічний API ----------
    def _unique(self, fn, n: int) -> list:
        seen, attempts = set(), 0
        while len(seen) < n and attempts < n * 30:
            seen.add(fn())
            attempts += 1
        return list(seen)

    def generate(self, n_samples: int = 20000) -> pd.DataFrame:
        """Повертає DataFrame зі стовпцями url, label (1 — фішинг, 0 — легітимний)."""
        half = max(1, n_samples // 2)
        legit = self._unique(self._legit, half)
        phish = self._unique(self._phish, half)
        df = pd.DataFrame({
            "url": legit + phish,
            "label": [0] * len(legit) + [1] * len(phish),
        })
        df = df.drop_duplicates("url")
        return df.sample(frac=1.0, random_state=self.rng.randint(0, 10**6)).reset_index(drop=True)
