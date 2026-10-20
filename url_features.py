import ipaddress
import re
from urllib.parse import urlparse
import math


SUSPICIOUS_WORDS = [
    "login",
    "verify",
    "account",
    "update",
    "secure",
    "signin",
    "banking",
    "password",
    "confirm",
    "wallet",
    "payment",
    "authenticate",
]

SHORTENERS = [
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "is.gd",
    "shorturl.at",
    "cutt.ly",
]


def extract_features(url):
    """Extract consistent numerical features from a URL."""

    url = str(url).strip()

    # Handle URLs that do not include a scheme.
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    try:
        ipaddress.ip_address(hostname)
        is_ip = 1
    except ValueError:
        is_ip = 0

    try:
        port_present = int(parsed.port is not None)
    except ValueError:
        port_present = 0

    # IP addresses have no subdomains; only split domain labels.
    # Strip the bare "www" prefix — it carries no security signal and
    # the PhiUSIIL dataset contains only www-prefixed legitimate URLs,
    # which causes the model to treat bare domains as phishing.
    if is_ip or not hostname:
        subdomain_count = 0
    else:
        hostname_parts = hostname.split(".")
        # Drop a leading "www" label before counting subdomains.
        if hostname_parts[0] == "www":
            hostname_parts = hostname_parts[1:]
        subdomain_count = max(0, len(hostname_parts) - 2)

    suspicious_word_count = sum(
        word in url.lower()
        for word in SUSPICIOUS_WORDS
    )

    shortener_present = int(
        any(shortener == hostname for shortener in SHORTENERS)
    )

    digits = sum(character.isdigit() for character in url)

    # Calculate hostname character entropy.
    if hostname:
        probabilities = [
            hostname.count(character) / len(hostname)
            for character in set(hostname)
        ]

        hostname_entropy = -sum(
            probability * math.log2(probability)
            for probability in probabilities
        )
    else:
        hostname_entropy = 0.0

    features = {
        "url_length": len(url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "at_count": url.count("@"),
        "question_count": url.count("?"),
        "equal_count": url.count("="),
        "underscore_count": url.count("_"),
        "percent_count": url.count("%"),
        "slash_count": url.count("/"),
        "digit_count": digits,
        "subdomain_count": subdomain_count,
        "uses_https": int(parsed.scheme.lower() == "https"),
        "has_ip_address": is_ip,
        "has_punycode": int("xn--" in hostname),
        "suspicious_word_count": suspicious_word_count,
        "is_shortener": shortener_present,
        "has_port": port_present,
        "has_fragment": int(bool(parsed.fragment)),
        "path_depth": len([part for part in path.split("/") if part]),
        "hostname_digit_count": sum(
            character.isdigit() for character in hostname
        ),
        "hostname_entropy": hostname_entropy,
    }

    return features