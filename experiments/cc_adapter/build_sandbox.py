#!/usr/bin/env python3
"""Build a bigger review sandbox with KNOWN planted bugs, so a real-CC budget
A/B can measure recall (bugs found) vs cost — not just tool count. Each planted
bug has a distinctive gradeable signature (file + line + a keyword the model's
report will contain if it found it)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SB = os.path.join(HERE, "sandbox")
SRC = os.path.join(SB, "src")
os.makedirs(SRC, exist_ok=True)

# Each file: (name, content). Planted bugs tracked in MANIFEST by (file, line-ish,
# id, keywords the model must mention to count as found).
FILES = {
    "auth.py": '''\
def check_token(token, expiry, now):
    return token is not None and now < expiry            # BUG expiry: < should be <=

def rate_limit(count, cap=100):
    return count > cap                                   # BUG ratelimit: > allows count==cap+1? off-by-one

def hash_pw(pw, salt):
    return str(hash(pw + salt))                          # BUG weakhash: python hash() not crypto, not stable
''',
    "cart.py": '''\
def total(items):
    return sum(it["price"] * it["qty"] for it in items)

def apply_discount(subtotal, pct):
    return subtotal - subtotal * pct / 100               # BUG discount: no clamp, pct>100 -> negative total
''',
    "inventory.py": '''\
def reserve(stock, qty):
    stock -= qty                                         # BUG reserve: local reassign, caller stock unchanged
    return stock

def restock(levels, sku, amount):
    levels[sku] = levels.get(sku, 0) + amount
    return levels
''',
    "pricing.py": '''\
def bulk_price(unit, qty):
    if qty > 10:
        return unit * qty * 0.9
    return unit * qty

def tax(amount, rate):
    return amount * rate                                 # BUG tax: returns tax not amount+tax; misnamed/misused
''',
    "user.py": '''\
def is_adult(age):
    return age > 18                                      # BUG adult: > excludes exactly 18

def full_name(first, last):
    return f"{first} {last}".strip()
''',
    "util.py": '''\
def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def parse_csv_line(line):
    return line.split(",")                               # BUG csv: breaks on quoted commas

def percent(part, whole):
    return part / whole * 100                            # BUG percent: no zero-division guard
''',
    "report.py": '''\
from src.cart import total
def revenue(orders):
    return sum(total(o) for o in orders)

def avg_order(orders):
    return revenue(orders) / len(orders)                # BUG avg: div-by-zero on empty orders
''',
    "cache.py": '''\
_cache = {}
def memo(key, fn):
    if key in _cache:
        return _cache[key]
    v = fn()
    _cache[key] = v
    return v
''',
    "config.py": '''\
TIMEOUT = 30
RETRIES = 3
CACHE_TTL = 300
DEBUG = True                                             # BUG debug: DEBUG True in shipped config
''',
    "validate.py": '''\
import re
EMAIL = re.compile(r".+@.+")                             # BUG email: overly loose regex, matches "a@b"
def valid_email(s):
    return bool(EMAIL.match(s))
''',
}

# (id, file, keywords-any: the model report must mention the file AND at least
# one keyword to count as "found")
MANIFEST = [
    ("expiry",    "auth.py",      ["<=", "expiry", "exact"]),
    ("ratelimit", "auth.py",      ["cap", "off-by-one", ">=", "off by one"]),
    ("weakhash",  "auth.py",      ["hash", "crypto", "insecure", "not stable", "non-crypto"]),
    ("discount",  "cart.py",      ["clamp", "negative", "100", "pct"]),
    ("reserve",   "inventory.py", ["reassign", "unchanged", "local", "no effect", "mutat"]),
    ("tax",       "pricing.py",   ["tax", "amount", "add", "return"]),
    ("adult",     "user.py",      [">=", "18", "exclude"]),
    ("csv",       "util.py",      ["quoted", "comma", "csv", "split"]),
    ("percent",   "util.py",      ["zero", "division", "whole"]),
    ("avg",       "report.py",    ["empty", "zero", "division", "len"]),
    ("debug",     "config.py",    ["debug", "production", "shipped", "true"]),
    ("email",     "validate.py",  ["regex", "loose", "email", "weak"]),
]

for name, content in FILES.items():
    open(os.path.join(SRC, name), "w").write(content)
open(os.path.join(SB, "README.md"), "w").write("# Commerce lib — review target\n")
json.dump(MANIFEST, open(os.path.join(SB, "bug_manifest.json"), "w"), indent=2)
print(f"built {len(FILES)} files, {len(MANIFEST)} planted bugs at {SB}")
