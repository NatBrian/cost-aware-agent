_cache = {}
def memo(key, fn):
    if key in _cache:
        return _cache[key]
    v = fn()
    _cache[key] = v
    return v
