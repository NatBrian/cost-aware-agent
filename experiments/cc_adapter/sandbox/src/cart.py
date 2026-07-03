def total(items):
    return sum(it["price"] * it["qty"] for it in items)

def apply_discount(subtotal, pct):
    return subtotal - subtotal * pct / 100               # BUG discount: no clamp, pct>100 -> negative total
