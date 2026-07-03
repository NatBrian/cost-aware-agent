def reserve(stock, qty):
    stock -= qty                                         # BUG reserve: local reassign, caller stock unchanged
    return stock

def restock(levels, sku, amount):
    levels[sku] = levels.get(sku, 0) + amount
    return levels
