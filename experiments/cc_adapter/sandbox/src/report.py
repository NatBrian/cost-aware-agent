from src.cart import total
def revenue(orders):
    return sum(total(o) for o in orders)

def avg_order(orders):
    return revenue(orders) / len(orders)                # BUG avg: div-by-zero on empty orders
