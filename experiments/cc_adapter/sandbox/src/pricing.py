def bulk_price(unit, qty):
    if qty > 10:
        return unit * qty * 0.9
    return unit * qty

def tax(amount, rate):
    return amount * rate                                 # BUG tax: returns tax not amount+tax; misnamed/misused
