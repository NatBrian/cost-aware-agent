def is_adult(age):
    return age > 18                                      # BUG adult: > excludes exactly 18

def full_name(first, last):
    return f"{first} {last}".strip()
