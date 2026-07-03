import re
EMAIL = re.compile(r".+@.+")                             # BUG email: overly loose regex, matches "a@b"
def valid_email(s):
    return bool(EMAIL.match(s))
