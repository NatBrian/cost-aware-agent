def check_token(token, expiry, now):
    return token is not None and now < expiry            # BUG expiry: < should be <=

def rate_limit(count, cap=100):
    return count > cap                                   # BUG ratelimit: > allows count==cap+1? off-by-one

def hash_pw(pw, salt):
    return str(hash(pw + salt))                          # BUG weakhash: python hash() not crypto, not stable
