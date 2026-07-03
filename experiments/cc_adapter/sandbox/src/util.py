def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def parse_csv_line(line):
    return line.split(",")                               # BUG csv: breaks on quoted commas

def percent(part, whole):
    return part / whole * 100                            # BUG percent: no zero-division guard
