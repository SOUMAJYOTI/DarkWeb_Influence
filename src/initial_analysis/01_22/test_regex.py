import re

text = 'gfgfdAAA1234ZZZuijjk'

m = re.search(r'((\w)\2{2,})', text)

if m:
    print(m)
