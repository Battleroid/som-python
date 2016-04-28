from som import SOM
from csv import reader
from pprint import pprint

f = open('input.csv', 'r')
r = reader(f)
r.next()  # skip header
data = []

for row in r:
    l = []
    n = row[0]
    v = map(float, row[1:])
    v = map(lambda x: x/max(v), v)
    l.append(n)
    for x in v:
        l.append(x)
    data.append(l)

s = SOM()

pprint(s.__dict__)

good, bad = s.train(data)

print '*** GOOD, total', len(good)
for k, v in good.items():
    print k, 'has', len(v), 'records:', ', '.join(v)

print '*** BAD, total', len(bad)
for k, v in bad.items():
    print k, 'has', len(v), 'records:', ', '.join(v)
