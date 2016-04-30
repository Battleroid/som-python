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

results = s.train(data)

while len(results) == 1:
    print 'Only one node, restarting...'
    s = SOM()
    results = s.train(data)

for k, v in results.items():
    print '#', k, '(', len(v),  '):', v

print 'Good:'
good = {k: v for k, v in results.items() if len(v) >= s.threshold}
for k, v in good.items():
    print k, v

print 'Bad:'
bad = {k: v for k, v in results.items() if len(v) < s.threshold}
for k, v in bad.items():
    print k, v
