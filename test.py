from som import SOM
from csv import reader

# prepare data
f = open('input.csv', 'r')
r = reader(f)
r.next()  # skip header
data = []

# parse data
for row in r:
    l = []
    n = row[0]
    v = map(float, row[1:])
    v = map(lambda x: x/max(v), v)
    l.append(n)
    for x in v:
        l.append(x)
    data.append(l)

# create SOM & train
s = SOM()
results = s.train(data)

# restart if only one node (growth threshold not triggered)
while len(results) == 1:
    print 'Only one node, restarting...'
    s = SOM()
    results = s.train(data)

# print all node values
print 'All Output Node(s) Associations:'
for k, v in results.items():
    print '#{k} ({l}): {v}'.format(k=k, l=len(v), v=', '.join(v))

# print all nodes above threshold
print '\nGood (contains >={} items):'.format(s.threshold)
good = {k: v for k, v in results.items() if len(v) >= s.threshold}
for k, v in good.items():
    print '#{k} ({l}): {v}'.format(k=k, l=len(v), v=', '.join(v))

# print all nodes below threshold
print '\nBad (contains <{} items):'.format(s.threshold)
bad = {k: v for k, v in results.items() if len(v) < s.threshold}
for k, v in bad.items():
    print '#{k} ({l}): {v}'.format(k=k, l=len(v), v=', '.join(v))

# print weights of all nodes
print '\nFinal Weights:'
for i, node in enumerate(s.outputs):
    node_string = ['{:.4f}'.format(w) for w in node]
    print '#{i}: {w}'.format(i=i, w=', '.join(node_string))
