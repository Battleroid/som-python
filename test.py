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

print """
Parameters summary:
    Epochs: {epochs}
    Cluster Threshold: >={threshold}
    Radius: {radius}
    Radius Decay Rate: {radius_decay}
    Rate: {rate}
    Rate Decay: {rate_decay}
    Growth Threshold: {growth}

Results summary:
    Total Final Outputs: {total_outputs}
    Total Canceled Outputs: {total_canceled}

--
""".format(
        epochs=s.epochs,
        threshold=s.threshold,
        radius=s.radius,
        radius_decay=s.radius_lower,
        rate=s.rate,
        rate_decay=s.rate_lower,
        growth=s.growth_threshold,
        total_outputs=len(s.outputs),
        total_canceled=len(s.canceled)
        )

# print all node values
print 'All Output Node(s) Associations:'
for k, v in results.items():
    print '#{k} ({l}): {v}'.format(k=k, l=len(v), v=', '.join(v))

# print weights of all nodes
print '\nFinal Weights:'
for i, node in enumerate(s.outputs):
    node_string = ['{:.4f}'.format(w) for w in node]
    print '#{i}: {w}'.format(i=i, w=', '.join(node_string))

# print weights, records, and indices of those canceled
print '\nCanceled Output Information:'
for k, v in s.canceled.iteritems():
    weights_string = ['{:.4f}'.format(w) for w in v['weights']]
    print '#{i} ({l}): {records}'.format(i=k, l=len(v['records']),
                                         records=', '.join(v['records']))
    print '#{i} weights: {w}'.format(i=k, w=', '.join(weights_string))
