from collections import Counter
import numpy as np
import csv
import matplotlib.pyplot as plt

outdeg = Counter()

with open('./enwiki-2013.txt', 'r') as f:
    for line in f:
        if line[0] == '#':   # más rápido que startswith
            continue
        parts = line.split(None, 1)  # solo necesitamos u
        if len(parts) == 2:
            u = int(parts[0])
            outdeg[u] += 1

vals = np.array(list(outdeg.values()), dtype=np.int32)

min_out = vals.min()
max_out = vals.max()

# Bins de igual tamaño (lineales)
bins = np.linspace(min_out, max_out, 51)  # 80 bins

hist, edges = np.histogram(vals, bins=bins)

print('Max outdegree:', max_out)
# Find the node with maximum outdegree
max_node = max(outdeg, key=outdeg.get)

# Load node names from CSV
names = {}
with open('./enwiki-2013-names.csv', 'r', encoding='utf-8', errors='replace') as f:
    reader = csv.DictReader(f)
    for row in reader:
        node_id = int(row['node_id'])
        names[node_id] = row['name']

# Print the node_id and name for the max outdegree node
print(f'"{max_node}","{names.get(max_node, "Unknown")}"')

plt.figure(figsize=(10,6))
plt.bar(edges[:-1], hist, width=np.diff(edges), edgecolor='black')
plt.yscale('log')
plt.xlabel('Outdegree')
plt.ylabel('Frequency')
plt.title('Outdegree Histogram (Wikipedia)')

# Añadir valores encima de cada barra en texto pequeño
for i in range(len(hist)):
    plt.text(edges[i] + (edges[i+1] - edges[i])/2, hist[i], f'{hist[i]}', ha='center', va='bottom', fontsize=6)

plt.savefig('outdegree-hist.png', dpi=150)
