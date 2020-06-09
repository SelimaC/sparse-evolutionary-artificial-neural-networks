import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import find

weights = {}
weights[1] = np.loadtxt("Weights1.txt")
weights[2] = np.loadtxt("Weights2.txt")
weights[3] = np.loadtxt("Weights3.txt")
weights[4] = np.loadtxt("Weights4.txt")
b = {}
b[1] = np.loadtxt("Biases1.txt")
b[2] = np.loadtxt("Biases2.txt")
b[3] = np.loadtxt("Biases3.txt")
b[4] = np.loadtxt("Biases4.txt")

srelu_weights_layer1 = np.loadtxt("SReluWeights1.txt")
srelu_weights_layer2 = np.loadtxt("SReluWeights2.txt")
srelu_weights_layer3 = np.loadtxt("SReluWeights3.txt")

def srelu(tl, al, tr, ar, x):
    if x >= tr:
        return tr + ar * (x - tr)
    if x <= tl:
        return tl + al * (x - tl)
    return x

vfun = np.vectorize(srelu)

print("SReLu SET-MLP Keras version")

# Layer 1
tl = srelu_weights_layer1[0]
al = srelu_weights_layer1[1]
tr = srelu_weights_layer1[2]
ar = srelu_weights_layer1[3]

print("\n\n")
print("############################################################################")
print(f"\nLayer 1 - 4000 neurons")
print(f"Threshold left (tl): Mean = {np.round(tl.mean(), 2)}, Min = {np.round(tl.min(), 2)}, Max = {np.round(tl.max(), 2)}, Median = {np.round(np.median(tl), 2)}")
print(f"Slope left (al): Mean: {np.round(al.mean(), 2)}, Min = {np.round(al.min(), 2)}, Max = {np.round(al.max(), 2)}, Median = {np.round(np.median(al), 2)}")
print(f"Threshold right (tr): Mean: {np.round(tr.mean(), 2)}, Min = {np.round(tr.min(), 2)}, Max = {np.round(tr.max(), 2)}, Median = {np.round(np.median(tr), 2)}")
print(f"Slope right (ar): Mean = {np.round(ar.mean(), 2)}, Min = {np.round(ar.min(), 2)}, Max = {np.round(ar.max(), 2)}, Median = {np.round(np.median(ar), 2)}")

x = np.linspace(-10, 10, 10000)
i, j, v = find(weights[1])
unique, counts = np.unique(j, return_counts=True)
incoming_edges = counts
i, _, _ = find(weights[2])
unique, counts = np.unique(i, return_counts=True)
outgoing_edges = counts
sum_incoming_weights = np.abs(weights[1]).sum(axis=0)
sum_outgoing_weights = np.abs(weights[2]).sum(axis=1)
avg_incoming_weights = np.abs(weights[1]).max(axis=0)
avg_outgoing_weights = np.abs(weights[2]).max(axis=1)
edges = sum_incoming_weights + sum_outgoing_weights
#connections = incoming_edges + outgoing_edges
idxs = (edges).argsort()[:1000]
for idx in idxs:
    y = vfun(tl[idx], al[idx], tr[idx], ar[idx], x)
    plt.plot(x, y, '-')
plt.title('SReLu Layer 1 - 4000 neurons')
plt.legend(loc="upper left")
plt.show()
print(f"Hubs Threshold left (tl): Mean = {np.round(tl[idxs].mean(), 2)}, Min = {np.round(tl[idxs].min(), 2)}, Max = {np.round(tl[idxs].max(), 2)}, Median = {np.round(np.median(tl[idxs]), 2)}")
print(f"Hubs Slope left (al): Mean: {np.round(al[idxs].mean(), 2)}, Min = {np.round(al[idxs].min(), 2)}, Max = {np.round(al[idxs].max(), 2)}, Median = {np.round(np.median(al[idxs]), 2)}")
print(f"Hubs Threshold right (tr): Mean: {np.round(tr[idxs].mean(), 2)}, Min = {np.round(tr.min(), 2)}, Max = {np.round(tr[idxs].max(), 2)}, Median = {np.round(np.median(tr[idxs]), 2)}")
print(f"Hubs Slope right (ar): Mean = {np.round(ar[idxs].mean(), 2)}, Min = {np.round(ar[idxs].min(), 2)}, Max = {np.round(ar[idxs].max(), 2)}, Median = {np.round(np.median(ar[idxs]), 2)}")




# Layer 2
tl = srelu_weights_layer2[0]
al = srelu_weights_layer2[1]
tr = srelu_weights_layer2[2]
ar = srelu_weights_layer2[3]

print("############################################################################")
print(f"\nLayer 2 - 1000 neurons")
print(f"Threshold left (tl): Mean = {np.round(tl.mean(), 2)}, Min = {np.round(tl.min(), 2)}, Max = {np.round(tl.max(), 2)}, Median = {np.round(np.median(tl), 2)}")
print(f"Slope left (al): Mean: {np.round(al.mean(), 2)}, Min = {np.round(al.min(), 2)}, Max = {np.round(al.max(), 2)}, Median = {np.round(np.median(al), 2)}")
print(f"Threshold right (tr): Mean: {np.round(tr.mean(), 2)}, Min = {np.round(tr.min(), 2)}, Max = {np.round(tr.max(), 2)}, Median = {np.round(np.median(tr), 2)}")
print(f"Slope right (ar): Mean = {np.round(ar.mean(), 2)}, Min = {np.round(ar.min(), 2)}, Max = {np.round(ar.max(), 2)}, Median = {np.round(np.median(ar), 2)}")

x = np.linspace(-10, 10, 10000)
i, _, _ = find(weights[1])
unique, counts = np.unique(i, return_counts=True)
outgoing_edges = counts
sum_incoming_weights = np.abs(weights[2]).sum(axis=0)
sum_outgoing_weights = np.abs(weights[3]).sum(axis=1)
avg_incoming_weights = weights[1].mean(axis=0)
avg_outgoing_weights = weights[2].mean(axis=1)
edges = sum_incoming_weights + sum_outgoing_weights
idxs = (edges).argsort()[:250]
for idx in idxs:
    y = vfun(tl[idx], al[idx], tr[idx], ar[idx], x)
    plt.plot(x, y, '-')
plt.title('SReLu Layer 2 - 1000 neurons')
plt.legend(loc="upper left")
plt.show()
print(f"Hubs Threshold left (tl): Mean = {np.round(tl[idxs].mean(), 2)}, Min = {np.round(tl[idxs].min(), 2)}, Max = {np.round(tl[idxs].max(), 2)}, Median = {np.round(np.median(tl[idxs]), 2)}")
print(f"Hubs Slope left (al): Mean: {np.round(al[idxs].mean(), 2)}, Min = {np.round(al[idxs].min(), 2)}, Max = {np.round(al[idxs].max(), 2)}, Median = {np.round(np.median(al[idxs]), 2)}")
print(f"Hubs Threshold right (tr): Mean: {np.round(tr[idxs].mean(), 2)}, Min = {np.round(tr.min(), 2)}, Max = {np.round(tr[idxs].max(), 2)}, Median = {np.round(np.median(tr[idxs]), 2)}")
print(f"Hubs Slope right (ar): Mean = {np.round(ar[idxs].mean(), 2)}, Min = {np.round(ar[idxs].min(), 2)}, Max = {np.round(ar[idxs].max(), 2)}, Median = {np.round(np.median(ar[idxs]), 2)}")




# Layer 3
tl = srelu_weights_layer3[0]
al = srelu_weights_layer3[1]
tr = srelu_weights_layer3[2]
ar = srelu_weights_layer3[3]

print("############################################################################")
print(f"\nLayer 3 - 4000 neurons")
print(f"Threshold left (tl): Mean = {np.round(tl.mean(), 2)}, Min = {np.round(tl.min(), 2)}, Max = {np.round(tl.max(), 2)}, Median = {np.round(np.median(tl), 2)}")
print(f"Slope left (al): Mean: {np.round(al.mean(), 2)}, Min = {np.round(al.min(), 2)}, Max = {np.round(al.max(), 2)}, Median = {np.round(np.median(al), 2)}")
print(f"Threshold right (tr): Mean: {np.round(tr.mean(), 2)}, Min = {np.round(tr.min(), 2)}, Max = {np.round(tr.max(), 2)}, Median = {np.round(np.median(tr), 2)}")
print(f"Slope right (ar): Mean = {np.round(ar.mean(), 2)}, Min = {np.round(ar.min(), 2)}, Max = {np.round(ar.max(), 2)}, Median = {np.round(np.median(ar), 2)}")


x = np.linspace(-10, 10, 10000)
i, _, _ = find(weights[1])
unique, counts = np.unique(i, return_counts=True)
outgoing_edges = counts
sum_incoming_weights = np.abs(weights[3]).sum(axis=0)
sum_outgoing_weights = np.abs(weights[4]).sum(axis=1)
avg_incoming_weights = weights[1].mean(axis=0)
avg_outgoing_weights = weights[2].mean(axis=1)
edges = sum_incoming_weights + sum_outgoing_weights
idxs = (edges).argsort()[:1000]
for idx in idxs:
    y = vfun(tl[idx], al[idx], tr[idx], ar[idx], x)
    plt.plot(x, y, '-')
plt.title('SReLu Layer 3 - 4000 neurons')
plt.legend(loc="upper left")
plt.show()

print(f"Hubs Threshold left (tl): Mean = {np.round(tl[idxs].mean(), 2)}, Min = {np.round(tl[idxs].min(), 2)}, Max = {np.round(tl[idxs].max(), 2)}, Median = {np.round(np.median(tl[idxs]), 2)}")
print(f"Hubs Slope left (al): Mean: {np.round(al[idxs].mean(), 2)}, Min = {np.round(al[idxs].min(), 2)}, Max = {np.round(al[idxs].max(), 2)}, Median = {np.round(np.median(al[idxs]), 2)}")
print(f"Hubs Threshold right (tr): Mean: {np.round(tr[idxs].mean(), 2)}, Min = {np.round(tr.min(), 2)}, Max = {np.round(tr[idxs].max(), 2)}, Median = {np.round(np.median(tr[idxs]), 2)}")
print(f"Hubs Slope right (ar): Mean = {np.round(ar[idxs].mean(), 2)}, Min = {np.round(ar[idxs].min(), 2)}, Max = {np.round(ar[idxs].max(), 2)}, Median = {np.round(np.median(ar[idxs]), 2)}")



