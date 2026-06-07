import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph
from numpy.linalg import eigh

# 1. ДАНІ: ПІВМІСЯЦІ

moon1 = np.array([
    [0.0, 1.0],
    [0.5, 1.2],
    [1.0, 1.0],
    [1.5, 1.2],
    [2.0, 1.0],
    [2.5, 1.2],
    [3.0, 1.0],
    [3.5, 1.2],
])

moon2 = np.array([
    [0.0, 0.0],
    [0.5, -0.2],
    [1.0, 0.0],
    [1.5, -0.2],
    [2.0, 0.0],
    [2.5, -0.2],
    [3.0, 0.0],
    [3.5, -0.2],
])

# 2. ДАНІ: ДРУЗІ

group1 = np.array([
    [0.0, 1.0],
    [0.5, 1.6],
    [1.0, 2.0],
    [1.5, 2.1],
    [2.0, 1.8],
    [2.5, 1.2],
    [3.0, 0.6],
    [3.5, 0.2]
])

group2 = np.array([
    [0.0, -1.0],
    [0.5, -1.5],
    [1.0, -1.9],
    [1.5, -2.0],
    [2.0, -1.7],
    [2.5, -1.2],
    [3.0, -0.7],
    [3.5, -0.3]
])

group3 = np.array([
    [1.0, 0.2],
    [1.5, -0.3],
    [2.0, 0.4],
    [2.5, -0.4],
    [3.0, 0.5],
    [3.5, -0.2],
    [4.0, 0.3],
    [4.5, -0.1]
])

MODE = input("виберіть значення : friends or moons = ").strip().lower()
#  ВИБІР X

if MODE == "moons":
    X = np.vstack([moon1, moon2])
    k = 2

elif MODE == "friends":
    X = np.vstack([group1, group2, group3])
    k = 3

else:
    raise ValueError("MODE має бути 'moons' або 'friends'")

# 3. ГРАФ СУСІДІВ

A = kneighbors_graph(X, n_neighbors=3, mode='connectivity').toarray()
A = (A + A.T) / 2

G = nx.from_numpy_array(A)
pos = nx.spring_layout(G, seed=42)

plt.figure(figsize=(6,6))
nx.draw(G, pos, node_size=200, node_color='lightblue', edge_color='gray')
plt.title("Граф даних")
plt.show()

# 4. K-MEANS

kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
labels_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=labels_kmeans, cmap='viridis', s=120)
plt.title("K-Means")
plt.grid()
plt.show()

# 5. ЛАПЛАС

D = np.diag(np.sum(A, axis=1))
L = D - A

plt.figure(figsize=(5,5))
plt.imshow(L, cmap='coolwarm')
plt.title("Матриця Лапласа")
plt.colorbar()
plt.show()

# 6. СПЕКТР

eigvals, eigvecs = eigh(L)

eigvals = np.clip(np.real(eigvals), 0, None)

plt.figure(figsize=(8,4))

x = np.arange(len(eigvals))

plt.plot(x, eigvals, 'o-', color='black')

#значення власних чисел
for i, val in enumerate(eigvals):
    plt.text(i, val, f"{val:.2f}", ha='center', va='bottom')

# осі
plt.xticks(x, [f"λ{i}" for i in x])

plt.xlabel("Номер власного значення")
plt.ylabel("Власне значення")

plt.title("Спектр матриці Лапласа")

plt.grid(True)
plt.show()

# друк у консоль
print("\nВласні значення:")
for i, v in enumerate(eigvals):
    print(f"λ{i} = {v:.4f}")

# 7. СПЕКТРАЛЬНА КЛАСТЕРИЗАЦІЯ

# беремо 3 ,або 2 найменші ненульові власні вектори бо за умови в нас 3 группи людей б фбо дані вигляду 2 лун
embedding = eigvecs[:, 1: k+1]

# k-means у спектральному просторі
kmeans_spec = KMeans(n_clusters= k,
                     n_init=10,
                     random_state=0)

labels_spec = kmeans_spec.fit_predict(embedding)

# 8. ПОРІВНЯННЯ

fig, ax = plt.subplots(1, 2, figsize=(12,5))

ax[0].scatter(X[:,0], X[:,1], c=labels_kmeans, cmap='viridis')
ax[0].set_title("K-Means")

ax[1].scatter(X[:,0], X[:,1], c=labels_spec, cmap='viridis')
ax[1].set_title("Спектральна ")

plt.show()