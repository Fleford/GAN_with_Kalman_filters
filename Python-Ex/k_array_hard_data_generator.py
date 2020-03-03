import numpy as np
import matplotlib.pyplot as plt

k_array = np.loadtxt("k_array_ref_gan.txt")
print(k_array)
print(k_array.shape)
random_matrix = np.random.randint(2, size=k_array.shape)
for x in range(8):
    random_matrix = random_matrix * np.random.randint(2, size=k_array.shape)
print(random_matrix)
print(random_matrix.shape)

# Display matrix
plt.matshow(k_array*random_matrix)
plt.show()
