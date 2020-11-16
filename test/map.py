import numpy as np
from matplotlib import pyplot as plt

x = np.random.randint(
    low=0.0, high=100, size=(2000, 1))
y = np.random.randint(
    low=0.0, high=100, size=(2000, 1))
health_code = {'healthy': 0, 'infected': 1,
               'recovered': 2, 'deceased': 3}
health = np.zeros((2000, health_code['infected']))
index_array = np.random.choice(2000, 100, replace=False)
health[index_array] = 1
health_color = np.vectorize({
    health_code['healthy']: 'blue',
    health_code['infected']: 'red',
    health_code['recovered']: 'green',
    health_code['deceased']: 'black'
}.get)(health)

plt.scatter(x, y, s=10, c=health_color)

plt.show()
