import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model


csv_file = pd.read_csv('train.csv')
# print(csv_file.head())
# Drop rows with NaN values in 'x' or 'y' columns
csv_file = csv_file.dropna(subset=['x', 'y'])
xvalues = []
for i in csv_file['x']:
    xvalues.append(i)
yvalues = []
for i in csv_file['y']:
    yvalues.append(i)
plt.xlabel('X Values')
plt.ylabel('Y Values')
plt.plot(xvalues[:20], yvalues[:20], 'r.')
plt.savefig("train.png")

regressor = linear_model.LinearRegression()
regressor.fit(csv_file[['x']], csv_file['y'])

for xco in range(0,100,5):
    yco = regressor.predict([[xco]])
    plt.plot(xco, yco, 'b+')
plt.show()