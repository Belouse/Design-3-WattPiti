import pandas as pd
import matplotlib.pyplot as plt

# Define the CSV file path (update as needed)
csv_simulation_1 = "Thermique\Simulation 03-26\Offset_1_10W_parsed.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_simulation_1)

# create de X, Y arrays with the themperature values
x_simulation_1 = df['COORDINATES.X']
y_simulation_1 = df['COORDINATES.Y']
temp_simulation_1 = df['NDTEMP.T']

#plot the data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_simulation_1, y_simulation_1, temp_simulation_1, c=temp_simulation_1, cmap='turbo')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Temperature (°C)')
fig.suptitle('Temperature distribution')
plt.show()



