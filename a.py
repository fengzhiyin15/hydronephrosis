import matplotlib.pyplot as plt
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D

# Import the data and assign to x, y and z
print('File should be 3 columns of 9 numbers.')
file_input = input('Enter file name (with .txt): ')

# Uncomment the code below if using the predetermined x and y coordinates
"""
sensor_type = input('Double or single sensor (d/s): ')
if sensor_type == 'd':
    x = [-20000, -20000, -20000, 0, 0, 0, 20000, 20000, 20000]
    y = [5000, 0, -5000, 5000, 0, -5000, 5000, 0, -5000]
elif sensor_type == 's':
    x = [-10000, -10000, -10000, 0, 0, 0, 10000, 10000, 10000]
    y = [5000, 0, -5000, 5000, 0, -5000, 5000, 0, -5000]
else:
    print('Error: Unknown sensor type.')
"""
x = np.genfromtxt(file_input, dtype=float, usecols=0)  # comment out if manually inputting sensor type
y = np.genfromtxt(file_input, dtype=float, usecols=1)  # comment out if manually inputting sensor type
z = np.genfromtxt(file_input, dtype=float, usecols=2)  # usecols=2 ----> usecols=0

# Find the plane that best fits the data
A = []

for i in range(len(x)):
    row = []
    row.append(x[i])
    row.append(y[i])
    row.append(1)
    A.append(row)

B = []

for i in range(len(z)):
    row = []
    row.append(z[i])
    B.append(row)

# Find the coefficients that create the plane of best fit
coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)

x2 = [max(x), min(x)]
y2 = [max(y), min(y)]
# Create grid over which to fit the plane
X, Y = np.meshgrid(x2, y2, copy=False)  # x2 ---> x and y2 ---> y if inputting sensor type

# Plane equation
Z = coeff[0]*X + coeff[1]*Y + coeff[2]

# Fill list with the plane values at the x and y coordinates
Z1 = []
for i in range(len(x)):
    Z1.append(float(coeff[0]*x[i]) + float(coeff[1]*y[i]) + float(coeff[2]))

# Find the distance of each point from the plane
distance = []
for i in range(len(x)):
    distance.append(z[i]-Z1[i])

# x and y values of the points furthest away from the plane
index_min = np.argmin(distance)
index_max = np.argmax(distance)

x_min = x[index_min]
y_min = y[index_min]
x_max = x[index_max]
y_max = y[index_max]

# Create the graph
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2, color='r')

# Title and axis labels
name, ext = os.path.splitext(file_input)
plt.title(f'{name}')
plt.xlabel('X (\u03BCm)')
plt.ylabel('Y (\u03BCm)')
ax.set_zlabel('Z (\u03BCm)')

plt.show()
fig.savefig(f'{name}_3d.png')

# Write data to the file
f = open(f'{name}_Result.txt', 'w')
f.write(f'{name}\n')
f.write('\n')
# Write the x, y and z values used
f.write("%s %s %s\n" % ('x', 'y', 'z'))
for i in range(len(x)):
    f.write("%.2f %.2f %.2f\n" % (x[i], y[i], z[i]))
f.write('\n')
# Write the plane of best fit plotted
f.write(f'Plane fitted: z = {float(coeff[0])}x + {float(coeff[1])}y + {float(coeff[2])}\n')
f.write('\n')
# Points furthest above and below the plane
f.write('Furthest point above the plane:\n')
f.write(f'x: {x_max}um\n')
f.write(f'y: {y_max}um\n')
f.write(f'Distance: {z[index_max]-Z1[index_max]}um\n')
f.write('\n')
f.write('Furthest point below the plane:\n')
f.write(f'x: {x_min}um\n')
f.write(f'y: {y_min}um\n')
f.write(f'Distance: {z[index_min]-Z1[index_min]}um\n')
f.write('\n')
# Write if the sensor is with acceptable measurements
if (abs(z[index_max]-Z1[index_max]) < 12.5) & (abs(z[index_min]-Z1[index_min]) < 12.5):
    f.write(u'This sensor is within the bow measurement limits (\u00B1 12.5um)')
else:
    f.write(u'This sensor is not within the bow measurement limits (\u00B1 12.5um)')
f.close()


# This code plots a projection of the plane and data onto the X-Z plane - it is not required
"""
fig2d = plt.figure()
ax2d = fig2d.add_subplot(111)
ax2d.scatter(x, z, color='r', label='Projection onto XZ axis')
z_line = []
for i in range(len(x)):
    z_line.append(coeff[0]*x[i] + coeff[2])
ax2d.plot(x, z_line, color='r', label = 'Plane of best fit with y=0')
for i, txt in enumerate(y):
    ax2d.annotate(txt, (x[i], z[i]))
plt.title(f'{name}')
plt.xlabel('X (\u03BCm)')
plt.ylabel('Z (\u03BCm)')
plt.legend()
fig2d.savefig(f'{name}_2d.png', bbox_inches='tight')
"""

# This code shows attempts to plot the data projected onto a perpendicular plane - it currently doesn't work
"""
Z_perp = (1/coeff[1])*(coeff[0]*coeff[1]*X + coeff[1]*coeff[2] - (coeff[0]**2 - 1)*Y)
ax.plot_surface(X, Y, Z_perp, rstride=1, cstride=1, alpha=0.2, color='r')
print(coeff[0]**2 - coeff[1]*((coeff[0]**2 + 1)/coeff[1]) +1)
print(f'Plane fitted: z = {float(coeff[0])}x + {float(coeff[1])}y + {float(coeff[2])}')
print(f'Plane fitted: z = {float(coeff[0])}x + {(-1)*(float(coeff[0]**2 + 1)/float(coeff[1]))}y + {float(coeff[2])}')
"""