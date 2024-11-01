# %%

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

# Function to check y-coordinate alignment with tolerance
def check_y_tolerance(y_coords, tolerance, min_turbines):
    for i, y in enumerate(y_coords):
        # Count how many other turbines have a y-coordinate within the tolerance range
        close_count = np.sum(np.abs(y_coords - y) <= tolerance)
        if close_count >= min_turbines:
            print(f"At least {min_turbines} turbines have y-coordinates within {tolerance} meters of y = {y:.2f}.")
            return True
    return False

# Define the x and y coordinates for turbine positions
x_values = [0, -550.63641309, 543.08231923, -7.55409386, -909.93793497,
            -1460.57434806, -366.85561573, -917.49202883, -1819.87586994,
            -2370.51228303, -1276.7935507, -1827.4299638, -2729.8138049,
            -3280.450218, -2186.73148567, -2737.36789876, -3639.75173987,
            -4190.38815296, -3096.66942064, -3647.30583373]
y_values = [0, 1018.50983267, 1398.43360557, 2416.94343824, 353.37480601,
            1371.88463867, 1751.80841158, 2770.31824425, 706.74961201,
            1725.25944468, 2105.18321758, 3123.69305025, 1060.12441802,
            2078.63425068, 2458.55802359, 3477.06785626, 1413.49922402,
            2432.00905669, 2811.93282959, 3830.44266226]
turbine_coords = np.array(list(zip(x_values, y_values)))

# Set tolerance and minimum turbines per row
tolerance = 1  # 1 meter tolerance
nwt_row = 5    # Minimum number of turbines with similar y-coordinates

# Rotate the coordinates and check for y-alignment
for theta_degrees in np.arange(0, 45, 0.1):  # Rotate from 0 to 45 degrees in 0.1 increments
    theta = np.radians(theta_degrees)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    turbine_coords_rotated = np.dot(turbine_coords, rotation_matrix.T)
    if check_y_tolerance(turbine_coords_rotated[:, 1], tolerance, nwt_row):
        break

# Group turbines by proximity for cabling
def group_turbines_by_proximity(turbine_coords, threshold=100):
    rows = []
    sorted_coords = sorted(turbine_coords, key=lambda coord: coord[1])
    current_row = [sorted_coords[0]]
    for coord in sorted_coords[1:]:
        if abs(coord[1] - current_row[-1][1]) <= threshold:
            current_row.append(coord)
        else:
            rows.append(current_row)
            current_row = [coord]
    rows.append(current_row)
    return rows

# Calculate intra-row cable length
def connect_turbines_in_row(row):
    return sum(np.linalg.norm(np.array(row[i]) - np.array(row[i-1])) for i in range(1, len(row)))

# Calculate inter-row connections
def connect_rows(rows):
    total_length = 0
    shortest_connections = []
    for i in range(1, len(rows)):
        row1, row2 = rows[i-1], rows[i]
        distances = distance_matrix(row1, row2)
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)
        min_length = distances[min_idx]
        total_length += min_length
        shortest_connections.append((row1[min_idx[0]], row2[min_idx[1]]))
    return total_length, shortest_connections

# Calculate total cabling length including intra- and inter-row connections
def calculate_total_cable_length(turbine_coords):
    rows = group_turbines_by_proximity(turbine_coords, threshold=100)
    intra_row_length = sum(connect_turbines_in_row(row) for row in rows)
    inter_row_length, shortest_connections = connect_rows(rows)
    return intra_row_length + inter_row_length, shortest_connections

# Calculate total cabling length in rotated frame
total_cable_length, shortest_connections = calculate_total_cable_length(turbine_coords_rotated)

# Apply inverse rotation to return to original coordinate system
inverse_theta = np.radians(-theta_degrees)
inverse_rotation_matrix = np.array([[np.cos(inverse_theta), -np.sin(inverse_theta)], 
                                    [np.sin(inverse_theta), np.cos(inverse_theta)]])

# Apply inverse rotation to turbines and cables
turbine_coords_original = np.dot(turbine_coords_rotated, inverse_rotation_matrix.T)
row_connections_original = []
for row in group_turbines_by_proximity(turbine_coords_rotated, threshold=100):
    for i in range(len(row) - 1):
        row_connections_original.append((np.dot(row[i], inverse_rotation_matrix.T), np.dot(row[i + 1], inverse_rotation_matrix.T)))
inter_row_connections_original = [(np.dot(p1, inverse_rotation_matrix.T), np.dot(p2, inverse_rotation_matrix.T)) for p1, p2 in shortest_connections]

# Plot the cabling layout in the original coordinate system
plt.figure(figsize=(8, 6))
plt.plot(turbine_coords_original[:, 0], turbine_coords_original[:, 1], 'bo', label='Turbines')  # Turbine locations

# Plot intra-row and inter-row connections in original coordinates
for p1, p2 in row_connections_original:
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-')
for p1, p2 in inter_row_connections_original:
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)

# Final plot settings
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Turbine Layout and Minimal Cabling Connection Route')
plt.legend()
plt.grid(True)
plt.show()
