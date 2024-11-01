
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def check_y_tolerance(y_coords, tolerance, min_turbines):
    for i, y in enumerate(y_coords):
        # Count how many other turbines have a y-coordinate within the tolerance range
        close_count = np.sum(np.abs(y_coords - y) <= tolerance)
        
        # Check if the count exceeds or matches the minimum threshold
        if close_count >= min_turbines:
            print(f"At least {min_turbines} turbines have y-coordinates within {tolerance} meters of y = {y:.2f}.")
            return True  # Return if the condition is met


x_values = [    0.      ,    -550.63641309  , 543.08231923  ,  -7.55409386,
  -909.93793497 ,-1460.57434806 , -366.85561573,  -917.49202883,
 -1819.87586994, -2370.51228303, -1276.7935507,  -1827.4299638,
 -2729.8138049,  -3280.450218,   -2186.73148567, -2737.36789876,
 -3639.75173987, -4190.38815296, -3096.66942064, -3647.30583373]
y_values = [ 0.  ,       1018.50983267, 1398.43360557 ,2416.94343824,  353.37480601,
 1371.88463867, 1751.80841158, 2770.31824425,  706.74961201, 1725.25944468,
 2105.18321758, 3123.69305025, 1060.12441802, 2078.63425068, 2458.55802359,
 3477.06785626, 1413.49922402, 2432.00905669, 2811.93282959, 3830.44266226]



# Set the tolerance and the number of turbines you want to check
tolerance = 1  # 1 meter tolerance
nwt_row = 5   # The minimum number of turbines with similar y-coordinates

# Define the rotation angle (in degrees)
for theta in np.arange(0,45,0.1):
    theta_degrees = theta  # Example: rotate by 45 degrees
    theta = np.radians(theta_degrees)  # Convert to radians

    # Define the rotation matrix
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # Sample turbine coordinates (x, y)
    turbine_coords = np.array(list(zip(x_values, y_values)))

    turbine_coords = np.dot(turbine_coords, rotation_matrix.T)

    if check_y_tolerance(turbine_coords[:,1],tolerance,nwt_row):
        break


print(turbine_coords)

# Step 1: Group turbines based on proximity (simplified row grouping by y-coordinates)
def group_turbines_by_proximity(turbine_coords, threshold=100):
    rows = []
    sorted_coords = sorted(turbine_coords, key=lambda coord: coord[1])  # Sort by y-coordinate
    
    current_row = [sorted_coords[0]]
    for coord in sorted_coords[1:]:
        if abs(coord[1] - current_row[-1][1]) <= threshold:
            current_row.append(coord)
        else:
            rows.append(current_row)
            current_row = [coord]
    rows.append(current_row)
    
    return rows

# Step 2: Calculate the minimum cabling length to connect turbines within each row (Euclidean distance)
def connect_turbines_in_row(row):
    total_length = 0
    for i in range(1, len(row)):
        # Use Euclidean distance between consecutive turbines
        total_length += np.linalg.norm(np.array(row[i]) - np.array(row[i-1]))
    return total_length

# Step 3: Connect the rows by finding the minimum cable length between turbines in adjacent rows
def connect_rows(rows):
    total_length = 0
    shortest_connections=[]

    for i in range(1, len(rows)):
        row1 = rows[i-1]
        row2 = rows[i]
        
        # Calculate the distance matrix between all turbines in row1 and row2
        distances = distance_matrix(row1, row2)
        
        # Find the minimum distance between turbines in adjacent rows
        # Find the minimum distance between turbines in adjacent rows
        min_idx = np.unravel_index(np.argmin(distances), distances.shape)  # Index of the minimum distance
        min_length = np.min(distances)
        total_length += min_length

        # Store the coordinates of the closest turbines for the plot
        shortest_connections.append((row1[min_idx[0]], row2[min_idx[1]]))


    
    return total_length , shortest_connections

# Step 4: Calculate the total cabling length
def calculate_total_cable_length(turbine_coords):
    # Group turbines by proximity (you can adjust the threshold based on your layout)
    rows = group_turbines_by_proximity(turbine_coords, threshold=100)
    
    # Calculate the cable length for turbines within each row (using Euclidean distances)
    intra_row_length = sum(connect_turbines_in_row(row) for row in rows)
    
    # Calculate the cable length to connect the rows
    inter_row_length , shortest_connections = connect_rows(rows)
    
    return intra_row_length + inter_row_length , shortest_connections

# Run the function on the turbine coordinates
total_cable_length , shortest_connections= calculate_total_cable_length(turbine_coords)
print(f"Total cabling length: {total_cable_length:.2f} meters")

# Optional: Plot the turbine layout and connections
plt.figure(figsize=(8, 6))
for coord in turbine_coords:
    plt.plot(coord[0], coord[1], 'bo')  # Turbine locations

# Connect turbines row-wise and plot
rows = group_turbines_by_proximity(turbine_coords, threshold=100)
for row in rows:
    row = np.array(row)
    plt.plot(row[:, 0], row[:, 1], 'r-')  # Row connections

# Plot the shortest inter-row connections
for conn in shortest_connections:
    row1_turbine, row2_turbine = conn
    plt.plot([row1_turbine[0], row2_turbine[0]], [row1_turbine[1], row2_turbine[1]], 'r-', linewidth=2 )



plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Turbine Layout and Cabling')
plt.grid(True)
plt.show()

# %%
