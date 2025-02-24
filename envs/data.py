import pandas as pd
import numpy as np


def csv_to_ndarray(csv_file):
    # Read CSV file
    data = pd.read_csv(csv_file)

    # Determine grid size
    max_x = data['x'].max()
    max_y = data['y'].max()
    gridsize = int(max(max_x, max_y) + 0.5)


    array = np.zeros((12, gridsize, gridsize, 2))

    for _, row in data.iterrows():
        frequency = int(row['frequency'])
        x = row['x']
        y = row['y']
        dx = row['dx']
        dy = row['dy']

        # Compute index
        try:
            index_x = int(x - 0.5)
            index_y = int(y - 0.5)
        except ValueError:
            # If the index calculation fails, skip the line
            continue

        # Verify that the index is in a valid range
        if 0 <= index_x < gridsize and 0 <= index_y < gridsize:
            array[frequency, index_x, index_y, 0] = dx
            array[frequency, index_x, index_y, 1] = dy

    return array


# 示例调用
ndarray = csv_to_ndarray('displacement_field50.csv')
print(ndarray.shape)