import numpy as np

def choose_start(start, target):
    """
    生成初始状态点，起点和终点之间的一个区域内，边缘由 edge_ratio 控制
    :param start: 起点坐标 (numpy array)
    :param target: 目标坐标 (numpy array)
    :param edge_ratio: 边缘比，用于扩展区域大小
    :return: 初始状态坐标 (numpy array)
    """

    edge_ratio = 0.2
    # Initialize the random number generator
    rand_x = np.random.rand()
    rand_y = np.random.rand()

    initial_state = []
    # Loop through each dimension
    for i in range(start.shape[0]):
        # Width W in the x direction
        if abs(target[i, 0] - start[i, 0]) > 0:
            W = edge_ratio * abs(target[i, 0] - start[i, 0])
        else:
            # Default value
            W = 0.1

        # Height H in the y direction
        if abs(target[i, 1] - start[i, 1]) > 0:
            H = edge_ratio * abs(target[i, 1] - start[i, 1])
        else:
            H = 0.1

        # Calculate minimum and maximum values for x and y
        x_min = min(start[i, 0], target[i, 0]) - W
        y_min = min(start[i, 1], target[i, 1]) - H
        x_max = max(start[i, 0], target[i, 0]) + W
        y_max = max(start[i, 1], target[i, 1]) + H

        # Generate a random initial state
        initial_state.append([x_min + rand_x * (x_max - x_min),
                              y_min + rand_y * (y_max - y_min)])

    return np.array(initial_state)


def rand_bound(min_val, max_val):
    """生成一个 min 和 max 之间的统一随机数"""
    return min_val + np.random.rand() * (max_val - min_val)


# Test
start = np.array([[0.4, 0.5], [0.5, 0.2]])
target = np.array([[0.6, 0.5], [0.5, 0.4]])


initial_state = choose_start(start, target)
print("Initial State:", initial_state)
