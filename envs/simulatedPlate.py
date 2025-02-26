import numpy as np
import scipy


class SimulatedPlate:
    def __init__(self, data):
        # Define current location and set initial location
        self.p = None
        self.target_p = np.array([[17.4, 20.5]])
        self.fields = data['mapGrid'][0]
        self.length = self.fields[0][0].shape[0]
        self.width = self.fields[0][0].shape[1]
        self.state_dimension = self.target_p.shape[0] * self.target_p.shape[1]
        self.action_size = self.fields.shape[0]
        self.action_space = None
        self.out = False
        self.arrive = False
        self.randomness = 0.1

    def set_positions(self, current_position):
        # Set current location
        self.p = np.array(current_position)

    def get_positions(self):
        return self.p

    def reset(self):
        # Reset task including done flag and initial points
        self.out = False
        self.arrive = False
        # n = self.target_p.shape[0]
        # self.p = np.column_stack((
        #     np.random.uniform(0, self.length, size=n),
        #     np.random.uniform(0, self.width, size=n)
        # ))
        self.p = np.array([[2.4, 1.5]])
        return self.p

    def reward_function(self, p_before, dp, target):
        """
        Calculate reward value

        :param p_before: The position of the particles at the last moment, shape (num_particles, 2)
        :param dp: The change of the position, shape (num_particles, 2)
        :param target: Target location, shape (num_targets, 2)
        :return: reward
        """

        # Check if the particle is out of bounds
        # If out of bounds, set a high negative reward
        if np.min(self.p) < 0 or np.max(self.p) > self.length:
            self.out = True
            return -1
        
        # The distance between the current particle and the target point
        dist = target - p_before

        # Calculate the projected proportion of dp on dist
        projection_ratios = np.sum(dp * dist, axis=1) / np.sum(dist ** 2, axis=1)

        # Gets the sign of the projection
        projection_signs = np.sign(projection_ratios)

        # Calculate the projected vector
        projection_vectors = (projection_ratios[:, np.newaxis] * dist)

        # Calculate the length of the projection vector
        projection_lengths_abs = np.linalg.norm(projection_vectors, axis=1)
        projection_lengths = projection_lengths_abs * projection_signs

        # reward = np.min(projection_lengths)
        reward = float(np.sum(projection_lengths))

        # The distance from the current particle to the target
        t = np.linalg.norm(self.p - target, axis=1)

        # If the distance is less than a certain threshold, set a high reward and terminate
        if np.all(t < 1):
            reward = 5
            self.arrive = True
            print("The particle moves to the target points")

        return reward


    # Step(Action)
    def play(self, frequency_id):
        """
        Simulate playback function, update particle position
        :param frequency_id: Index of the current frequency
        """
        if frequency_id < 0 or frequency_id >= len(self.fields):
            raise ValueError(f"Invalid frequencyId, should be between 1..{len(self.fields)} (was: {frequency_id})")

        # Get the variance from data based on the frequency ID and calculate the randomness
        # variance = self.fields[frequency_id]['variance'][tuple(self.p.astype(int))]
        # randomness = np.sqrt(max(variance, 0))

        # Get deltaX and deltaY
        p_index = self.p.astype(int)
        dx = self.fields[frequency_id][0][p_index[:, 0], p_index[:, 1]] * 100
        dy = self.fields[frequency_id][1][p_index[:, 0], p_index[:, 1]] * 100

        # Generate noise based on standard normal distribution
        noise = self.randomness * np.random.randn(p_index.shape[0], p_index.shape[1])

        # Update particle position
        p_before = self.p
        dx_dy = np.column_stack((dx, dy))
        delta_p = dx_dy + dx_dy * noise
        # self.p = self.p + delta_p + noise
        self.p = self.p + delta_p

        # Calculate reward value
        reward = self.reward_function(p_before, delta_p, self.target_p)

        return self.p, reward, self.out | self.arrive, self.arrive


if __name__ == "__main__":
    # Test
    # data = [
    #     {
    #         'variance': np.random.rand(10, 10),  # 模拟方差图
    #         'deltaX': np.random.rand(10, 10),    # 模拟x方向的位移图
    #         'deltaY': np.random.rand(10, 10)     # 模拟y方向的位移图
    #     },
    #     {
    #         'variance': np.random.rand(10, 10),  # 模拟方差图
    #         'deltaX': np.random.rand(10, 10),  # 模拟x方向的位移图
    #         'deltaY': np.random.rand(10, 10)  # 模拟y方向的位移图
    #     }
    # ]

    # Create a SimulatedPlate object
    data = scipy.io.loadmat('vectorField_RL_2019_P2.mat')
    plate = SimulatedPlate(data)

    plate.reset()

    # Set initial position
    # start = [[0.4, 0.5], [4.5, 5.2]]
    # plate.set_positions(start)

    for i in range(2):
        # Get current location
        print("Initial position:", plate.get_positions())

        # Play the motion at a certain frequency
        plate.play(i)

        # Get the current location after the update
        print("New position after play:", plate.get_positions())
