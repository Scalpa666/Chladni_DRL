import pandas as pd
import numpy as np


# Simulated plate with definitive target point position
class SimulatedPlateD:
    def __init__(self, csv_file):
        self.length = None
        self.width = None
        self.action_size = None
        self.displacement_fields = self.load_data(csv_file)

        self.p = None
        self.target_p = None
        self.state_dimension = None
        self.terminate = False
        self.arrive = False
        self.randomness = 0.1

    def load_data(self, csv_file):
        # Read CSV file
        data = pd.read_csv(csv_file)

        # Determine the grid size and frequency class number, which also corresponds to the action size.
        max_x = data['x'].max()
        max_y = data['y'].max()
        max_freq = data['frequency'].max()
        self.length = int(max_x + 0.5)
        self.width = int(max_y + 0.5)
        self.action_size = int(max_freq + 1)

        array = np.zeros((self.action_size, self.length, self.width, 2))

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
            if 0 <= index_x < self.length and 0 <= index_y < self.width:
                array[frequency, index_x, index_y, 0] = dx
                array[frequency, index_x, index_y, 1] = dy

        return array

    def set_positions(self, current_position=None):
        # Set current location
        # self.p = np.array(current_position)
        self.p = np.random.rand(2)

    def set_target_positions(self, target_position):
        # Set target location
        self.target_p = np.array(target_position)

    def get_positions(self):
        return self.p

    def reset(self):
        # Reset task including done flag and initial points
        self.terminate = False
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

    # Play(Action)
    def step(self, frequency_id):
        """
        Simulate playback function, update particle position
        :param frequency_id: Index of the current frequency
        """
        if frequency_id < 0 or frequency_id >= self.action_size:
            raise ValueError(f"Invalid frequencyId, should be between 0..{self.action_size - 1} (was: {frequency_id})")

        # Get the variance from data based on the frequency ID and calculate the randomness
        # variance = self.fields[frequency_id]['variance'][tuple(self.p.astype(int))]
        # randomness = np.sqrt(max(variance, 0))

        # Get deltaX and deltaY
        p_index = self.p.astype(int)
        dx = self.displacement_fields[frequency_id][p_index[:, 0], p_index[:, 1]][0] * 100
        dy = self.displacement_fields[frequency_id][p_index[:, 0], p_index[:, 1]][1] * 100

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

        return self.p, reward, self.arrive, self.out | self.arrive


# Example usage
if __name__ == "__main__":
    # Create a SimulatedPlateM instance
    plate = SimulatedPlateD("displacement_field50.csv")

    # Print the shape of data
    print(plate.displacement_fields.shape)

    plate.reset()
    plate.set_positions()
    plate.set_target_positions([5.0, 6.7])
