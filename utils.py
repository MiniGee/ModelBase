
# Calculate average over data of certain sample size
class MovingAvg():

    def __init__(self, window):
        self._window = window
        self._num_values = 0

        self._data = [0.0] * window
        self._current_idx = 0


    # Add value to data
    def add(self, x):
        self._data[self._current_idx] = x

        # Increment index
        self._current_idx += 1
        if self._current_idx >= self._window:
            self._current_idx = 0

        if self._num_values < self._window:
            self._num_values += 1


    # Calculate average
    def value(self):
        sum = 0.0

        for i in range(self._num_values):
            sum += self._data[i]

        return sum / self._num_values