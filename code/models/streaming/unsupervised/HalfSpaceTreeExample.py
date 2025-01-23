# Imports

# import numpy
import numpy as np
np.float = float
np.int = np.int32
np.bool = np.bool_


from skmultiflow.data import AnomalySineGenerator
from skmultiflow.anomaly_detection import HalfSpaceTrees



# Setup a data stream
stream = AnomalySineGenerator(random_state=1, n_samples=1000, n_anomalies=250)
# Setup Half-Space Trees estimator
half_space_trees = HalfSpaceTrees(random_state=1)
print(half_space_trees.n_estimators)
half_space_trees.initialise_work_space()
half_space_trees.build_trees()
# Setup variables to control loop and track performance
max_samples = 1000
n_samples = 0
true_positives = 0
detected_anomalies = 0
# Train the estimator(s) with the samples provided by the data stream
while n_samples < max_samples and stream.has_more_samples():
    X, y = stream.next_sample()
    y_pred = half_space_trees.predict(X)
    if y[0] == 1:
        true_positives += 1
        if y_pred[0] == 1:
            detected_anomalies += 1
    half_space_trees.partial_fit(X, y)
    n_samples += 1
print('{} samples analyzed.'.format(n_samples))
print('Half-Space Trees correctly detected {} out of {} anomalies'.
      format(detected_anomalies, true_positives))