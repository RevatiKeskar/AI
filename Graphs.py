import matplotlib.pyplot as plt
import numpy as np
import time
# Assume results contain data from multiple runs
results = [
    {'heuristic': 'manhattan', 'moves': 15, 'time': 0.1},
    {'heuristic': 'linear_conflict', 'moves': 15, 'time': 0.1},
    {'heuristic': 'misplaced', 'moves': 15, 'time': 0.3},
    # Additional data for multiple runs...
]
# Data extraction
heuristics = [res['heuristic'] for res in results]
move_counts = [res['moves'] for res in results]
times = [res['time'] for res in results]
# Plotting Number of Moves
plt.figure(figsize=(14, 7))
plt.subplot(2, 2, 1)
plt.bar(heuristics, move_counts, color=['blue', 'green', 'orange'])
plt.title('Number of Moves for Each Heuristic')
plt.xlabel('Heuristic Type')
plt.ylabel('Number of Moves')
# Plotting Computation Time
plt.subplot(2, 2, 2)
plt.bar(heuristics, times, color=['blue', 'green', 'orange'])
plt.title('Computation Time for Each Heuristic')
plt.xlabel('Heuristic Type')
plt.ylabel('Time (seconds)')
# Plotting Heuristic Efficiency
plt.subplot(2, 2, 3)
plt.scatter(times, move_counts, c=['blue', 'green', 'orange'], s=100, alpha=0.6)
for i, txt in enumerate(heuristics):
    plt.annotate(txt, (times[i], move_counts[i]))
plt.title('Heuristic Efficiency')
plt.xlabel('Computation Time (seconds)')
plt.ylabel('Number of Moves')

# Show the plots
plt.tight_layout()
plt.show()