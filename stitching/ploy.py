import matplotlib.pyplot as plt

# Data based on the output you provided
resolutions = ['224x224', '256x256', '512x512', '1024x1024']
times = [0.1496, 0.0275, 0.0341, 0.1033]

# Create the plot
plt.figure(figsize=(8, 6))
plt.plot(resolutions, times, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)

# Labeling the plot
plt.xlabel('Image Resolution', fontsize=12)
plt.ylabel('Computation Time (seconds)', fontsize=12)
plt.title('Computation Time vs Image Resolution', fontsize=14)
plt.grid(True)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Display the plot
plt.tight_layout()  # Adjust layout to avoid overlap of labels
plt.show()
