import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

file = pd.read_csv("1bagfromeduardo/_slash_adma_slash_data_scaled.csv")
data_plot = file[["secs","gnss_pos_rel_y","gnss_pos_rel_x"]]
data_plot.loc[:,"secs"] = data_plot.loc[:,"secs"] - 316782228

grouped =data_plot.groupby(data_plot['secs'].round())

# Calculate the mean (average) X and Y positions within each second
averaged_positions = grouped[['gnss_pos_rel_x', 'gnss_pos_rel_y']].mean()

# Reset the index to turn the grouping results into a new DataFrame
new_df = averaged_positions.reset_index()

# Plotting y against x
plt.figure(figsize=(8, 6))
plt.plot(new_df["gnss_pos_rel_x"].values, new_df["gnss_pos_rel_y"].values, marker='o', linestyle='-')

# Adding labels and title
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Trajectory Plot')
plt.grid(True)
plt.show()

#Video code
def animate(i):
    plt.cla()  # Clear the current plot
    plt.xlim(new_df['gnss_pos_rel_x'].min() - 1, new_df['gnss_pos_rel_x'].max() + 1)  # Adjust x-axis limits if needed
    plt.ylim(new_df['gnss_pos_rel_y'].min() - 1, new_df['gnss_pos_rel_y'].max() + 1)  # Adjust y-axis limits if needed
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Trajectory Evolution')
    print(i)

    # Plotting x and y coordinates at the current time point (frame)
    plt.scatter(new_df['gnss_pos_rel_x'].iloc[:i], new_df['gnss_pos_rel_y'].iloc[:i], color='blue', marker='o')

# Create a figure and animate the plot
fig = plt.figure(figsize=(8, 6))
ani = animation.FuncAnimation(fig, animate, frames=len(new_df), interval=200)  # Change interval as needed (in milliseconds)

# Save the animation as a video file (Optional)
ani.save('trajectory_evolution.mp4', writer='ffmpeg', fps=100, codec='libx264', bitrate=5000)  # Change the filename and fps (frames per second) as needed
plt.show()