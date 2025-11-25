import re
import numpy as np
import matplotlib.pyplot as plt

# Function to parse the log file
def parse_log(log):
    steps = []
    avg_rewards = []
    avg_values = []
    
    # Regular expression to match valid lines with step, avg_reward, avg_value
    pattern = r"step=(\d+), avg_reward=([-\d.]+), avg_value=([-\d.]+)"
    
    # Iterate over each line in the log
    for line in log.splitlines():
        match = re.match(pattern, line)
        if match:
            step = int(match.group(1))
            avg_reward = float(match.group(2))
            avg_value = float(match.group(3))
            
            steps.append(step)
            avg_rewards.append(avg_reward)
            avg_values.append(avg_value)
    
    return steps, avg_rewards, avg_values

# Function to smooth data using a simple moving average
def smooth_data(data, window_size=10):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Function to plot the data with smoothed curve and trendline
def plot_data_with_trend(steps, avg_rewards, avg_values):
    # Smooth the data
    smoothed_rewards = smooth_data(avg_rewards)
    smoothed_values = smooth_data(avg_values)
    
    # Fit trendlines using a 1st degree polynomial (linear regression)
    trendline_rewards = np.polyfit(steps[:len(smoothed_rewards)], smoothed_rewards, 1)
    trendline_values = np.polyfit(steps[:len(smoothed_values)], smoothed_values, 1)
    
    # Create trendline functions
    trendline_rewards_func = np.poly1d(trendline_rewards)
    trendline_values_func = np.poly1d(trendline_values)
    
    # Create two separate plots
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot avg_reward with smoothed curve and trendline
    axes[0].plot(steps[:len(smoothed_rewards)], smoothed_rewards, label="Smoothed Avg Reward", color='b', linewidth=1.5)
    axes[0].plot(steps[:len(smoothed_rewards)], trendline_rewards_func(steps[:len(smoothed_rewards)]), label="Trendline (Reward)", color='g', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Training Steps')
    axes[0].set_ylabel('Avg Reward')
    axes[0].set_title('Training Progress: Avg Reward')
    axes[0].grid(True)
    axes[0].legend()

    # Plot avg_value with smoothed curve and trendline
    axes[1].plot(steps[:len(smoothed_values)], smoothed_values, label="Smoothed Avg Value", color='r', linewidth=1.5)
    axes[1].plot(steps[:len(smoothed_values)], trendline_values_func(steps[:len(smoothed_values)]), label="Trendline (Value)", color='purple', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Training Steps')
    axes[1].set_ylabel('Avg Value')
    axes[1].set_title('Training Progress: Avg Value')
    axes[1].grid(True)
    axes[1].legend()

    # Adjust layout for better spacing
    plt.tight_layout()

    # Show the plots
    plt.show()

# Example usage of the function
if __name__ == "__main__":
    # Replace this with the actual path to your log file
    log_file_path = 'logs/training_log_2025-11-23_16-04-30.txt'
    
    with open(log_file_path, 'r') as file:
        log_data = file.read()
    
    # Parse the log
    steps, avg_rewards, avg_values = parse_log(log_data)
    
    # Plot the data with smoothing and trendline
    plot_data_with_trend(steps, avg_rewards, avg_values)
