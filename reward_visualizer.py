"""
Real-time reward component visualization for TruckEnv.
Displays a bar chart of individual reward contributions using OpenCV and matplotlib.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from collections import deque
from datetime import datetime


class RewardVisualizer:
    """
    Visualizes reward components in real-time using OpenCV.
    Creates a bar chart showing contributions from:
    - Goal bonus (when reaching goal)
    - Collision penalty
    - Distance reward
    - Yaw alignment reward
    - Trailer angle alignment reward
    """
    
    def __init__(self, max_history=100, window_name="Reward Visualization"):
        """
        Initialize the reward visualizer.
        
        Args:
            max_history: Maximum number of steps to keep in history for statistics
            window_name: Name of the OpenCV window to display
        """
        self.window_name = window_name
        self.max_history = max_history
        self.window_created = False
        
        # Track individual reward components
        self.reward_components = {
            'Goal Bonus': 0.0,
            'Collision': 0.0,
            'Distance': 0.0,
            'Yaw': 0.0,
            'Trailer Angle': 0.0
        }
        
        # History for trend visualization
        self.history = {key: deque(maxlen=max_history) for key in self.reward_components.keys()}
        self.total_reward_history = deque(maxlen=max_history)
        self.step_count = 0
        
        # Try to create window (may fail in headless environments)
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1000, 600)
            self.window_created = True
        except Exception as e:
            print(f"Warning: Could not create visualization window: {e}")
            self.window_created = False
    
    def update(self, reward_dict, total_reward):
        """
        Update the visualization with new reward values.
        
        Args:
            reward_dict: Dictionary with keys matching self.reward_components
            total_reward: Total reward for this step
        """
        # Update current values
        for key in self.reward_components.keys():
            value = reward_dict.get(key, 0.0)
            self.reward_components[key] = value
            self.history[key].append(value)
        
        self.total_reward_history.append(total_reward)
        self.step_count += 1
        
        # Render visualization
        self._render()
    
    def _render(self):
        """Render the current visualization and display it."""
        # Skip rendering if window was not created successfully
        if not self.window_created:
            return
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Left plot: Current reward components as bar chart
        ax1 = axes[0]
        components = list(self.reward_components.keys())
        values = [self.reward_components[k] for k in components]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax1.bar(range(len(components)), values, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(components)))
        ax1.set_xticklabels(components, rotation=45, ha='right')
        ax1.set_ylabel('Reward Value', fontsize=11, fontweight='bold')
        ax1.set_title('Reward Components (Current Step)', fontsize=12, fontweight='bold')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')
        
        # Right plot: Total reward over time
        ax2 = axes[1]
        if len(self.total_reward_history) > 0:
            ax2.plot(list(self.total_reward_history), 'b-', linewidth=2, label='Total Reward')
            ax2.fill_between(range(len(self.total_reward_history)), 
                            list(self.total_reward_history), alpha=0.3)
            ax2.set_xlabel('Step', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Total Reward', fontsize=11, fontweight='bold')
            ax2.set_title('Cumulative Reward Over Time', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        
        # Convert matplotlib figure to numpy array for OpenCV
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        
        size = canvas.get_width_height()
        img = np.frombuffer(raw_data, dtype=np.uint8)
        img = img.reshape(size[1], size[0], 3)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Display
        cv2.imshow(self.window_name, img_bgr)
        cv2.waitKey(1)
        
        plt.close(fig)
    
    def get_summary(self):
        """
        Get a summary of reward statistics.
        
        Returns:
            Dictionary with statistics about rewards
        """
        summary = {
            'step': self.step_count,
            'current_components': dict(self.reward_components),
            'total_reward_history': list(self.total_reward_history),
        }
        
        if len(self.total_reward_history) > 0:
            summary['avg_total_reward'] = np.mean(list(self.total_reward_history))
            summary['max_total_reward'] = np.max(list(self.total_reward_history))
            summary['min_total_reward'] = np.min(list(self.total_reward_history))
        
        return summary
    
    def close(self):
        """Close the visualization window."""
        if self.window_created:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception as e:
                print(f"Warning: Error closing visualization window: {e}")


class EpisodeRewardTracker:
    """
    Tracks cumulative rewards across an episode and provides statistics.
    """
    
    def __init__(self):
        """Initialize the episode tracker."""
        self.episode_rewards = {
            'Goal Bonus': 0.0,
            'Collision': 0.0,
            'Distance': 0.0,
            'Yaw': 0.0,
            'Trailer Angle': 0.0
        }
        self.total_reward = 0.0
        self.step_count = 0
    
    def add_step(self, reward_dict):
        """
        Add a step's reward contribution to the episode.
        
        Args:
            reward_dict: Dictionary with reward components
        """
        for key in self.episode_rewards.keys():
            value = reward_dict.get(key, 0.0)
            self.episode_rewards[key] += value
            self.total_reward += value
        
        self.step_count += 1
    
    def get_episode_summary(self):
        """
        Get summary statistics for the episode.
        
        Returns:
            Dictionary with episode statistics
        """
        return {
            'total_steps': self.step_count,
            'total_reward': self.total_reward,
            'reward_breakdown': dict(self.episode_rewards),
            'avg_reward_per_step': self.total_reward / self.step_count if self.step_count > 0 else 0.0
        }
    
    def reset(self):
        """Reset the tracker for a new episode."""
        self.episode_rewards = {key: 0.0 for key in self.episode_rewards.keys()}
        self.total_reward = 0.0
        self.step_count = 0
