"""
GVR Peak Analysis Module
This module extracts and visualizes the peak detection functionality from read_data.py
for easier debugging and troubleshooting.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy.signal import find_peaks
import os


def analyze_gvr_peaks(damaged_signal: np.ndarray, 
                      healthy_signal: np.ndarray,
                      dt: float = 0.001,
                      window_length: int = 3000,
                      step_size: int = 50,
                      cutoff_freq: float = 5.0,
                      prob_threshold: float = 5.0,
                      visualize: bool = True,
                      output_dir: str = './gvr_analysis_output'):
    """
    Analyze GVR peaks for damage detection
    
    Args:
        damaged_signal: Acceleration signal from damaged structure
        healthy_signal: Acceleration signal from healthy structure
        dt: Time step
        window_length: Length of analysis window
        step_size: Step size between windows
        cutoff_freq: Low-pass filter cutoff frequency
        prob_threshold: Probability threshold for damage classification
        visualize: Whether to generate visualization plots
        output_dir: Directory to save plots
    
    Returns:
        tuple: (auto_labels, probabilities, DI_double_prime, analysis_data)
    """
    
    # Create output directory if needed
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize filter
    nyquist = 0.5 / dt
    b, a = signal.butter(4, cutoff_freq / nyquist, btype='low')
    
    # 1. Preprocessing: filter signals
    filtered_damaged = signal.filtfilt(b, a, damaged_signal, axis=0)
    filtered_healthy = signal.filtfilt(b, a, healthy_signal, axis=0)
    
    n_channels = damaged_signal.shape[1]
    num_windows = (filtered_damaged.shape[0] - window_length) // step_size + 1
    
    # 2. Calculate DI_series (must be computed in loop)
    DI_series = np.zeros((num_windows, n_channels))
    for win_idx in range(num_windows):
        start = win_idx * step_size
        end = start + window_length
        
        win_damaged = filtered_damaged[start:end]
        win_healthy = filtered_healthy[start:end]
        
        # Paper formula (8)
        for ch in range(n_channels):
            numerator = np.sum((win_damaged[:, ch] - win_healthy[:, ch]) ** 2)
            denominator = np.sum(win_healthy[:, ch] ** 2) + 1e-10
            DI_series[win_idx, ch] = np.sqrt(numerator) / np.sqrt(denominator)

    # Spatial first derivative: calculate difference between adjacent sensors' DI
    # Logic: DI[i] - DI[i-1]
    DI_prime = np.zeros_like(DI_series)
    DI_prime[:, 1:] = DI_series[:, 1:] - DI_series[:, :-1]
    
    # Spatial second derivative: calculate rate of change of spatial gradient (detects peaks)
    # Logic: abs((DI[i]-DI[i-1]) - (DI[i-1]-DI[i-2]))
    DI_double_prime = np.zeros_like(DI_prime)
    # Note: After taking first derivative then second derivative, effective length is (n_channels - 2)
    DI_double_prime[:, 1:] = np.abs(DI_prime[:, 1:] - DI_prime[:, :-1])
    
    # 4. Count fault occurrences across channels
    fault_occurrences = np.zeros(n_channels)
    
    for win_idx in range(num_windows):
        # Get spatial GVR distribution for current window
        spatial_gvr = DI_double_prime[win_idx]
        
        if np.max(spatial_gvr) > 1e-8:
            prominence_threshold = np.max(spatial_gvr) * 0.1
        else:
            prominence_threshold = 0
        
        # 2. Find all peaks that meet conditions
        # distance=2: Prevents adjacent sensors (like 4 and 5) from being identified as two separate damage points
        peaks, properties = find_peaks(
            spatial_gvr, 
            prominence=prominence_threshold, 
            distance=2,
        ) # e.g., above mean+2*std
        
        # 3. Count
        for ch in peaks:
            fault_occurrences[ch] += 1
    
    # 5. Calculate damage probability
    probabilities = (fault_occurrences / num_windows) * 100
    
    # 6. Generate labels based on probability threshold
    auto_labels = (probabilities > prob_threshold).astype(int)
    
    # Visualization if requested
    if visualize:
        visualize_gvr_analysis(
            DI_series, DI_prime, DI_double_prime, 
            fault_occurrences, probabilities, auto_labels,
            num_windows, output_dir
        )
    
    analysis_data = {
        'DI_series': DI_series,
        'DI_prime': DI_prime,
        'DI_double_prime': DI_double_prime,
        'fault_occurrences': fault_occurrences,
        'num_windows': num_windows,
        'prominence_threshold': prominence_threshold if 'prominence_threshold' in locals() else 0
    }
    
    return auto_labels, probabilities, DI_double_prime, analysis_data


def visualize_gvr_analysis(DI_series, DI_prime, DI_double_prime, 
                          fault_occurrences, probabilities, auto_labels,
                          num_windows, output_dir):
    """
    Visualize GVR analysis results
    """
    n_channels = DI_series.shape[1]
    
    # Plot 1: DI Series over time for all channels
    plt.figure(figsize=(15, 10))
    for ch in range(min(15, n_channels)):  # Only plot first 15 channels to avoid overcrowding
        plt.subplot(5, 3, ch+1)
        plt.plot(DI_series[:, ch], alpha=0.7, label=f'Channel {ch+1}')
        plt.title(f'DI Series - Channel {ch+1}')
        plt.xlabel('Window Index')
        plt.ylabel('DI Value')
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'di_series_all_channels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: DI Prime (first derivative) over time
    plt.figure(figsize=(15, 10))
    for ch in range(min(15, n_channels)):
        plt.subplot(5, 3, ch+1)
        plt.plot(DI_prime[:, ch], alpha=0.7, color='orange', label=f'Channel {ch+1}')
        plt.title(f'DI Prime - Channel {ch+1}')
        plt.xlabel('Window Index')
        plt.ylabel("DI'")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'di_prime_all_channels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: DI Double Prime (second derivative) over time
    plt.figure(figsize=(15, 10))
    for ch in range(min(15, n_channels)):
        plt.subplot(5, 3, ch+1)
        plt.plot(DI_double_prime[:, ch], alpha=0.7, color='red', label=f'Channel {ch+1}')
        plt.title(f'DI Double Prime - Channel {ch+1}')
        plt.xlabel('Window Index')
        plt.ylabel("DI''")
        plt.grid(True, alpha=0.3)
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'di_double_prime_all_channels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Summary statistics
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Fault occurrences per channel
    axes[0, 0].bar(range(len(fault_occurrences)), fault_occurrences)
    axes[0, 0].set_title('Fault Occurrences Per Channel')
    axes[0, 0].set_xlabel('Channel')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Damage probabilities per channel
    axes[0, 1].bar(range(len(probabilities)), probabilities)
    axes[0, 1].set_title('Damage Probability Per Channel (%)')
    axes[0, 1].set_xlabel('Channel')
    axes[0, 1].set_ylabel('Probability (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Auto labels
    colors = ['green' if label == 0 else 'red' for label in auto_labels]
    axes[1, 0].bar(range(len(auto_labels)), auto_labels, color=colors)
    axes[1, 0].set_title('Auto Labels (0=Healthy, 1=Damaged)')
    axes[1, 0].set_xlabel('Channel')
    axes[1, 0].set_ylabel('Label')
    axes[1, 0].set_ylim(-0.1, 1.1)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Overall statistics
    stats_text = f"""Statistics:
Total Windows: {num_windows}
Channels: {len(fault_occurrences)}
Avg Fault Occurrences: {np.mean(fault_occurrences):.2f}
Avg Probability: {np.mean(probabilities):.2f}%
Damaged Channels: {np.sum(auto_labels)} / {len(auto_labels)}
"""
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_statistics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 5: Heatmap of DI Double Prime
    plt.figure(figsize=(12, 8))
    im = plt.imshow(DI_double_prime.T, aspect='auto', origin='lower', 
                    extent=[0, num_windows, 0, n_channels], cmap='viridis')
    plt.colorbar(im, label="DI'' Value")
    plt.title('Heatmap of DI Double Prime Over Time and Channels')
    plt.xlabel('Window Index')
    plt.ylabel('Channel')
    plt.savefig(os.path.join(output_dir, 'di_double_prime_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def test_gvr_peak_analysis():
    """
    Test function with sample data
    """
    print("Testing GVR Peak Analysis...")
    
    # Generate sample data
    np.random.seed(42)
    n_steps = 30000
    n_channels = 15
    
    # Healthy signal: random noise
    healthy_signal = np.random.normal(0, 0.1, (n_steps, n_channels))
    
    # Damaged signal: similar to healthy but with some differences in specific channels
    damaged_signal = healthy_signal.copy()
    # Add slight variations in channels 5 and 10 to simulate damage
    damaged_signal[:, 5] += np.random.normal(0.05, 0.05, n_steps)
    damaged_signal[:, 10] += np.random.normal(0.08, 0.08, n_steps)
    
    # Run analysis
    labels, probabilities, DI_double_prime, analysis_data = analyze_gvr_peaks(
        damaged_signal, healthy_signal, visualize=True
    )
    
    print(f"Analysis Results:")
    print(f"Auto Labels: {labels}")
    print(f"Probabilities (%): {probabilities}")
    print(f"Shape of DI_double_prime: {DI_double_prime.shape}")
    print(f"Number of damaged channels detected: {np.sum(labels)}")
    
    return labels, probabilities, DI_double_prime, analysis_data


if __name__ == "__main__":
    # Run test
    test_gvr_peak_analysis()
    print("GVR Peak Analysis module created successfully!")
    print("Output plots saved in ./gvr_analysis_output/")