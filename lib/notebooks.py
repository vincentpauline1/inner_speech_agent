import ipywidgets as ipw
import matplotlib.pyplot as plt
import numpy as np


def plot_groups_metrics(groups_metrics, metrics_name, split_name="test"):
    """Plot training/validation metrics for multiple model groups.
    
    Creates an organized grid visualization comparing metrics across model groups:
    - Each row represents a different model group
    - Each column shows a different metric (e.g. loss, accuracy)
    - Within each plot, multiple lines show different model runs
    - Final values are highlighted with markers
    - Clear labels and consistent styling for easy interpretation
    
    Args:
        groups_metrics (dict): Nested dictionary containing metrics for each group and model
        metrics_name (list): List of metric names to plot (e.g. ['loss', 'accuracy'])
        split_name (str, optional): Data split to plot ('train', 'val', 'test'). Defaults to "test"
    """
    nb_groups = len(groups_metrics)
    nb_metrics = len(metrics_name)
    groups_name = sorted(groups_metrics.keys())

    # Create figure with optimized dimensions and higher DPI for clarity
    plt.figure(figsize=(nb_metrics * 5 + 3, nb_groups * 4.5), dpi=120)
    
    # Define consistent colors and styles
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i_group, group_name in enumerate(groups_name):
        group_metrics = groups_metrics[group_name]

        for i_metric, metric_name in enumerate(metrics_name):
            ax = plt.subplot(nb_groups, nb_metrics, 1 + nb_metrics * i_group + i_metric)
            
            # Add styled group name label
            if i_metric == 0:
                ax.text(
                    -0.45, 0.5,
                    group_name,
                    verticalalignment="center",
                    horizontalalignment="right",
                    transform=ax.transAxes,
                    fontsize=13,
                    fontweight='bold'
                )
            
            # Plot metric curves with consistent styling
            for idx, (model_name, metrics) in enumerate(group_metrics.items()):
                metric_values = metrics[split_name][metric_name]
                color = colors[idx % len(colors)]
                
                # Plot line with custom styling
                line = ax.plot(metric_values, 
                             label=model_name,
                             color=color,
                             linewidth=2,
                             alpha=0.8)[0]
                
                # Add emphasized final value marker
                last_epoch = len(metric_values) - 1
                last_value = metric_values[-1]
                ax.scatter(last_epoch, last_value,
                          color=color,
                          s=120,
                          zorder=10,
                          edgecolor='white',
                          linewidth=1.5)
                
                # Annotate final value
                ax.annotate(f'{last_value:.3f}',
                          (last_epoch, last_value),
                          xytext=(7, 7),
                          textcoords='offset points',
                          fontsize=10)
            
            # Style the subplot
            ax.set_title(f"{metric_name.capitalize()} Over Epochs", 
                        fontsize=14, 
                        pad=15)
            ax.set_xlabel("Epoch", fontsize=12)
            ax.set_ylabel(metric_name.capitalize(), fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(labelsize=10)
            
            # Add legend at the top of the first subplot in each row
            if i_metric == 0:
                ax.legend(bbox_to_anchor=(0, 1.25),
                         loc='lower left',
                         borderaxespad=0.,
                         ncol=3,
                         fontsize=11)
            
    plt.tight_layout(h_pad=2.5, w_pad=1.5)
    plt.show()

def show_ema(ema, reference=None, dataset=None):
    """Interactive visualization of electromagnetic articulography (EMA) data.
    
    Provides an interactive plot to explore articulator trajectories:
    - Real-time scatter plot of articulator positions 
    - Optional palate trace for anatomical context
    - Configurable motion trails to show temporal dynamics
    - Optional reference positions for comparison
    
    Args:
        ema (ndarray): EMA trajectories, shape (frames, features) with alternating x,y coords
        reference (ndarray, optional): Reference EMA data for comparison
        dataset (Dataset, optional): Contains palate trace and coordinate bounds
    """
    nb_frames = len(ema)
    ema_x, ema_y = ema[:, 0::2], ema[:, 1::2]
    
    # Determine plot boundaries from dataset or data range
    if dataset is not None:
        xlim = (dataset.ema_limits["xmin"] * 0.95, dataset.ema_limits["xmax"] * 1.05)
        ylim = (dataset.ema_limits["ymin"] * 0.95, dataset.ema_limits["ymax"] * 1.05)
    else:
        xlim = (ema_x.min() * 0.95, ema_x.max() * 1.05)
        ylim = (ema_y.min() * 0.95, ema_y.max() * 1.05)

    def show_ema_frame(i_frame=0, trail_len=20, show_reference=True, show_reference_trail=False):
        """Render a single EMA frame with configurable visualization options.
        
        Args:
            i_frame (int): Current frame index
            trail_len (int): Number of previous frames in motion trail
            show_reference (bool): Toggle reference position display
            show_reference_trail (bool): Toggle reference trail display
        """
        # Generate opacity gradient for motion trails
        trail_opacity = np.linspace(0, 0.5, trail_len)
        trail_len = min(i_frame, trail_len - 1)
        trail_opacity = trail_opacity[-trail_len:]
        trail_start = i_frame - trail_len

        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot anatomical reference if available
        if dataset is not None and dataset.has_palate:
            ax.plot(dataset.palate[:, 0], dataset.palate[:, 1], 'k-', 
                   label='Palate', linewidth=2)
            
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(f"Articulator Positions - Frame {i_frame}", pad=10)
        ax.set_xlabel("X Position (mm)", labelpad=8)
        ax.set_ylabel("Y Position (mm)", labelpad=8)
        
        # Plot current positions and motion trails
        ax.scatter(ema_x[i_frame], ema_y[i_frame], c="tab:blue", s=80, 
                  label="Current Position")
        if trail_len > 0:
            ax.scatter(ema_x[trail_start:i_frame], ema_y[trail_start:i_frame],
                alpha=trail_opacity, c="tab:blue", s=40, label="Motion Trail")
            
        # Add reference visualization if available
        if show_reference and reference is not None:
            ax.scatter(reference[i_frame, 0::2], reference[i_frame, 1::2],
                      c="tab:red", s=80, label="Reference Position")
            if show_reference_trail:
                ax.scatter(reference[trail_start:i_frame, 0::2],
                    reference[trail_start:i_frame, 1::2],
                    alpha=trail_opacity, c="tab:red", s=40,
                    label="Reference Trail"
                )
                
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    # Create interactive controls
    ipw.interact(
        show_ema_frame,
        i_frame=(0, nb_frames - 1),
        trail_len=(0, 50, 1),
        show_reference=True,
        show_reference_trail=False,
    )


def show_occlusions_metrics(phones_metrics, palate):
    """Interactive visualization of articulatory metrics for phonetic analysis.
    
    Generates detailed plots for analyzing articulator behavior:
    - Spatial distribution of articulator positions
    - Statistical distribution of distances
    - Separate analysis for lips, tongue tip, and tongue mid
    
    Args:
        phones_metrics (dict): Metrics for each phone category
        palate (ndarray): Palate contour coordinates
    """
    distances = ["lips", "tongue_tip", "tongue_mid"]

    def show_phone_metrics(phone):
        """Display comprehensive metrics for a selected phone.
        
        Args:
            phone (str): Target phone for analysis
        """
        consonant_metrics = phones_metrics[phone]
        
        fig = plt.figure(figsize=(12, 10), dpi=120)
        
        for i_distance, distance in enumerate(distances):
            distance_ema = np.array(consonant_metrics[f"min_{distance}_ema"])
            distances_array = consonant_metrics[f"min_{distance}_distance"]
            
            # Print summary statistics
            print(f"\n{distance.replace('_', ' ').title()} metrics for '{phone}':")
            print(f"  Mean distance: {np.mean(distances_array):.2f} mm")
            print(f"  Std deviation: {np.std(distances_array):.2f} mm")
            print(f"  Min/Max: {np.min(distances_array):.2f}/{np.max(distances_array):.2f} mm")
            
            # Plot articulator trajectories
            ax = plt.subplot(3, 2, 1 + 2 * i_distance, aspect="equal")
            ax.plot(palate[:, 0], palate[:, 1], 'k-', label='Palate', linewidth=2)
            ax.scatter(distance_ema[:, 0::2], distance_ema[:, 1::2], s=20, alpha=0.6,
                      label=f'{distance.replace("_", " ").title()} positions')
            ax.set_title(f"{distance.replace('_', ' ').title()} Positions - '{phone}'", pad=10)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Plot distance distributions
            ax = plt.subplot(3, 2, 2 + 2 * i_distance)
            ax.hist(distances_array, bins=20, density=True, alpha=0.7,
                   label=f'{distance.replace("_", " ").title()} distances')
            ax.axvline(np.mean(distances_array), color='r', linestyle='--',
                      label=f'Mean: {np.mean(distances_array):.2f} mm')
            ax.set_title(f"Distance Distribution", pad=10)
            ax.set_xlabel("Distance (mm)", labelpad=8)
            ax.set_ylabel("Density", labelpad=8)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.show()

    # Create interactive phone selector
    phones = sorted(phones_metrics.keys())
    ipw.interact(show_phone_metrics, phone=phones)
