#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_safe_state_action_dataset(dataset, env_map, save_path=None):
    """
    Visualize safe state-action pairs in a grid layout.
    
    Args:
        dataset: TensorDataset containing (states, actions) pairs
        env_map: Environment map for the task
        save_path: Optional path to save the figure
    """
    # Extract states and actions
    states = dataset.tensors[0].numpy()  # Shape: (N, state_dim)
    actions = dataset.tensors[1].numpy()  # Shape: (N,)
    
    # Group by state position: state_pos -> list of actions
    state_action_data = {}
    
    for state, action in zip(states, actions):
        # Extract position from one-hot encoding (exclude task indicator which is last)
        state_ohe = state[:-1]
        pos = np.argmax(state_ohe)
        
        if pos not in state_action_data:
            state_action_data[pos] = []
        # Convert action to int to make it hashable
        action_int = int(action) if isinstance(action, (np.ndarray, np.generic)) else action
        state_action_data[pos].append(action_int)
    
    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    action_names = ["←", "↓", "→", "↑"]
    action_colors = ['blue', 'green', 'red', 'purple']
    
    # Convert map to array if needed
    if isinstance(env_map[0], str):
        desc = np.array([list(row) for row in env_map])
    else:
        desc = np.array(env_map)
    
    nrow, ncol = desc.shape
    
    # Set up grid
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(-0.5, nrow - 0.5)
    ax.set_aspect('equal')
    ax.grid(True, linewidth=1, alpha=0.5, color='black')
    ax.set_xticks(range(ncol))
    ax.set_yticks(range(nrow))
    ax.invert_yaxis()
    
    # Draw environment cells
    for i in range(nrow):
        for j in range(ncol):
            cell = desc[i, j]
            
            if cell == 'S':  # Start
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                        color='lightgreen', alpha=0.3)
                ax.add_patch(rect)
                ax.text(j, i - 0.3, 'S', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='green')
            elif cell == 'F':  # Frozen (safe)
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                        color='lightblue', alpha=0.2)
                ax.add_patch(rect)
            elif cell == 'H':  # Hole (unsafe)
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                        color='red', alpha=0.3)
                ax.add_patch(rect)
                ax.text(j, i - 0.3, 'H', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='darkred')
            elif cell == 'G':  # Goal
                rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1, 
                                        color='gold', alpha=0.4)
                ax.add_patch(rect)
                ax.text(j, i - 0.3, 'G', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='darkgoldenrod')
    
    # Draw safe actions for each state
    for pos, actions_list in state_action_data.items():
        row = pos // ncol
        col = pos % ncol
        
        # Get unique actions
        unique_actions = sorted(set(actions_list))
        
        # Draw action indicators
        if len(unique_actions) == 1:
            # Single action - large arrow in center
            action = unique_actions[0]
            ax.text(col, row + 0.2, action_names[action], ha='center', va='center',
                   fontsize=20, fontweight='bold', color=action_colors[action],
                   bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
        else:
            # Multiple actions - show all around the cell
            positions = [
                (-0.25, 0),    # LEFT
                (0, 0.25),     # DOWN
                (0.25, 0),     # RIGHT
                (0, -0.25)     # UP
            ]
            for action in unique_actions:
                dx, dy = positions[action]
                ax.text(col + dx, row + dy, action_names[action], 
                       ha='center', va='center',
                       fontsize=12, fontweight='bold', 
                       color=action_colors[action],
                       bbox=dict(boxstyle='circle,pad=0.1', 
                               facecolor='white', alpha=0.9, edgecolor=action_colors[action]))
        
        # Add count indicator
        count_text = f"n={len(actions_list)}"
        ax.text(col, row + 0.4, count_text, ha='center', va='top',
               fontsize=8, color='black', alpha=0.7)
    
    ax.set_title(f'Safe State-Action Pairs - {len(state_action_data)} unique states\n'
                f'Total pairs: {sum(len(a) for a in state_action_data.values())}', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    fig.suptitle('Safe State-Action Pairs for Rashomon Set Computation', 
                fontsize=16, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(color=action_colors[i], label=f'{action_names[i]} {["LEFT", "DOWN", "RIGHT", "UP"][i]}')
        for i in range(4)
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              bbox_to_anchor=(0.5, -0.05), frameon=True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {save_path}")
    
    return fig

# %%
