import os
import math
import gymnasium
import torch
import numpy as np
import matplotlib.pyplot as plt

ENV_ENGINES = ['gymnasium'] # , 'safety_gymnasium']

def render_gymnasium_agent(
        actor, 
        env: gymnasium.Env | None = None,
        env_id: str | None = None, 
        env_engine: str = 'gymnasium',
        num_episodes: int = 1,
        log_std: float | None = None, deterministic: bool = True,
        seed: int = 42, 
    ):
    """
    Render the agent in the Gymnasium or Safety Gymnasium environment using human render mode.

    Args:
        actor: Trained policy network
        env: Gymnasium environment instance (if None, env_id must be provided)
        env_id: Environment ID (to create environment if env is None)
        env_engine: 'gymnasium' or 'safety_gymnasium' (used if env_id is provided)
        num_episodes: Number of episodes to render
        log_std: Log standard deviation for continuous action spaces
        deterministic: Whether to use deterministic actions
        seed: Random seed
    
    Returns:
        None
    """
    assert (env is not None) or (env_id is not None), "Either env or env_id must be provided"

    if env is not None:
        assert env.unwrapped.render_mode == 'human', "Environment must be created with render_mode='human'"
        env_id = env.spec.id
    elif env_id is not None:
        assert env_engine in ENV_ENGINES, f"env_engine must be one of {ENV_ENGINES}"
        if env_engine == 'gymnasium':
            env = gymnasium.make(
                env_id,
                render_mode='human',
            )
        else:
            raise NotImplementedError("Safety Gymnasium rendering not implemented yet")
            # env = safety_gymnasium.make(
            #     env_id,
            #     render_mode='human',
            # )
            # env = safety_gymnasium.wrappers.SafetyGymnasium2Gymnasium(env)

    if not deterministic and isinstance(env.action_space, gymnasium.spaces.Box):
        assert log_std is not None, "log_std must be provided for stochastic continuous actions"
        assert log_std > 0, "log_std must be positive"

    print(f"\n\n=== Rendering {actor} in {env_id} ===")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        done = False
        obs, _ = env.reset(seed=seed + episode)
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if isinstance(env.action_space, gymnasium.spaces.Box):
                    # Continuous actions: use mean from actor
                    if deterministic:
                        action = actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = actor(obs_t)
                        std = np.exp(log_std).expand_as(mean)
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    # Clip to action space bounds
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                else:
                    # Discrete actions: use argmax
                    if deterministic:
                        logits = actor(obs_t)
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        logits = actor(obs_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()


def plot_gymnasium_episode(
        actor: torch.nn.Module,
        env_id: str | None = None,
        env: gymnasium.Env | None = None,
        n_cols: int = 4,
        log_std: torch.nn.Parameter | None = None,
        deterministic: bool = True,
        seed: int = 42,
        env_kwargs: dict | None = None,
        save_path: str | None = None,
        figsize_per_frame: tuple[float, float] = (3.0, 3.0),
        title: str | None = None,
    ):
    """
    Run one episode in a Gymnasium environment and display all frames as a matplotlib grid.

    Args:
        actor: Trained policy network (nn.Sequential or nn.Module).
        env_id: Gymnasium environment ID (e.g. 'FrozenLake-v1').
        env: Optional pre-created Gymnasium environment (if None, env_id is used to create it).
        n_cols: Number of columns in the image grid.
        log_std: Log standard deviation parameter for continuous action spaces.
        deterministic: Whether to select actions deterministically.
        seed: Random seed for the episode.
        env_kwargs: Optional extra keyword arguments passed to gymnasium.make.
        save_path: If provided, save the figure to this file path (parent dirs
            are created automatically).
        figsize_per_frame: (width, height) in inches for each subplot cell.
        title: Optional suptitle for the figure.

    Returns:
        list[np.ndarray]: The collected RGB frames.
    """
    
    if (env is None) and (env_id is None):
        raise ValueError("Either env or env_id must be provided")
    
    if env is not None:
        assert env.unwrapped.render_mode == 'rgb_array', "Environment must be created with render_mode='rgb_array'"
        env_id = env.spec.id
    elif env_id is not None:
        env_kwargs = env_kwargs or {}
        env = gymnasium.make(env_id, render_mode='rgb_array', **env_kwargs)

    continuous_actions = isinstance(env.action_space, gymnasium.spaces.Box)
    if not deterministic and continuous_actions:
        assert log_std is not None, "log_std must be provided for stochastic continuous actions"

    # --- collect frames ---
    frames: list[np.ndarray] = []
    obs, _ = env.reset(seed=seed)
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    done = False
    while not done:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            if continuous_actions:
                if deterministic:
                    action = actor(obs_t).cpu().numpy()[0]
                else:
                    mean = actor(obs_t)
                    std = torch.exp(log_std)  # type: ignore[arg-type]
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample().cpu().numpy()[0]
                action = np.clip(action, env.action_space.low, env.action_space.high)
            else:
                logits = actor(obs_t)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()

        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        done = terminated or truncated

    env.close()

    # --- plot grid ---
    n_frames = len(frames)
    n_rows = math.ceil(n_frames / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_frame[0] * n_cols, figsize_per_frame[1] * n_rows),
    )
    # Always work with a flat array of axes
    axes = np.asarray(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx < n_frames:
            ax.imshow(frames[idx])
            ax.set_title(f"Step {idx}", fontsize=9)
        ax.axis('off')

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return frames


def plot_state_action_pairs(
    env: gymnasium.Env,
    state_action_pairs: list[tuple[int, int]],
    nrow: int | None = None,
    ncol: int | None = None,
    action_map: dict[int, tuple[float, float]] | None = None,
    arrow_color: str = "red",
    arrow_alpha: float = 0.85,
    arrow_scale: float = 0.35,
    arrow_width: float = 2.5,
    head_width: float = 8.0,
    head_length: float = 5.0,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
    save_path: str | None = None,
    seed: int = 42,
):
    """
    Plot arrows for a list of (state, action) pairs on top of the initial
    rendered frame of a grid-world Gymnasium environment.

    The environment must use ``render_mode='rgb_array'``.  States are integer
    indices into the flattened grid (``state = row * ncol + col``), and actions
    are integer direction codes.

    Args:
        env: A Gymnasium grid-world environment created with
            ``render_mode='rgb_array'``.  The environment will be reset
            internally so that the initial frame can be rendered.
        state_action_pairs: Iterable of ``(state_int, action_int)`` tuples.
            ``state_int`` is the flat grid index; ``action_int`` selects a
            direction from *action_map*.
        nrow: Number of rows in the grid.  If *None* the function tries to
            read ``env.unwrapped.nrow`` or ``env.unwrapped.desc.shape[0]``.
        ncol: Number of columns in the grid.  Inferred the same way as *nrow*.
        action_map: Mapping from action index to a unit (dx, dy) direction
            **in image coordinates** (x → right, y → down).  Defaults to the
            FrozenLake convention::

                {0: (-1, 0),  # LEFT
                 1: (0, 1),   # DOWN
                 2: (1, 0),   # RIGHT
                 3: (0, -1)}  # UP

        arrow_color: Matplotlib colour for arrows.
        arrow_alpha: Arrow transparency (0–1).
        arrow_scale: Arrow length as a fraction of cell size.
        arrow_width: Line width of the arrow shaft.
        head_width: Width of the arrowhead in points.
        head_length: Length of the arrowhead in points.
        title: Optional figure title.
        figsize: Optional ``(width, height)`` in inches.  If *None* the size
            is derived from the rendered frame.
        save_path: If given, save the figure to this path (parent directories
            are created automatically).
        seed: Seed passed to ``env.reset()``.

    Returns:
        matplotlib.figure.Figure: The figure containing the annotated frame.
    """
    import matplotlib.patches as mpatches  # local import to keep top-level light

    # ------------------------------------------------------------------
    # Infer grid dimensions
    # ------------------------------------------------------------------
    if nrow is None or ncol is None:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, "desc"):
            _nrow, _ncol = np.array(unwrapped.desc).shape
        elif hasattr(unwrapped, "nrow") and hasattr(unwrapped, "ncol"):
            _nrow, _ncol = unwrapped.nrow, unwrapped.ncol
        else:
            raise ValueError(
                "Cannot infer grid dimensions.  Pass nrow and ncol explicitly."
            )
        nrow = nrow or _nrow
        ncol = ncol or _ncol

    # ------------------------------------------------------------------
    # Default action map (FrozenLake: LEFT=0, DOWN=1, RIGHT=2, UP=3)
    # Directions expressed as (dx, dy) in *image* coordinates.
    # ------------------------------------------------------------------
    if action_map is None:
        action_map = {
            0: (-1, 0),   # LEFT
            1: (0, 1),    # DOWN
            2: (1, 0),    # RIGHT
            3: (0, -1),   # UP
        }

    # ------------------------------------------------------------------
    # Render the initial frame
    # ------------------------------------------------------------------
    assert env.unwrapped.render_mode == "rgb_array", (
        "Environment must be created with render_mode='rgb_array'"
    )
    env.reset(seed=seed)
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None – check render_mode.")

    img_h, img_w = frame.shape[:2]
    cell_w = img_w / ncol
    cell_h = img_h / nrow

    # ------------------------------------------------------------------
    # Create figure
    # ------------------------------------------------------------------
    if figsize is None:
        dpi = 100
        figsize = (img_w / dpi + 0.5, img_h / dpi + 0.5)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(frame)
    ax.axis("off")

    # ------------------------------------------------------------------
    # Draw arrows
    # ------------------------------------------------------------------
    for state, action in state_action_pairs:
        row = state // ncol
        col = state % ncol

        # Centre of the cell in pixel coordinates
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h

        dx_unit, dy_unit = action_map[action]
        dx = dx_unit * cell_w * arrow_scale
        dy = dy_unit * cell_h * arrow_scale

        ax.annotate(
            "",
            xy=(cx + dx, cy + dy),         # arrow tip
            xytext=(cx - dx, cy - dy),      # arrow tail (opposite side)
            arrowprops=dict(
                arrowstyle=f"->,head_width={head_width / 72:.4f},head_length={head_length / 72:.4f}",
                color=arrow_color,
                lw=arrow_width,
                alpha=arrow_alpha,
            ),
        )

    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="bold")

    fig.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return fig

def plot_gymnasium_episode_multitask(
        actor: torch.nn.Module,
        env_task1: gymnasium.Env | None = None,
        env_task2: gymnasium.Env | None = None,
        env_id_task1: str | None = None,
        env_id_task2: str | None = None,
        env_kwargs_task1: dict | None = None,
        env_kwargs_task2: dict | None = None,
        n_cols: int = 4,
        log_std: torch.nn.Parameter | None = None,
        deterministic: bool = True,
        seed: int = 42,
        save_path: str | None = None,
        figsize_per_frame: tuple[float, float] = (3.0, 3.0),
        title: str | None = None,
        task1_label: str = "Task 1 (Source)",
        task2_label: str = "Task 2 (Downstream)",
        task1_title_y: float = 0.7,
        task2_title_y: float = 0.7,
        suptitle_y: float = 0.98,
        frame_stride: int = 1,
    ):
    """
    Run one episode in two Gymnasium environments (source and downstream tasks)
    and display all frames as a matplotlib grid with both tasks clearly separated.

    The top rows show frames from Task 1 and the bottom rows show frames from
    Task 2, with a bold label separating the two sections.

    Args:
        actor: Trained policy network (nn.Sequential or nn.Module).
        env_task1: Optional pre-created Gymnasium environment for Task 1.
        env_task2: Optional pre-created Gymnasium environment for Task 2.
        env_id_task1: Gymnasium environment ID for Task 1 (used if env_task1 is None).
        env_id_task2: Gymnasium environment ID for Task 2 (used if env_task2 is None).
        env_kwargs_task1: Extra keyword arguments for Task 1 gymnasium.make.
        env_kwargs_task2: Extra keyword arguments for Task 2 gymnasium.make.
        n_cols: Number of columns in the image grid.
        log_std: Log standard deviation parameter for continuous action spaces.
        deterministic: Whether to select actions deterministically.
        seed: Random seed for both episodes.
        save_path: If provided, save the figure to this file path.
        figsize_per_frame: (width, height) in inches for each subplot cell.
        title: Optional suptitle for the entire figure.
        task1_label: Label displayed above the Task 1 rows.
        task2_label: Label displayed above the Task 2 rows.
        frame_stride: Display every n-th frame starting from frame 0.
            The last frame is always included regardless of the stride.
            Default 1 shows every frame.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: The collected RGB frames
            for Task 1 and Task 2 respectively (all frames, before striding).
    """

    def _collect_frames(env, env_id, env_kwargs):
        """Create env if needed and collect one episode of frames."""
        if env is None and env_id is None:
            raise ValueError("Either env or env_id must be provided for each task")

        close_env = False
        if env is not None:
            assert env.unwrapped.render_mode == 'rgb_array', \
                "Environment must be created with render_mode='rgb_array'"
        elif env_id is not None:
            env_kwargs = env_kwargs or {}
            env = gymnasium.make(env_id, render_mode='rgb_array', **env_kwargs)
            close_env = True

        continuous_actions = isinstance(env.action_space, gymnasium.spaces.Box)
        if not deterministic and continuous_actions:
            assert log_std is not None, \
                "log_std must be provided for stochastic continuous actions"

        frames: list[np.ndarray] = []
        obs, _ = env.reset(seed=seed)
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                if continuous_actions:
                    if deterministic:
                        action = actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = actor(obs_t)
                        std = torch.exp(log_std)  # type: ignore[arg-type]
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                else:
                    logits = actor(obs_t)
                    if deterministic:
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            done = terminated or truncated

        if close_env:
            env.close()

        return frames

    # --- collect frames for both tasks ---
    frames_task1 = _collect_frames(env_task1, env_id_task1, env_kwargs_task1)
    frames_task2 = _collect_frames(env_task2, env_id_task2, env_kwargs_task2)

    # --- apply frame stride (always keep frame 0 and the last frame) ---
    def _apply_stride(frames, stride):
        """Return (display_frames, original_indices) honouring stride + last frame."""
        indices = list(range(0, len(frames), stride))
        if frames and (len(frames) - 1) not in indices:
            indices.append(len(frames) - 1)
        return [frames[i] for i in indices], indices

    display_t1, indices_t1 = _apply_stride(frames_task1, frame_stride)
    display_t2, indices_t2 = _apply_stride(frames_task2, frame_stride)

    # --- compute grid layout ---
    n_frames_t1 = len(display_t1)
    n_frames_t2 = len(display_t2)
    n_rows_t1 = math.ceil(n_frames_t1 / n_cols)
    n_rows_t2 = math.ceil(n_frames_t2 / n_cols)

    # Dedicated label rows for Task 1 (top) and Task 2 (separator)
    label_ratio = 0.18  # height of a label row relative to a frame row
    total_rows = 1 + n_rows_t1 + 1 + n_rows_t2  # label1 + t1 frames + label2 + t2 frames

    fig_w = figsize_per_frame[0] * n_cols
    fig_h = figsize_per_frame[1] * (n_rows_t1 + n_rows_t2) + figsize_per_frame[1] * label_ratio * 2

    height_ratios = (
        [label_ratio]
        + [1] * n_rows_t1
        + [label_ratio]
        + [1] * n_rows_t2
    )

    fig, axes = plt.subplots(
        total_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={
            'height_ratios': height_ratios,
            'hspace': 0.08,
            'wspace': 0.04,
        },
    )
    axes = np.asarray(axes).reshape(total_rows, n_cols)

    # --- Task 1 label row (row 0) ---
    for col_idx in range(n_cols):
        axes[0, col_idx].axis('off')
    axes[0, n_cols // 2].text(
        x=0.5, y=task1_title_y, 
        s=task1_label,
        transform=axes[0, n_cols // 2].transAxes,
        fontsize=11, fontweight='bold',
        ha='center', va='center',
    )

    # --- Task 1 frame rows ---
    for row_idx in range(n_rows_t1):
        for col_idx in range(n_cols):
            ax = axes[1 + row_idx, col_idx]
            frame_idx = row_idx * n_cols + col_idx
            if frame_idx < n_frames_t1:
                ax.imshow(display_t1[frame_idx])
                ax.set_title(f"Step {indices_t1[frame_idx]}", fontsize=8, pad=2)
            ax.axis('off')

    # --- Task 2 label row ---
    sep_row = 1 + n_rows_t1
    for col_idx in range(n_cols):
        axes[sep_row, col_idx].axis('off')
    axes[sep_row, n_cols // 2].text(
        x=0.5, y=task2_title_y, 
        s=task2_label,
        transform=axes[sep_row, n_cols // 2].transAxes,
        fontsize=11, fontweight='bold',
        ha='center', va='center',
    )
    axes[sep_row, 0].set_ylabel(
        task2_label, fontsize=11, fontweight='bold', labelpad=10
    )

    # --- Task 2 frame rows ---
    for row_idx in range(n_rows_t2):
        for col_idx in range(n_cols):
            ax = axes[sep_row + 1 + row_idx, col_idx]
            frame_idx = row_idx * n_cols + col_idx
            if frame_idx < n_frames_t2:
                ax.imshow(display_t2[frame_idx])
                ax.set_title(f"Step {indices_t2[frame_idx]}", fontsize=8, pad=2)
            ax.axis('off')

    fig.tight_layout(pad=0.3)

    if title is not None:
        fig.suptitle(title, fontsize=13, fontweight='bold',y=suptitle_y)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()
    return frames_task1, frames_task2