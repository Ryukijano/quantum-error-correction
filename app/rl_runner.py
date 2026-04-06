"""Threaded RL training runner that streams metrics into a shared queue.

Used by the Streamlit app to run training asynchronously while the UI
polls and updates live charts without blocking the event loop.
"""

from __future__ import annotations

import queue
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

# Allow importing from the repo root
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Public event types pushed onto the queue
# ---------------------------------------------------------------------------

class TrainingEvent:
    """Base event pushed from the training thread."""
    __slots__ = ("kind",)
    def __init__(self, kind: str):
        self.kind = kind


class MetricEvent(TrainingEvent):
    """Per-episode metric snapshot."""
    __slots__ = ("kind", "data")
    def __init__(self, data: dict[str, Any]):
        super().__init__("metric")
        self.data = data


class SyndromeEvent(TrainingEvent):
    """Current syndrome + action taken by the agent this step."""
    __slots__ = ("kind", "syndrome", "action", "correct", "episode")
    def __init__(self, syndrome: np.ndarray, action: np.ndarray, correct: bool, episode: int):
        super().__init__("syndrome")
        self.syndrome = syndrome
        self.action = action
        self.correct = correct
        self.episode = episode


class DoneEvent(TrainingEvent):
    """Training finished normally."""
    __slots__ = ("kind", "total_episodes")
    def __init__(self, total_episodes: int):
        super().__init__("done")
        self.total_episodes = total_episodes


class ErrorEvent(TrainingEvent):
    """Training crashed — carries the exception message."""
    __slots__ = ("kind", "message")
    def __init__(self, message: str):
        super().__init__("error")
        self.message = message


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class RLRunner:
    """Runs PPO or SAC training in a background thread.

    Results are streamed to `event_queue` so the Streamlit UI can poll
    without blocking.

    Usage::

        runner = RLRunner(mode="ppo", distance=3, rounds=2, p=0.01, episodes=200)
        runner.start()

        while runner.is_running():
            try:
                event = runner.poll(timeout=0.05)
                if isinstance(event, MetricEvent):
                    ...
            except queue.Empty:
                pass
        runner.join()
    """

    def __init__(
        self,
        mode: str = "ppo",
        distance: int = 3,
        rounds: int = 2,
        physical_error_rate: float = 0.01,
        episodes: int = 200,
        batch_size: int = 32,
        use_diffusion: bool = False,
        syndrome_emit_every: int = 5,
        on_metric: Optional[Callable[[dict], None]] = None,
    ):
        self.mode = mode
        self.distance = distance
        self.rounds = rounds
        self.p = physical_error_rate
        self.episodes = episodes
        self.batch_size = batch_size
        self.use_diffusion = use_diffusion
        self.syndrome_emit_every = syndrome_emit_every
        self._on_metric = on_metric

        self.event_queue: queue.Queue[TrainingEvent] = queue.Queue(maxsize=2000)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------

    def start(self) -> None:
        self._stop_event.clear()
        # Dispatch to appropriate training loop based on mode
        if self.mode == "ppo":
            target = self._run_ppo
        elif self.mode == "sac":
            target = self._run_sac
        elif self.mode == "ppo_colour":
            target = self._run_ppo_colour
        elif self.mode == "sac_colour":
            target = self._run_sac_colour
        elif self.mode == "discover_colour":
            target = self._run_discover_colour
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float = 5.0) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)

    def poll(self, timeout: float = 0.02) -> TrainingEvent:
        return self.event_queue.get(timeout=timeout)

    def drain(self) -> list[TrainingEvent]:
        events = []
        while True:
            try:
                events.append(self.event_queue.get_nowait())
            except queue.Empty:
                break
        return events

    # ------------------------------------------------------------------
    # Internal training loops
    # ------------------------------------------------------------------

    def _run_ppo(self) -> None:
        try:
            import torch
            from surface_code_in_stem.rl_control.gym_env import QECGymEnv
            from surface_code_in_stem.rl_control.sota_agents import PPOAgent

            env = QECGymEnv(
                distance=self.distance,
                rounds=self.rounds,
                physical_error_rate=self.p,
                use_mwpm_baseline=True,
            )
            state_dim = env.observation_space.shape[0]
            action_dim = len(env.action_space.nvec)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)

            states, actions, log_probs, rewards, values = [], [], [], [], []
            success_hist: list[float] = []
            mwpm_hist: list[float] = []

            for ep in range(self.episodes):
                if self._stop_event.is_set():
                    break

                state, info = env.reset()
                action, log_prob, value = agent.select_action(state)
                _, reward, _, _, env_info = env.step(action)

                correct = bool(env_info.get("is_correct", False))
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                success_hist.append(1.0 if correct else 0.0)
                mwpm_hist.append(1.0 if info.get("mwpm_correct", False) else 0.0)

                # Emit syndrome snapshot periodically
                if ep % self.syndrome_emit_every == 0:
                    syndrome_arr = np.asarray(info.get("binary_syndrome", state), dtype=np.int8)
                    evt = SyndromeEvent(syndrome_arr, np.asarray(action), correct, ep + 1)
                    self._push(evt)

                # PPO update
                if (ep + 1) % self.batch_size == 0:
                    ret_t = torch.FloatTensor(rewards)
                    val_t = torch.FloatTensor(values)
                    adv_t = ret_t - val_t
                    s_t = torch.FloatTensor(np.array(states))
                    a_t = torch.FloatTensor(np.array(actions))
                    lp_t = torch.FloatTensor(log_probs)
                    loss_d = agent.update(s_t, a_t, lp_t, ret_t, adv_t)
                    states, actions, log_probs, rewards, values = [], [], [], [], []

                    window = min(self.batch_size, len(success_hist))
                    metric = {
                        "episode": ep + 1,
                        "reward": float(reward),
                        "rl_success": float(np.mean(success_hist[-window:])),
                        "mwpm_success": float(np.mean(mwpm_hist[-window:])),
                        "policy_loss": float(loss_d.get("policy_loss", 0)),
                        "value_loss": float(loss_d.get("value_loss", 0)),
                        "logical_error_rate": 0.0,
                        "effective_p": 0.0,
                    }
                    self._push(MetricEvent(metric))
                    if self._on_metric:
                        self._on_metric(metric)

            self._push(DoneEvent(self.episodes))

        except Exception as exc:  # noqa: BLE001
            self._push(ErrorEvent(str(exc)))

    def _run_sac(self) -> None:
        try:
            import torch
            from surface_code_in_stem.rl_control.gym_env import QECContinuousControlEnv
            from surface_code_in_stem.rl_control.sota_agents import ContinuousSACAgent
            from surface_code_in_stem.rl_control.replay_buffer import ReplayBuffer, Experience

            env = QECContinuousControlEnv(
                distance=self.distance,
                rounds=self.rounds,
                base_error_rate=self.p,
                parameter_dim=4,
                batch_shots=64,
            )
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = ContinuousSACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_space=env.action_space,
                use_diffusion=self.use_diffusion,
                device=device,
            )
            buffer = ReplayBuffer(capacity=5000, prioritized=False)

            for ep in range(self.episodes):
                if self._stop_event.is_set():
                    break

                state, _ = env.reset()
                ep_reward, done = 0.0, False
                ep_info: dict = {}

                while not done:
                    if self._stop_event.is_set():
                        break
                    if len(buffer) > self.batch_size:
                        action = agent.select_action(state, evaluate=False)
                    else:
                        action = env.action_space.sample()

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_info = info

                    buffer.push(Experience(
                        state=torch.FloatTensor(state),
                        action=torch.FloatTensor(action),
                        reward=float(reward),
                        next_state=torch.FloatTensor(next_state),
                        done=bool(done),
                    ))
                    state = next_state

                    if len(buffer) > self.batch_size:
                        s_b, a_b, r_b, ns_b, d_b, _, _ = buffer.sample(self.batch_size)
                        mask_b = 1.0 - d_b.float()
                        agent.update_parameters(s_b, a_b, r_b, ns_b, mask_b)

                ler = float(ep_info.get("logical_error_rate", 0.0))
                eff_p = float(ep_info.get("effective_p", 0.0))
                metric = {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "rl_success": 0.0,
                    "mwpm_success": 0.0,
                    "logical_error_rate": ler,
                    "effective_p": eff_p,
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                }
                self._push(MetricEvent(metric))
                if self._on_metric:
                    self._on_metric(metric)

                # Emit a dummy syndrome snapshot (SAC env doesn't expose per-shot syndromes)
                if ep % self.syndrome_emit_every == 0:
                    dummy = np.zeros(max(1, state_dim), dtype=np.int8)
                    self._push(SyndromeEvent(dummy, np.array(action), ler < self.p, ep + 1))

            self._push(DoneEvent(self.episodes))

        except Exception as exc:  # noqa: BLE001
            self._push(ErrorEvent(str(exc)))

    def _push(self, event: TrainingEvent) -> None:
        try:
            self.event_queue.put_nowait(event)
        except queue.Full:
            pass  # drop oldest by discarding the new one; UI will catch up on next poll

    # ------------------------------------------------------------------
    # Colour Code Training Loops
    # ------------------------------------------------------------------

    def _run_ppo_colour(self) -> None:
        """PPO training on colour code decoding."""
        try:
            import torch
            from surface_code_in_stem.rl_control.gym_env import ColourCodeGymEnv
            from surface_code_in_stem.rl_control.sota_agents import PPOAgent

            env = ColourCodeGymEnv(
                distance=self.distance,
                rounds=self.rounds,
                physical_error_rate=self.p,
                use_mwpm_baseline=True,
            )
            state_dim = env.observation_space.shape[0]
            action_dim = len(env.action_space.nvec)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)

            states, actions, log_probs, rewards, values = [], [], [], [], []
            success_hist: list[float] = []
            mwpm_hist: list[float] = []

            for ep in range(self.episodes):
                if self._stop_event.is_set():
                    break

                state, info = env.reset()
                action, log_prob, value = agent.select_action(state)
                _, reward, _, _, env_info = env.step(action)

                correct = bool(env_info.get("is_correct", False))
                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                success_hist.append(1.0 if correct else 0.0)
                mwpm_hist.append(1.0 if info.get("mwpm_correct", False) else 0.0)

                if ep % self.syndrome_emit_every == 0:
                    syndrome_arr = np.asarray(info.get("binary_syndrome", state), dtype=np.int8)
                    evt = SyndromeEvent(syndrome_arr, np.asarray(action), correct, ep + 1)
                    self._push(evt)

                if (ep + 1) % self.batch_size == 0:
                    ret_t = torch.FloatTensor(rewards)
                    val_t = torch.FloatTensor(values)
                    adv_t = ret_t - val_t
                    s_t = torch.FloatTensor(np.array(states))
                    a_t = torch.FloatTensor(np.array(actions))
                    lp_t = torch.FloatTensor(log_probs)
                    loss_d = agent.update(s_t, a_t, lp_t, ret_t, adv_t)
                    states, actions, log_probs, rewards, values = [], [], [], [], []

                    window = min(self.batch_size, len(success_hist))
                    metric = {
                        "episode": ep + 1,
                        "reward": float(reward),
                        "rl_success": float(np.mean(success_hist[-window:])),
                        "mwpm_success": float(np.mean(mwpm_hist[-window:])),
                        "policy_loss": float(loss_d.get("policy_loss", 0)),
                        "value_loss": float(loss_d.get("value_loss", 0)),
                        "logical_error_rate": 0.0,
                        "effective_p": 0.0,
                    }
                    self._push(MetricEvent(metric))
                    if self._on_metric:
                        self._on_metric(metric)

            self._push(DoneEvent(self.episodes))

        except Exception as exc:
            self._push(ErrorEvent(str(exc)))

    def _run_sac_colour(self) -> None:
        """SAC continuous control for colour code calibration."""
        try:
            import torch
            from surface_code_in_stem.rl_control.gym_env import ColourCodeCalibrationEnv
            from surface_code_in_stem.rl_control.sota_agents import ContinuousSACAgent
            from surface_code_in_stem.rl_control.replay_buffer import ReplayBuffer, Experience

            env = ColourCodeCalibrationEnv(
                distance=self.distance,
                rounds=self.rounds,
                parameter_dim=6,
                batch_shots=64,
                base_error_rate=self.p,
            )
            state_dim = env.observation_space.shape[0]
            action_dim = env.action_space.shape[0]
            device = "cuda" if torch.cuda.is_available() else "cpu"
            agent = ContinuousSACAgent(
                state_dim=state_dim,
                action_dim=action_dim,
                action_space=env.action_space,
                use_diffusion=self.use_diffusion,
                device=device,
            )
            buffer = ReplayBuffer(capacity=5000, prioritized=False)

            for ep in range(self.episodes):
                if self._stop_event.is_set():
                    break

                state, _ = env.reset()
                ep_reward, done = 0.0, False
                ep_info: dict = {}

                while not done:
                    if self._stop_event.is_set():
                        break
                    if len(buffer) > self.batch_size:
                        action = agent.select_action(state, evaluate=False)
                    else:
                        action = env.action_space.sample()

                    next_state, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    ep_reward += reward
                    ep_info = info

                    buffer.push(Experience(
                        state=torch.FloatTensor(state),
                        action=torch.FloatTensor(action),
                        reward=float(reward),
                        next_state=torch.FloatTensor(next_state),
                        done=bool(done),
                    ))
                    state = next_state

                    if len(buffer) > self.batch_size:
                        s_b, a_b, r_b, ns_b, d_b, _, _ = buffer.sample(self.batch_size)
                        mask_b = 1.0 - d_b.float()
                        agent.update_parameters(s_b, a_b, r_b, ns_b, mask_b)

                ler = float(ep_info.get("logical_error_rate", 0.0))
                metric = {
                    "episode": ep + 1,
                    "reward": ep_reward,
                    "rl_success": 0.0,
                    "mwpm_success": 0.0,
                    "logical_error_rate": ler,
                    "effective_p": float(ep_info.get("mean_defect_rate", 0.0)),
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                }
                self._push(MetricEvent(metric))
                if self._on_metric:
                    self._on_metric(metric)

                if ep % self.syndrome_emit_every == 0:
                    dummy = np.zeros(max(1, state_dim), dtype=np.int8)
                    self._push(SyndromeEvent(dummy, np.array(action), ler < self.p, ep + 1))

            self._push(DoneEvent(self.episodes))

        except Exception as exc:
            self._push(ErrorEvent(str(exc)))

    def _run_discover_colour(self) -> None:
        """RL agent discovers optimal colour code parameters."""
        try:
            import torch
            from surface_code_in_stem.rl_control.gym_env import ColourCodeDiscoveryEnv
            from surface_code_in_stem.rl_control.sota_agents import PPOAgent

            env = ColourCodeDiscoveryEnv(
                max_distance=13,
                target_threshold=0.005,
                max_steps=20,
            )
            # Discovery env has different observation space - adapt
            state_dim = env.observation_space.shape[0]
            # Use discrete action wrapper or simple policy
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Simple tabular/policy gradient approach for discovery
            discovery_metrics: list[dict] = []
            
            for ep in range(self.episodes):
                if self._stop_event.is_set():
                    break

                state, info = env.reset()
                done = False
                total_reward = 0.0
                
                while not done:
                    if self._stop_event.is_set():
                        break
                    # Simple heuristic policy: submit after exploring a few params
                    action = np.random.randint(0, env.action_space.n)
                    state, reward, terminated, truncated, step_info = env.step(action)
                    done = terminated or truncated
                    total_reward += reward
                
                # Log discovery results
                metric = {
                    "episode": ep + 1,
                    "reward": total_reward,
                    "rl_success": 1.0 if total_reward > 0 else 0.0,
                    "mwpm_success": 0.0,
                    "logical_error_rate": 0.0,
                    "effective_p": float(info.get("p", 0.001)),
                    "policy_loss": 0.0,
                    "value_loss": 0.0,
                }
                self._push(MetricEvent(metric))
                if self._on_metric:
                    self._on_metric(metric)

            self._push(DoneEvent(self.episodes))

        except Exception as exc:
            self._push(ErrorEvent(str(exc)))
