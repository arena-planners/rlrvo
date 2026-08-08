"""rl_rvo_nav wrapper for the arena_planners bridge."""

from __future__ import annotations

import math
import pathlib

import numpy as np
import torch
import torch.nn as nn
from arena_planners.sdk import load_manifest, main_loop
from gym.spaces import Box

from policy_rnn_ac import rnn_ac

_WEIGHTS = pathlib.Path(__file__).parent / "model" / "pre_train_check_point_1000.pt"

_NM: int = 5
_STATE_DIM: int = 6
_RNN_INPUT_DIM: int = 8
_RNN_HIDDEN_DIM: int = 256
_HIDDEN_SIZES_AC: tuple[int, int] = (256, 256)
_HIDDEN_SIZES_V: tuple[int, int] = (256, 256)
_RNN_MODE: str = "biGRU"
_V_PREF: float = 1.0
_RADIUS: float = 0.3
_PED_RADIUS: float = 0.3
_CTIME_THRESHOLD: float = 5.0
_ACCELER_VEL: float = 1.0

_policy: rnn_ac | None = None
_prev_vel: list[float] = [0.0, 0.0]


def _get_policy() -> rnn_ac:
    global _policy
    if _policy is None:
        obs_space = Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)
        act_space = Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        model = rnn_ac(
            obs_space,
            act_space,
            state_dim=_STATE_DIM,
            rnn_input_dim=_RNN_INPUT_DIM,
            rnn_hidden_dim=_RNN_HIDDEN_DIM,
            hidden_sizes_ac=_HIDDEN_SIZES_AC,
            hidden_sizes_v=_HIDDEN_SIZES_V,
            activation=nn.ReLU,
            output_activation=nn.Tanh,
            output_activation_v=nn.Identity,
            use_gpu=False,
            rnn_mode=_RNN_MODE,
        )
        ckpt = torch.load(str(_WEIGHTS), map_location="cpu")
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        _policy = model
    return _policy


def _wraptopi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _vo_entry(
    x: float,
    y: float,
    vx: float,
    vy: float,
    r: float,
    mx: float,
    my: float,
    mvx: float,
    mvy: float,
    mr: float = _PED_RADIUS,
    ctime_threshold: float = _CTIME_THRESHOLD,
) -> list[float]:
    rel_x = x - mx
    rel_y = y - my
    dis = math.sqrt(rel_x * rel_x + rel_y * rel_y)
    dis = max(dis, r + mr + 1e-6)
    angle_mr = math.atan2(my - y, mx - x)
    half_angle = math.asin(min((r + mr) / dis, 1.0))
    left_ori = _wraptopi(angle_mr + half_angle)
    right_ori = _wraptopi(angle_mr - half_angle)
    apex_vx = (vx + mvx) / 2.0
    apex_vy = (vy + mvy) / 2.0
    min_dis = max(dis - mr, 0.0)
    rel_speed = math.sqrt((vx - mvx) ** 2 + (vy - mvy) ** 2)
    if rel_speed < 1e-6:
        exp_time = ctime_threshold + 10.0
    else:
        exp_time = min(max(min_dis / rel_speed, 0.0), ctime_threshold + 10.0)
    exp_time_inv = 1.0 / (exp_time + 0.2)
    return [
        apex_vx,
        apex_vy,
        math.cos(left_ori),
        math.sin(left_ori),
        math.cos(right_ori),
        math.sin(right_ori),
        min_dis,
        exp_time_inv,
    ]


def step(features: dict) -> list[float]:
    policy = _get_policy()

    robot_pose = features.get("robot_pose")
    robot_state = features.get("robot_state")
    if robot_pose is None or robot_state is None:
        return [0.0, 0.0]

    px = float(robot_pose[0])
    py = float(robot_pose[1])
    theta = float(robot_pose[2])
    vx = float(robot_state[2]) if len(robot_state) > 3 else 0.0
    vy = float(robot_state[3]) if len(robot_state) > 3 else 0.0

    goal_pose = features.get("goal_pose")
    target: tuple[float, float] | None = None
    if goal_pose is not None and len(goal_pose) >= 2:
        target = (float(goal_pose[0]), float(goal_pose[1]))
    if target is None:
        return [0.0, 0.0]
    gx, gy = target

    dist = math.hypot(gx - px, gy - py)
    if dist < 1e-3:
        return [0.0, 0.0]
    des_vx = _V_PREF * (gx - px) / dist
    des_vy = _V_PREF * (gy - py) / dist

    propri_obs = np.array([vx, vy, des_vx, des_vy, theta, _RADIUS], dtype=np.float32)

    peds = features.get("pedestrians")
    if peds is None:
        peds = []
    vo_entries: list[list[float]] = []
    for ped in peds:
        mx = float(ped[1])
        my = float(ped[2])
        mvx = float(ped[3]) if len(ped) > 3 else 0.0
        mvy = float(ped[4]) if len(ped) > 4 else 0.0
        vo_entries.append(_vo_entry(px, py, vx, vy, _RADIUS, mx, my, mvx, mvy))

    vo_entries.sort(key=lambda e: e[-1], reverse=True)
    vo_entries = vo_entries[:_NM]

    if not vo_entries:
        exter_obs = np.zeros(_RNN_INPUT_DIM, dtype=np.float32)
    else:
        exter_obs = np.array(vo_entries, dtype=np.float32).flatten()

    obs = np.concatenate([propri_obs, exter_obs]).astype(np.float32)
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32)

    with torch.no_grad():
        a_inc = policy.act(obs_tensor, std_factor=1e-5)

    global _prev_vel
    abs_vx = float(np.clip(_prev_vel[0] + _ACCELER_VEL * float(a_inc[0]), -_V_PREF, _V_PREF))
    abs_vy = float(np.clip(_prev_vel[1] + _ACCELER_VEL * float(a_inc[1]), -_V_PREF, _V_PREF))
    _prev_vel = [abs_vx, abs_vy]

    return [abs_vx, abs_vy]


def on_reset(episode_id: str, initial_state: dict | None) -> None:
    global _prev_vel
    _prev_vel = [0.0, 0.0]


if __name__ == "__main__":
    manifest = load_manifest(pathlib.Path(__file__).parent / "planner.yaml")
    main_loop(step, manifest=manifest, on_reset=on_reset)
