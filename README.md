# rlrvo

Arena wrapper for **rl_rvo_nav**, a GRU-based actor-critic policy trained with Reciprocal Velocity Obstacle (RVO) collision-time reward shaping. Adapted from [hanruihua/rl_rvo_nav](https://github.com/hanruihua/rl_rvo_nav).

## Run

```sh
arena launch mobile:=drl mobile.planner:=rlrvo
```

Requires a global plan. Defaults to `nav2/navfn`.

## Files

- `planner.py`: entry point. Builds proprioceptive + per-pedestrian VO cone observations, runs the GRU policy, projects incremental holonomic velocity to `[v, omega]`.
- `policy_rnn_ac.py`: vendored upstream `rnn_ac` actor-critic.
- `planner.yaml`: observation manifest.
- `weights.yaml`: pulls the pretrained checkpoint + arg file from [arena-rosnav/rlrvo](https://huggingface.co/arena-rosnav/rlrvo) on `arena feature planners add rlrvo`.
- See [ATTRIBUTION.md](ATTRIBUTION.md) for code provenance.

## License

MIT (inherited from upstream).
