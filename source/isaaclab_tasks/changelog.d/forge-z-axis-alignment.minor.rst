Added
^^^^^

* Added a directional local-z-axis alignment reward to FORGE PegInsert, with isolated per-episode
  decay-angle sampling, Hydra-configurable weights and degree bounds, and trajectory diagnostics.

Changed
^^^^^^^

* Enabled the alignment reward for PegInsert with weight 0.1 and decay angles sampled from 2 to 10
  degrees. To retain the previous total reward, set ``env.task.z_axis_align_scale=0.0``. Observation
  and action dimensions were preserved; the sampled decay angle was not added to either network input.
