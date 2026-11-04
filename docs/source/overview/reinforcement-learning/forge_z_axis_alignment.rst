FORGE PegInsert Z-axis Alignment Penalty
=======================================

FORGE PegInsert adds a tolerance penalty for the angle between the peg and socket
root frames' local positive Z axes in world coordinates. It uses the true asset
orientations, independently of position and rotation about either local Z axis.
The existing rewards, success criteria, controller, PPO configuration, and
24/61/7 actor/critic/action dimensions are preserved. The controller's roll/pitch
action restriction is unchanged; the penalty does not add active tilt control.

Reward and configuration
------------------------

With the axis angle :math:`\theta\in[0,\pi]`, tolerance :math:`\alpha`, and signed
weight :math:`s`, the added contribution is:

.. math::

   p = \left[\operatorname{clamp}\left(\frac{\theta-\alpha}{\pi-\alpha},0,1\right)\right]^{1/3},
   \qquad r_{\mathrm{total}}=r_{\mathrm{old}}+s p.

``env.task.z_axis_align_alpha_deg`` is a fixed tolerance in degrees, defaulting to
5.0, and must be finite and in [0, 180). ``env.task.z_axis_align_scale`` defaults to
-1.0 for PegInsert and must be finite and in [-1, 0]. Other FORGE tasks default to
zero and reject nonzero weights. No parameter is added to the policy observations.

Angles at or below the tolerance have zero penalty. The positive penalty increases
continuously to 1 at 180 degrees; the actual contribution is nonpositive. The
cube-root curve has a steep, nonsmooth onset at the tolerance. At a 5-degree
tolerance, the penalties at 15, 30, 45 and 90 degrees are approximately 0.385,
0.523, 0.611 and 0.786. These are formula values, not training results.

Hydra overrides apply to train, play and evaluate without new shell arguments::

   env.task.z_axis_align_alpha_deg=5.0 env.task.z_axis_align_scale=-1.0

Set the scale to zero to recover the old total reward while retaining PegInsert
geometry and unweighted-penalty diagnostics. Record the environment configuration
and source revision with checkpoints: reward implementation is not stored in model
weights. Frozen-policy evaluation scores the new reward without updating the policy.

Episode metrics
---------------

Training and evaluation emit the following tags directly under ``ZAxisAlign/``,
without an ``Episode/`` prefix. With episode length T and fixed scale s:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Tag suffix
     - Per-episode meaning
   * - ``z_axis_align_penalty_mean``
     - Mean positive penalty, including zero samples; [0, 1], lower is better.
   * - ``z_axis_align_weighted_mean``
     - Mean actual contribution; [s, 0], closer to zero is better.
   * - ``z_axis_align_weighted_return``
     - Sum of contributions; [s*T, 0], also depends on episode length and scale.
   * - ``z_axis_angle_mean_deg``
     - Mean axis angle [deg].
   * - ``z_axis_angle_p95_deg``
     - Episode 95th-percentile angle [deg].
   * - ``z_axis_angle_final_deg``
     - Last scored angle before reset [deg].
   * - ``z_axis_angle_max_deg``
     - Maximum episode angle [deg].
   * - ``z_axis_aligned_fraction_5deg``
     - Fraction of steps at or below 5 degrees; higher is better.
   * - ``z_axis_aligned_fraction_15deg``
     - Fraction of steps at or below 15 degrees; higher is better.
   * - ``z_axis_violation_fraction``
     - Fraction of steps above the configured tolerance; lower is better.
   * - ``z_axis_excess_mean_deg``
     - Mean max(angle - tolerance, 0) [deg], including nonviolating zeros.
   * - ``z_axis_reverse_fraction``
     - Fraction of steps above 90 degrees; lower is better.
   * - ``z_axis_align_alpha_deg``
     - Configured episode tolerance [deg], not a performance score.

All statistics include the terminal scoring step and exclude reset positioning
simulation. Each episode has equal weight. Training reports newly completed
episodes since the previous output; evaluation reports all retained episodes
cumulatively and excludes episodes beyond each environment's requested quota.
The P95 curve averages individual episode P95 values, not pooled trajectory angles.
The fixed 5/15-degree metrics are diagnostic standards, not reward shape parameters.

Instantaneous means retain ``logs_rew_z_axis_align`` (positive penalty) and
``logs_rew_z_axis_align_weighted`` (nonpositive contribution), each with
``/frame``, ``/iter`` and ``/time`` suffixes. Evaluation records these at completion
steps; they are not episode means. Compare geometric metrics, original insertion
success and force metrics rather than total reward alone. Penalty means across
different tolerances, or weighted values across different scales, are not directly
comparable performance measures.

Weights & Biases
---------------

Use ``--track --wandb-entity ENTITY --wandb-project-name PROJECT`` with either
``isaaclab.sh train --rl_library rl_games`` or
``isaaclab-2.sh evaluate --rl_library rl_games``. The task ID is
``IsaacContrib-Forge-PegInsert-Direct``. The uv-managed equivalents are
``uv run isaaclab train`` and ``uv run isaaclab evaluate``.

Training synchronizes its TensorBoard events; evaluation uploads reports explicitly
to a separate run without duplicating TensorBoard synchronization. The 13 episode
tags use cumulative environment transitions as their step. Both runs save the
environment and agent configurations and the ``z_axis_align_design`` marker
``cube_root_excess_angle_penalty_v1``. Online uploads require existing W&B
authentication and connectivity. Local files or offline runs alone do not verify
online delivery.
