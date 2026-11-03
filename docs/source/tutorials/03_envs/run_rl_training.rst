.. _tutorial-run-rl-training:

Training with an RL Agent
=========================

.. currentmodule:: isaaclab

In the previous tutorials, we covered how to define an RL task environment, register
it into the ``gym`` registry, and interact with it using a random agent. We now move
on to the next step: training an RL agent to solve the task.

Although the :class:`envs.ManagerBasedRLEnv` conforms to the :class:`gymnasium.Env` interface,
it is not exactly a ``gym`` environment. The input and outputs of the environment are
not numpy arrays, but rather based on torch tensors with the first dimension being the
number of environment instances.

Additionally, most RL libraries expect their own variation of an environment interface.
For example, `Stable-Baselines3`_ expects the environment to conform to its
`VecEnv API`_ which expects a list of numpy arrays instead of a single tensor. Similarly,
`RSL-RL`_, `RL-Games`_ and `SKRL`_ expect a different interface. Since there is no one-size-fits-all
solution, we do not base the :class:`envs.ManagerBasedRLEnv` on any particular learning library.
Instead, we implement wrappers to convert the environment into the expected interface.
These are specified in the :mod:`isaaclab_rl` module.

In this tutorial, we will use `Stable-Baselines3`_ to train an RL agent to solve the
cartpole balancing task.

.. caution::

  Wrapping the environment with the respective learning framework's wrapper should happen in the end,
  i.e. after all other wrappers have been applied. This is because the learning framework's wrapper
  modifies the interpretation of environment's APIs which may no longer be compatible with :class:`gymnasium.Env`.

The Code
--------

For this tutorial, we use the training implementation from `Stable-Baselines3`_ workflow in the
``isaaclab_rl.entrypoints.backends.train_sb3`` module.

.. dropdown:: Code for train_sb3.py
    :icon: code

    .. literalinclude:: ../../../../source/isaaclab_rl/isaaclab_rl/entrypoints/backends/train_sb3.py
      :language: python
      :linenos:
      :emphasize-lines: 97-100, 104-109, 121-137, 145-157, 164-170

The Code Explained
------------------

.. currentmodule:: isaaclab_rl.utils

Most of the code above is boilerplate code to create logging directories, saving the parsed configurations,
and setting up different Stable-Baselines3 components. For this tutorial, the important part is creating
the environment and wrapping it with the Stable-Baselines3 wrapper.

There are three wrappers used in the code above:

1. :class:`gymnasium.wrappers.RecordVideo`: This wrapper records a video of the environment
   and saves it to the specified directory. This is useful for visualizing the agent's behavior
   during training.
2. :class:`wrappers.sb3.Sb3VecEnvWrapper`: This wrapper converts the environment
   into a Stable-Baselines3 compatible environment.
3. `stable_baselines3.common.vec_env.VecNormalize`_: This wrapper normalizes the
   environment's observations and rewards.

Each of these wrappers wrap around the previous wrapper by following ``env = wrapper(env, *args, **kwargs)``
repeatedly. The final environment is then used to train the agent. For more information on how these
wrappers work, please refer to the :ref:`how-to-env-wrappers` documentation.

The Code Execution
------------------

We train a PPO agent from Stable-Baselines3 to solve the cartpole balancing task.

Training the agent
~~~~~~~~~~~~~~~~~~

There are three main ways to train the agent. Each of them has their own advantages and disadvantages.
It is up to you to decide which one you prefer based on your use case.

Headless execution
""""""""""""""""""

When no visualizer is requested, no interactive visualizer window is opened during training. This is useful
when training on a remote server or when you do not need live visual feedback, which can add some compute cost.
Rendering can still be active for sensor/camera data capture when enabled by the workflow.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64


   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

        ./isaaclab.sh train --rl_library sb3 --task Isaac-Cartpole --num_envs 64


Headless execution with off-screen render
"""""""""""""""""""""""""""""""""""""""""

Since the above command does not open an interactive visualizer, it is not possible to monitor behavior
live in a viewport window. To capture visual output during training, enable camera/sensor rendering
in the workflow and pass ``--video`` to record the agent behavior.

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run --extra video isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64 --video

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

        ./isaaclab.sh train --rl_library sb3 --task Isaac-Cartpole --num_envs 64 --video

The videos are saved to the ``logs/sb3/Isaac-Cartpole/<run-dir>/videos/train`` directory. You can open these videos
using any video player.

For tasks with on-scene cameras, you can also save the sensor image outputs directly during training
with ``--capture_env_sensors``. See :doc:`/source/how-to/capture_sensor_frames` for the available
options and output formats.

Interactive execution
"""""""""""""""""""""

.. currentmodule:: isaaclab

While the above two methods are useful for training the agent, they don't allow you to interact with the
simulation to see what is happening. In this case, run the
training command as follows:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code-block:: bash

        uv run isaaclab train --rl_library sb3 --task Isaac-Cartpole --num_envs 64 --viz kit

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code-block:: bash

        ./isaaclab.sh train --rl_library sb3 --task Isaac-Cartpole --num_envs 64 --viz kit

This will open the Kit visualizer window and you can see the agent training in the environment. However, this
can slow down the training process because interactive visual feedback is enabled. As a workaround, you
can switch between different render modes in the ``"Isaac Lab"`` window that is docked on the bottom-right
corner of the screen. To learn more about these render modes, please check the
:class:`sim.SimulationContext.RenderMode` class.

Viewing the logs
~~~~~~~~~~~~~~~~

On a separate terminal, you can monitor the training progress by executing the following command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         # execute from the root directory of the repository
         uv run python -m tensorboard.main --logdir logs/sb3/Isaac-Cartpole

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code:: bash

         # execute from the root directory of the repository
         ./isaaclab.sh -p -m tensorboard.main --logdir logs/sb3/Isaac-Cartpole

Playing the trained agent
~~~~~~~~~~~~~~~~~~~~~~~~~

Once the training is complete, you can visualize the trained agent by executing the following command:

.. tab-set::

   .. tab-item:: uv (Recommended)

      .. code:: bash

         # execute from the root directory of the repository
         uv run isaaclab play --rl_library sb3 --task Isaac-Cartpole --num_envs 32 --viz kit

   .. tab-item:: isaaclab.sh / isaaclab.bat

      .. code:: bash

         # execute from the root directory of the repository
         ./isaaclab.sh play --rl_library sb3 --task Isaac-Cartpole --num_envs 32 --viz kit

The above command will load the latest checkpoint from the ``logs/sb3/Isaac-Cartpole``
directory. You can also specify a specific checkpoint by passing the ``--checkpoint`` flag.

Evaluating an RL-Games checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the dedicated ``isaaclab evaluate`` workflow to collect a fixed number of inference-only
episodes from an RL-Games checkpoint. It uses the training environment configuration by default,
creates no optimizer, does not train the policy, and writes a separate evaluation run under
``logs/rl_games/<agent>/evaluations/``. When ``--track`` is enabled, TensorBoard-compatible metrics
such as ``rewards/iter``, ``episode_lengths/iter``, and task episode metrics are synchronized to an
independent Weights & Biases run.

For the FORGE PegInsert task, the training configuration has a 10 s episode horizon. With its 15 Hz
policy rate, the configured maximum is 150 policy steps. Factory currently terminates when its counter
reaches ``max_episode_length - 1``, so an ordinary episode contains 149 scored control steps.
Video lengths and intervals are measured in control steps, independently of episode boundaries:

.. code-block:: bash

   set -o pipefail
   uv run isaaclab evaluate \
     --rl_library rl_games \
     --task IsaacContrib-Forge-PegInsert-Direct \
     --checkpoint /path/to/Forge.pth \
     --num_envs 1 \
     --num_episodes 100 \
     --video --video_length 300 --video_interval 1500 \
     --seed 0 \
     --deterministic \
     --train_env_cfg \
     --viz kit \
     --track \
     --wandb-entity dylan1000-bht \
     --wandb-project-name isaac-forge \
     --wandb-name forge-peginsert-evaluation \
     agent.params.config.player.deterministic=True \
     2>&1 | tee /tmp/isaaclab-evaluate-$(date +%Y%m%d_%H%M%S).log

FORGE PegInsert local-Z tolerance penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

PegInsert includes a positive misalignment penalty with a negative reward scale. The angle
``theta`` is the directed angle between the two assets' local +Z axes in world coordinates, using
their true root XYZW quaternions. Position does not enter this term; quaternion sign, spinning an
asset around its own local Z, and rotating both assets together do not change it. Opposite axes
have an angle of 180 degrees. The PegInsert root-axis convention must be rechecked for replacement
assets; the term cannot be enabled on other FORGE tasks without independent geometric validation.

For a per-episode tolerance ``alpha`` and ``x = (theta - alpha) / (pi - alpha)``, the penalty is zero
when ``theta <= alpha`` and ``exp(kappa * (1 - x**(-2/3)))`` otherwise. Its value is in [0, 1];
the reward contribution is ``z_axis_align_scale * penalty``. Defaults are:

* ``env.task.z_axis_align_scale=-1.0`` for PegInsert, zero for other FORGE tasks. Positive scales
  are rejected. Zero skips reward addition but retains PegInsert diagnostics.
* ``env.task.z_axis_align_alpha_range=[10.0,20.0]`` in degrees, sampled once per episode.
  Use ``[15.0,15.0]`` for a fixed tolerance without consuming random numbers.
* ``env.task.z_axis_align_kappa=0.1756542881``. At alpha=15 degrees and theta=30 degrees, the
  penalty is approximately 0.5 and its default weighted contribution is approximately -0.5.

These are experimental starting values. The new penalty does not distinguish angles inside the
tolerance; existing keypoint rewards still provide more precise pose guidance. Actor/critic inputs
and action semantics are unchanged (24/61 observations and 7 actions); neither network observes alpha.
The hidden tolerance randomization can therefore increase value-estimation uncertainty. The original
roll/pitch action restriction, success criterion and termination mechanism are unchanged.

Tolerance sampling uses an independent, device-local Torch generator. Its recorded seed is
``(resolved_environment_seed + RANK) % 2**64``; the training seed may already contain its rank offset.
If no environment seed was resolved, ``torch.initial_seed()`` supplies the base seed without a draw.
Explicit ``env.reset(seed=...)`` restarts this stream before sampling; automatic reset does not reseed.
Replay requires the same seed, rank, device, environment count and reset order; it does not guarantee
bitwise deterministic physics. Invalid/zero asset quaternions raise an error rather than silently
producing perfect alignment.

The following 13 completed-episode metrics use the exact ``ZAxisAlign/`` prefix. Each is first
calculated for one episode using its actual number of scored steps, including the terminal step.
Training averages completed episodes within the current report window; evaluation averages retained
completed episodes cumulatively. P95 uses linear interpolation, and both P95 and max are calculated
per episode before averaging across episodes.

.. list-table:: Metric names below ``ZAxisAlign/``
   :header-rows: 1
   :widths: 55 45

   * - Name
     - Per-episode value
   * - ``z_axis_align_reward_mean``
     - Mean positive penalty (smaller is better).
   * - ``z_axis_align_weighted_mean``
     - Mean weighted contribution (closer to zero is better).
   * - ``z_axis_align_weighted_return``
     - Sum of weighted contributions, without length normalization.
   * - ``z_axis_angle_mean_deg``
     - Mean angle [deg].
   * - ``z_axis_angle_p95_deg``
     - 95th percentile angle [deg].
   * - ``z_axis_angle_final_deg``
     - Final scored angle before reset [deg].
   * - ``z_axis_aligned_fraction_5deg``
     - Fraction of steps with angle <=5 degrees.
   * - ``z_axis_align_reference_15deg_reward_mean``
     - Mean positive penalty recomputed with alpha=15 degrees and the run's kappa.
   * - ``z_axis_align_alpha_deg``
     - Actual episode tolerance [deg].
   * - ``z_axis_violation_fraction``
     - Fraction of steps with angle >alpha.
   * - ``z_axis_excess_mean_deg``
     - Mean max(theta-alpha, 0), including nonviolating steps [deg].
   * - ``z_axis_angle_max_deg``
     - Maximum episode angle [deg].
   * - ``z_axis_aligned_fraction_15deg``
     - Fraction of steps with angle <=15 degrees.

``logs_rew_z_axis_align`` and ``logs_rew_z_axis_align_weighted`` retain the existing direct-metric
``/frame``, ``/iter`` and ``/time`` tags. They describe a single vectorized control step, not an
episode average. The reference metric is diagnostic only, and comparisons across kappa values
should use geometric angle metrics instead. Continue checking original success and ``Episode/F_mean``
and ``Episode/F_max``; those force metrics describe smoothed sensor magnitudes.

The wrapper passes ``extras["z_axis_align"]`` through as a separate channel containing completed
``env_ids`` and a ``metrics`` mapping of short metric names to per-episode tensors. Training uses
:class:`~isaaclab_rl.rl_games.RlGamesAlgoObserver` in ordinary and PBT runs. Consumers handle tensor
device differences explicitly, and evaluation excludes completions beyond the per-environment quota.
New TensorBoard metric steps are training epochs or evaluation report indices; evaluation's W&B
history uses cumulative environment transitions. Existing tracking options control uploads.
Run manifests and tracked configs also retain source revision, dirty state, checkpoint path, and
the actual independent tolerance RNG seed.

Append these quoted Hydra overrides to train, play, or evaluate commands:

.. code-block:: bash

   'env.task.z_axis_align_scale=-1.0' \
   'env.task.z_axis_align_alpha_range=[10.0,20.0]' \
   'env.task.z_axis_align_kappa=0.1756542881'

For an existing Conda installation, activate the selected environment and run ``isaaclab`` directly
instead of ``uv run isaaclab``. The reward has no ``z_axis_align_sigma_range`` setting; the training
CLI's ``--sigma`` controls the policy distribution and is unrelated. Old policies remain structurally
loadable when observation order, normalization and action semantics match; changing rewards does not
guarantee arbitrary full-resume compatibility, and an old critic still predicts the old return.
Use fresh training for the first effectiveness comparison, with fixed-alpha and zero-scale controls.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _VecEnv API: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
.. _`stable_baselines3.common.vec_env.VecNormalize`: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
