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

Use the dedicated ``isaaclab-2.sh evaluate`` workflow to collect a fixed number of inference-only
episodes from an RL-Games checkpoint. It uses the training environment configuration by default,
creates no optimizer, does not train the policy, and writes a separate evaluation run under
``logs/rl_games/<agent>/evaluations/``. When ``--track`` is enabled, TensorBoard-compatible metrics
such as ``rewards/iter``, ``episode_lengths/iter``, and task episode metrics are synchronized to an
independent Weights & Biases run.

For the FORGE PegInsert task, the training configuration has a 10 s episode horizon. With its 15 Hz
policy rate, an episode is 150 policy steps. Therefore, a 300-step video contains two episodes and an
interval of 1500 steps records every ten episodes:

.. code-block:: bash

   set -o pipefail
   ./isaaclab-2.sh evaluate \
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

FORGE PegInsert z-axis alignment reward
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FORGE PegInsert adds a direction-only reward for aligning the held peg's local positive z axis with
the socket's local positive z axis. Both axes are transformed using the true asset root world
quaternions, in the current checkout's ``(x, y, z, w)`` convention. This follows the existing task's
root-frame alignment convention; check the asset geometry and transforms when replacing assets.
The reward does not measure lateral displacement or insertion depth. Spinning either asset about
its own local z axis leaves it unchanged, whereas rotating a tilted asset about the world z axis
can change the alignment.

For the axis-direction dot product :math:`c`, clamped to :math:`[-1, 1]`, the reward is

.. math::

   r_z = \exp\left(-\frac{1-c}{2\sin^2(\sigma/2)}\right),
   \qquad r = r_{\mathrm{original}} + w r_z.

The decay angle :math:`\sigma` gives :math:`r_z=\exp(-1)` at that misalignment; it is not a success
threshold. Opposite directions receive a low reward with the default decay angles. Existing
keypoint rewards, success criteria, observations, actions, and roll/pitch action restrictions are
preserved. The reward can encourage less tilt during contact, but does not add policy control over
roll or pitch.

The task fields are:

* ``env.task.z_axis_align_scale``: finite non-negative weight :math:`w`; defaults to ``0.1`` for
  PegInsert and ``0.0`` for the other FORGE tasks.
* ``env.task.z_axis_align_sigma_range``: two finite bounds in degrees, defaulting to ``[2.0,10.0]``.
  They must satisfy ``0 < lower <= upper <= 180`` and produce finite positive float32 denominators.
  Equal bounds select a fixed angle.

Each environment samples uniformly in degrees once per episode and stores the result in radians.
Sampling uses a separate generator seeded from the resolved environment seed, so it does not
advance the RNG used by existing domain randomization. Fixed bounds consume no random numbers.
The sampled angle is absent from both actor and critic observations: this is unobserved reward
randomization and may increase value-estimation uncertainty. Checkpoint input/output dimensions
are preserved, but reward code and configuration must still be restored separately from weights.

Append these Hydra overrides to an existing ``uv run isaaclab train``, ``play``, or ``evaluate``
command for the task:

.. code-block:: bash

   'env.task.z_axis_align_scale=0.1' 'env.task.z_axis_align_sigma_range=[2.0,10.0]'

Use ``'env.task.z_axis_align_sigma_range=[5.0,5.0]'`` for fixed 5-degree scoring. To recover the
previous total reward, use ``'env.task.z_axis_align_scale=0.0'``; alignment diagnostics remain
available. The original task is registered as ``IsaacContrib-Forge-PegInsert-Direct``.

With tracking enabled, new training and evaluation runs save both settings under ``env_cfg.task``
in W&B configuration. The raw and weighted instantaneous reward tags are
``logs_rew_z_axis_align/{frame,iter,time}`` and ``logs_rew_z_axis_align_weighted/{frame,iter,time}``.
The evaluator retains these values from the episode-completion step, rather than averaging them
over the trajectory. Evaluation ``iter`` counts reports; TensorBoard selects the matching suffix's
x-axis, while direct W&B evaluation logging uses frames as its step.

The following metrics include every normal environment step, including the terminal step, and
exclude simulation steps used to place assets during reset. Each metric is reduced per trajectory
and then averaged over the completed batch. Training averages completed reports within a logging
interval; evaluation averages them cumulatively. FORGE retains its synchronized-episode behavior.

.. list-table:: Alignment episode metrics (logged without an ``Episode/`` prefix)
   :header-rows: 1
   :widths: 55 45

   * - Metric
     - Definition
   * - ``ZAxisAlign/z_axis_align_reward_mean``
     - Mean unweighted alignment reward.
   * - ``ZAxisAlign/z_axis_align_weighted_mean``
     - Mean contribution :math:`w r_z` to total reward.
   * - ``ZAxisAlign/z_axis_align_weighted_return``
     - Sum of :math:`w r_z` over the trajectory.
   * - ``ZAxisAlign/z_axis_angle_mean_deg``
     - Mean axis misalignment [deg].
   * - ``ZAxisAlign/z_axis_angle_p95_deg``
     - Mean of per-trajectory P95 angles [deg], not a pooled P95.
   * - ``ZAxisAlign/z_axis_angle_final_deg``
     - Last pre-reset axis misalignment [deg].
   * - ``ZAxisAlign/z_axis_aligned_fraction_5deg``
     - Fraction of steps with misalignment at most 5 degrees.
   * - ``ZAxisAlign/z_axis_align_reference_5deg_mean``
     - Mean reward recomputed with fixed 5-degree decay; diagnostic only.
   * - ``ZAxisAlign/z_axis_align_sigma_deg``
     - Sampled decay angle [deg].

Compare the previous reward (weight zero), the default randomized reward, and a fixed-5-degree
ablation with matching seeds and training budgets. Evaluate checkpoints under identical reward
settings and use the angle metrics and fixed-5-degree reference score alongside insertion success,
``Episode/F_mean``, and ``Episode/F_max``. The latter is the peak sampled smoothed force within an
episode, not a peak over all physics substeps. Higher total return alone is insufficient evidence
because the added term is positive; already parallel axes may make it nearly constant.

.. _Stable-Baselines3: https://stable-baselines3.readthedocs.io/en/master/
.. _VecEnv API: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecenv-api-vs-gym-api
.. _`stable_baselines3.common.vec_env.VecNormalize`: https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#vecnormalize
.. _RL-Games: https://github.com/Denys88/rl_games
.. _RSL-RL: https://github.com/leggedrobotics/rsl_rl
.. _SKRL: https://skrl.readthedocs.io
