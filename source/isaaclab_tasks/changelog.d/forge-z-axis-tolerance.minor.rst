Added
^^^^^

* Added a local-Z tolerance penalty to FORGE PegInsert, with per-episode independent
  tolerance sampling, explicit reseeding, and thirteen completed-episode diagnostics.
  PegInsert now defaults to a positive penalty multiplied by a negative scale of -1.0.
  Set ``env.task.z_axis_align_scale=0.0`` to retain the previous reward with diagnostics;
  use ``env.task.z_axis_align_alpha_range=[15.0,15.0]`` for a fixed tolerance [deg].
  Observation/action interfaces, success criteria, and termination behavior remained unchanged.
