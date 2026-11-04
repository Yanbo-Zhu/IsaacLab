Added
^^^^^

* Added a cube-root local-Z alignment tolerance penalty and thirteen per-episode diagnostics
  for FORGE PegInsert, without changing policy inputs or actions. The penalty defaulted to a
  five-degree tolerance and a weight of -1; setting ``env.task.z_axis_align_scale=0`` restored
  the previous total reward while retaining diagnostics. Other FORGE tasks remained disabled.
