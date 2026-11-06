Fixed
^^^^^

* Fixed the FORGE PegInsert observation projection for XYZW quaternions by zeroing
  the Z and W components while preserving X and Y. Full-rotation velocity estimates
  and the GearMesh and NutThread tasks were unchanged. Observation and action shapes
  remained compatible with existing checkpoints, but PegInsert policies require
  retraining to account for the corrected orientation inputs.
