Fixed
^^^^^

* Fixed FORGE PegInsert velocity observations by differencing consecutive noisy poses with an
  independent history and using full noisy rotations before the existing observation projection.
  Preserved observation and action interfaces, controller and critic velocities, and other tasks.
  Existing checkpoints remained loadable, but policies trained with the previous velocity inputs
  require retraining to assess learning improvements with the corrected observations.
