Added
^^^^^

* Added independent ``ZAxisAlign/`` episode metrics to RL-Games training and evaluation,
  with equal episode weighting, evaluation quota filtering, and writer shutdown before
  environment teardown. Existing metrics and PPO optimization were preserved.
