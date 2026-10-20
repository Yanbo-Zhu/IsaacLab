Added
^^^^^

* Added support for sparse FORGE alignment episode reports under the ``ZAxisAlign/`` namespace in
  RL-Games training and inference-only evaluation, while preserving existing ``Episode/`` metrics.

Fixed
^^^^^

* Flushed completed alignment reports so short training runs retained every metric on exit.
