Added
^^^^^

* Added ``RlGamesAlgoObserver`` and evaluation support for completed-episode
  ``ZAxisAlign/`` metrics, weighted by retained episode count, with report flushing
  asynchronous writer shutdown and source/checkpoint/random-seed provenance in run metadata. Existing metrics
  retained their reporting behavior; new TensorBoard alignment metrics used epoch
  or evaluation report index as their step.
