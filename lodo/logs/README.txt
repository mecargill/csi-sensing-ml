log_2026-11-21_12-20-57 has LSTM correct results. resnet is incorrect, worse than it should be. i reran just the resnet after fixing the bug, that's in log_2026-11-21_18-16-02. based on those .pt files, I generated confidence_gaps_2026-11-22_17-14-07_974996.

Summary of which logs go together:

log_2026-11-21_12-20-57 (lstm only is correct)
log_2026-11-21_18-16-02 (only resnet run, correct)
confidence_gaps_2026-11-22_17-14-07_974996 (generated from models output by the above 2 logs (using correct resnet models))
