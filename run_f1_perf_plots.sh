#!/bin/zsh
cd "/Users/macbook2024/Dropbox/AAA Backup/A Working/junk5"
python3 output.py \
  --in_xlsx "output_F1_detailed/predictions_f1_detailed.xlsx" \
  --out_xlsx "output_F1_detailed/set_ranker_perf_F1_detailed.xlsx" \
  --out_pdf "output_F1_detailed/set_ranker_perf_plots_F1_detailed.pdf" \
  --baseline RawRank