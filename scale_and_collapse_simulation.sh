# collapse prevention
uv run scripts/collapse_gradient_analysis.py --out-dir figures/collapse --mc 10 --r-steps 50
# scale-cost
uv run scripts/bench_cost.py --batches 1000,5000,10000,20000,30000,40000,50000 --dim 10000 --k 2500 --views 8