# usage: runeval.sh MODEL_DIR TAG ; runs pair+stress and bench/10+stress, seeds 0..2, strict
M=$1; T=$2; cd "$(cd "$(dirname "$0")/../.." && pwd)"; mkdir -p results
for s in 0 1 2; do .venv/Scripts/python eval_local.py --model $M --seed $s --strict --stress --out results/J_${T}_pair_s$s.json > /dev/null 2>results/J_${T}_pair_s$s.err; done
for s in 0 1 2; do .venv/Scripts/python eval_local.py --model $M --seed $s --strict --stress --labeled-scope benchmark --max-pairs-per-bench 10 --out results/J_${T}_b10_s$s.json > /dev/null 2>results/J_${T}_b10_s$s.err; done
echo finished $T
