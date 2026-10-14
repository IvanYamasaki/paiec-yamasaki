# Missing local seeds for the report (TODO items 5, 6, 8). One process at a time.
# Same flags as lines/J_robusto/runeval.sh; outputs stay in report/runs/.
cd "$(dirname "$0")/../.." || exit 1
P=.venv/Scripts/python
O=report/runs
run() { tag=$1; shift; [ -f $O/$tag.json ] && return; echo "start $tag $(date +%T)"; $P eval_local.py "$@" --strict --stress --out $O/$tag.json > /dev/null 2> $O/$tag.err; echo "end $tag $(date +%T) rc=$?"; }
B10="--labeled-scope benchmark --max-pairs-per-bench 10"
for s in 1 2; do run J3_pair_s$s --model lines/J_robusto/submit_J3_hib --seed $s; done
for s in 1 2; do run J4_pair_s$s --model lines/J_robusto/submit_J4_rasch --seed $s; done
run v6_b10_s1 --model lines/A_bayes/v6 --seed 1 $B10
for s in 0 1 2; do run J3_b10_s$s --model lines/J_robusto/submit_J3_hib --seed $s $B10; done
for s in 0 1 2; do run J4_b10_s$s --model lines/J_robusto/submit_J4_rasch --seed $s $B10; done
for m in 20 40; do for s in 0 1 2; do run J2_b${m}_s$s --model lines/J_robusto/submit_J2_min --seed $s --labeled-scope benchmark --max-pairs-per-bench $m; done; done
echo ALLDONE
