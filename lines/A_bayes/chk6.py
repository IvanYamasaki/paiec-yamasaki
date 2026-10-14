import sys; sys.path.insert(0, 'v6'); import model as M, sim
T = sim.trajectories(0); o = T['obs']
print(len(M.NOTES))
for b in ['multi_swebench', 'real_webagents', 'researchcodebench']:
    M.local_fit(o[o.bench != b].reset_index(drop=True), T['subjects'], T['items'])
    print(b, {k: round(M.P[k], 3) for k in ['beta', 'sdr2', 'vx', 'xbar', 'sd2', 'rho', 'm', 's2mu']})
