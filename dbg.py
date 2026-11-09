import torch, numpy as np, sim, train
from sim import Data, Env
data = Data("cuda")
tr, va, te = train.splits(data)
f = train.baselines.__code__
N = data.N
d = va[0]
env = Env(data, 1, [d])
per, g, m = env.reset(torch.tensor([d], device="cuda"), torch.tensor([10000.0], dtype=torch.float64, device="cuda"))
rng = np.random.default_rng(0)
while True:
    b = int(env.b[0]); pos = int(env.pos[0])
    k = N
    if pos < N:
        u = float(data.C[env.d[0], b, pos]) / float(env.entry[0]) - 1
        k = N if (u <= -0.03 or u >= 0.05) else pos
    elif 6 <= b <= 60 and int(env.trades[0]) < 6:
        rday = per[0, :, 4] / 10
        cand = m[0, :-1] & (rday > 0.05) & (rday < 0.20) & (per[0, :, 6] > 1.0)
        idx = torch.nonzero(cand).flatten()
        if len(idx): k = int(idx[rday[idx].argmax()])
    if not bool(m[0, k]): k = N
    pre = (float(env.cash[0]), int(env.pos[0]), float(env.sh[0]))
    _, done = env.step(torch.tensor([k], device="cuda"))
    post = (float(env.cash[0]), int(env.pos[0]), float(env.sh[0]))
    if pre != post:
        j = post[1] if post[1] < N else pre[1]
        print(b, data.syms[j], "O", float(data.O[d, b+1, j]), "C", float(data.C[d, b, j]), "V", float(data.V[d, b, j]), "pc", float(data.pc[d, j]), pre, "->", post, "mark", float(env.mark(env.b)[0]))
    if bool(done[0]): break
    per, g, m = env.obs()
print("end cash", float(env.cash[0]), "pos", int(env.pos[0]))
