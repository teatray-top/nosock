import sys, numpy as np, torch
import reader as R
from sim import Data
iv, h, st = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
R.STOP = st
X = Data("cuda", iv)
ret, stop = R.labels(X, h)
days = list(range(1, X.D)); n_test = len(days) // 4; te = days[-n_test:]
f, r, s, v, e = R.build(X, te, ret, stop)
m = R.Reader(f.shape[-1]).cuda(); m.load_state_dict(torch.load(f"runs/reader_{iv}_h{h}_s{st}.pt")); m.eval()
with torch.no_grad():
    o = torch.cat([m(f[d]) for d in range(len(te))]).reshape(*f.shape[:3], 3)
y = r[v].cpu().numpy(); sc = o[..., 0][v].cpu().numpy(); F = f[v].cpu().numpy()
pc = X.pc[te][:, None, :].expand(r.shape)[v].cpu().numpy()
pt = X.prev_turn[te][:, None, :].expand(r.shape)[v].cpu().numpy()
names = ["r1","r3","r6","r12","rday","gap","volr","cumturn","hidist","lodist","tick%","prevturn","t"]
k = int(0.01 * len(y)); top = np.argsort(-sc)[:k]
print("feature medians: all vs top1%")
for i, nm in enumerate(names):
    j = i if i < 12 else 24
    print(f"{nm:9s} {np.median(F[:, j]):+.3f} {np.median(F[top, j]):+.3f}")
print("price med", np.median(pc), np.median(pc[top]), "prev turnover med(억)", np.median(pt)/1e8, np.median(pt[top])/1e8)
liq = pt >= 3e8
for nm, mk in (("liquid>=3억", liq), ("illiquid", ~liq)):
    yy, ss = y[mk], sc[mk]; kk = max(1, int(0.01 * len(yy))); t = np.argsort(-ss)[:kk]
    print(nm, "n", mk.sum(), "mean %+.2f%%" % (yy.mean()*100), "top1 %+.2f%% win %.2f" % (yy[t].mean()*100, (yy[t] > 0).mean()))
