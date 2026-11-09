import sys, numpy as np, torch
import reader as R
from sim import Data
iv, h = sys.argv[1], int(sys.argv[2])
X = Data("cuda", iv)
ret, stop = R.labels(X, h)
days = list(range(1, X.D)); n_test = len(days) // 4; te = days[-n_test:]
f, r, s, v, e = R.build(X, te, ret, stop)
m = R.Reader(f.shape[-1]).cuda(); m.load_state_dict(torch.load(f"runs/reader_{iv}_h{h}.pt")); m.eval()
with torch.no_grad():
    o = torch.cat([m(f[d]) for d in range(len(te))]).reshape(*f.shape[:3], 3)
vol = X.F[te][..., 10 if False else 0].abs()
atr = f[..., 9] - f[..., 8]
mk = v
y, sc, st = r[mk].cpu().numpy(), o[..., 0][mk].cpu().numpy(), s[mk].cpu().numpy()
rng = atr[mk].cpu().numpy()
qs = np.quantile(rng, np.linspace(0, 1, 6))
print(iv, "day range so far (high/low spread) quintiles -> all mean | top5% by reader mean, win | stop rate")
for i in range(5):
    b = (rng >= qs[i]) & (rng <= qs[i + 1])
    yy, ss = y[b], sc[b]
    k = max(1, int(0.05 * len(yy))); top = np.argsort(-ss)[:k]
    print(f"Q{i+1} range~{np.median(rng[b]):.2f}  all {yy.mean()*100:+.2f}%  top5% {yy[top].mean()*100:+.2f}% win {(yy[top]>0).mean():.2f}  stop {st[b].mean():.2f}")
