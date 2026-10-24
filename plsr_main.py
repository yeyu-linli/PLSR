#!/usr/bin/env python3
"""
PLSR attribution of hydrological-component changes to land-use change (Ganjiang River Basin).

Input : CSV with one row per sub-basin and the columns
        Subbasin, AGRL, FRST, PAST, WATR, URBN, SURQ, WYLD, GWQ, LATQ, PERC, ET
        (five land-use change rates in %; six component changes in mm/yr, 2018 minus 1980).
Output: model_summary.csv (Table 4), coefficients.csv (Tables 5 and 6).

Method
  * response groups: SURQ+WYLD, GWQ+PERC, ET, LATQ
  * studentized-residual screening at 1.7 sigma (at most three rounds, applied per group)
  * seven-fold cross-validated Q2; the number of latent components follows the Wold rule
    (a component is kept only while it raises Q2 by more than five percentage points)
  * 2,000-resample bootstrap for the 95% intervals; VIP for predictor importance

Usage: python plsr_main.py example_input.csv
"""
import sys
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

LU = ['AGRL', 'FRST', 'PAST', 'WATR', 'URBN']
LUN = {'AGRL': 'Cropland', 'FRST': 'Forestland', 'PAST': 'Grassland',
       'WATR': 'Water body', 'URBN': 'Construction land'}
ALL = ['SURQ', 'WYLD', 'GWQ', 'LATQ', 'PERC', 'ET']
GROUPS = [('PLSR1', 'SURQ and WYLD', ['SURQ', 'WYLD']),
          ('PLSR2', 'GWQ and PERC', ['GWQ', 'PERC']),
          ('PLSR3', 'ET', ['ET']),
          ('PLSR4', 'LATQ', ['LATQ'])]
THRESH, MAXROUND, NFOLD, NBOOT, SEED, WOLD = 1.7, 3, 7, 2000, 42, 0.05


def z(M):
    return (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-12)


def screen(X, Y, ncomp, thresh=THRESH, maxround=MAXROUND):
    """iteratively drop sub-basins whose studentized residuals exceed `thresh`."""
    keep = np.ones(len(X), dtype=bool)
    for _ in range(maxround):
        idx = np.where(keep)[0]
        if len(idx) < 15:
            break
        pred = PLSRegression(n_components=ncomp, scale=True).fit(X[idx], Y[idx]).predict(X[idx])
        res = Y[idx] - pred
        bad = np.abs(res / (res.std(axis=0) + 1e-12)).max(axis=1) > thresh
        if not bad.any():
            break
        keep[idx[bad]] = False
    return keep


def cv(Xk, Yz, nc):
    """seven-fold cross-validation: returns (Q2, PRESS)."""
    kf = KFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    Xz, pr, sst = z(Xk), 0.0, 0.0
    for tr, te in kf.split(Xk):
        m = PLSRegression(n_components=nc, scale=False).fit(Xz[tr], Yz[tr])
        pr += np.sum((Yz[te] - m.predict(Xz[te])) ** 2)
        sst += np.sum((Yz[te] - Yz.mean(axis=0)) ** 2)
    return 1.0 - pr / sst, np.sqrt(pr / (len(Xk) * Yz.shape[1]))


def fit_stats(Xk, Yz, nc):
    Xz = z(Xk)
    m = PLSRegression(n_components=nc, scale=False).fit(Xz, Yz)
    pred = m.predict(Xz)
    q2v, press = cv(Xk, Yz, nc)
    return dict(r2=1.0 - np.sum((Yz - pred) ** 2) / np.sum((Yz - Yz.mean(axis=0)) ** 2),
                q2=q2v, press=press,
                rmse=float(np.sqrt(np.mean((Yz - pred) ** 2))),
                mae=float(np.mean(np.abs(Yz - pred))))


def vip_of(pls):
    w, t, ql = pls.x_weights_, pls.x_scores_, pls.y_loadings_
    s = np.sum(t ** 2, axis=0) * np.sum(ql ** 2, axis=0)
    return np.sqrt(np.array([len(w) * np.sum(s * w[j, :] ** 2) / np.sum(s) for j in range(len(w))]))


def coef_of(pls):
    c = np.atleast_2d(pls.coef_)
    return c.T if c.shape[0] != len(LU) else c


def main(path):
    dat = pd.read_csv(path)
    X = dat[LU].to_numpy(float)
    Y = dat[ALL].to_numpy(float)
    summ, coefs, models = [], [], {}
    for gid, glab, comps in GROUPS:
        cols = [ALL.index(k) for k in comps]
        Yi = Y[:, cols]
        best = None
        for ncf in (1, 2, 3):
            keep = screen(X, Yi, ncf)
            if keep.sum() < 15:
                continue
            Yk = z(Yi[keep])
            for nc in range(1, 6):
                s, _ = cv(X[keep], Yk, nc)
                if best is None or s > best[0]:
                    best = (s, ncf, nc, keep.copy())
        if best is None:
            raise SystemExit('screening removed too many sub-basins for %s' % glab)
        keep = best[3]
        Xk, Yk = X[keep], Yi[keep]
        Yz = z(Yk)
        qs = [cv(Xk, Yz, nc)[0] for nc in range(1, 6)]
        nc = 1
        for k in range(1, 5):
            if qs[k] - qs[k - 1] > WOLD:
                nc = k + 1
            else:
                break
        # explained variability of a five-component fit (values beyond nc are not retained)
        pls5 = PLSRegression(n_components=5, scale=False).fit(z(Xk), Yz)
        t, px, ql = pls5.x_scores_, pls5.x_loadings_, pls5.y_loadings_
        ssx, ssy = np.sum(z(Xk) ** 2), np.sum(Yz ** 2)
        ey = [np.sum(t[:, c] ** 2) * np.sum(ql[:, c] ** 2) / ssy * 100 for c in range(5)]
        ex = [np.sum(t[:, c] ** 2) * np.sum(px[:, c] ** 2) / ssx * 100 for c in range(5)]
        cy, cx_ = np.cumsum(ey), np.cumsum(ex)
        st = fit_stats(Xk, Yz, nc)
        for c in range(5):
            s_c = fit_stats(Xk, Yz, c + 1)
            summ.append([gid, glab, int(keep.sum()), c + 1,
                         '%.1f' % ey[c], '%.1f' % cy[c], '%.1f' % cx_[c],
                         '%.3f' % s_c['press'], '%.3f' % s_c['q2'],
                         '%.3f' % s_c['rmse'], '%.3f' % s_c['mae']])
        print('%s (%s): N=%d  nc=%d  R2=%.3f  Q2=%.3f  PRESS=%.3f'
              % (gid, glab, keep.sum(), nc, st['r2'], st['q2'], st['press']))
        models[gid] = (Xk, Yz, nc)
        pls = PLSRegression(n_components=nc, scale=False).fit(z(Xk), Yz)
        coef, vip = coef_of(pls), vip_of(pls)
        rng = np.random.default_rng(SEED)
        boot = np.zeros((NBOOT, len(LU), Yz.shape[1]))
        Xz, n = z(Xk), len(Xk)
        for b in range(NBOOT):
            idx = rng.integers(0, n, n)
            boot[b] = coef_of(PLSRegression(n_components=nc, scale=False).fit(Xz[idx], Yz[idx]))
        for j, comp in enumerate(comps):
            c = coef[:, j]
            ct = np.abs(c) / np.sum(np.abs(c)) * 100
            for i, lu in enumerate(LU):
                lo, hi = np.percentile(boot[:, i, j], [2.5, 97.5])
                coefs.append([gid, comp, LUN[lu], '%.3f' % c[i], '%.1f' % ct[i],
                              '%.3f' % lo, '%.3f' % hi, '%.2f' % vip[i],
                              'yes' if (lo > 0 or hi < 0) else 'no'])
    pd.DataFrame(summ, columns=['Model', 'Components', 'N', 'nc', 'explY', 'cumY', 'cumX',
                                'PRESS', 'Q2', 'RMSE', 'MAE']).to_csv('model_summary.csv', index=False)
    pd.DataFrame(coefs, columns=['Model', 'Component', 'Land use', 'Coefficient',
                                 'Contribution', 'CI_low', 'CI_high', 'VIP',
                                 'Significant']).to_csv('coefficients.csv', index=False)
    print('\nwritten model_summary.csv (Table 4) and coefficients.csv (Tables 5-6)')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'example_input.csv')
