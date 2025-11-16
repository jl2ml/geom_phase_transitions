#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate experiment CSVs into phase maps (heatmaps) for E1 and E2.

Inputs produced by experiments/exp_grokking.py:
  results/E1_phase_M{M}/summary.csv
  results/E2_width_M{M}/summary.csv

This script will:
  - read the summary CSV(s),
  - aggregate across seeds,
  - compute grokking rate and median T_g,
  - save aggregated CSV tables,
  - render heatmaps as PNGs (matplotlib-only).
Compatible with Python 3.6 + pandas 1.1.x + matplotlib 3.3.x.
"""

from __future__ import print_function
import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# -------------------------- utils --------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def find_summary(results_root, exp, M):
    # exp: E1 or E2
    if M is None:
        # try to find any matching folder
        cand_dirs = []
        for name in os.listdir(results_root):
            if exp == 'E1' and name.startswith('E1_phase_M'):
                cand_dirs.append(os.path.join(results_root, name))
            if exp == 'E2' and name.startswith('E2_width_M'):
                cand_dirs.append(os.path.join(results_root, name))
        if not cand_dirs:
            raise FileNotFoundError("No {} dirs found under {}".format(exp, results_root))
        # pick first for now
        d = cand_dirs[0]
        path = os.path.join(d, 'summary.csv')
        return path, d
    else:
        d = os.path.join(results_root, '{}_{}_M{}'.format('E1','phase',M) if exp=='E1' else '{}_{}_M{}'.format('E2','width',M))
        path = os.path.join(d, 'summary.csv')
        if not os.path.exists(path):
            raise FileNotFoundError("Summary CSV not found at {}".format(path))
        return path, d

def parse_filters(filter_str):
    # format: key=value;key2=value2
    if not filter_str:
        return {}
    filt = {}
    parts = filter_str.split(';')
    for p in parts:
        if not p:
            continue
        if '=' not in p:
            continue
        k,v = p.split('=',1)
        # try numeric conversion
        try:
            if '.' in v or 'e' in v or 'E' in v:
                vnum = float(v)
            else:
                vnum = int(v)
            filt[k] = vnum
        except Exception:
            filt[k] = v
    return filt

def apply_filters(df, filt):
    for k, v in filt.items():
        if k not in df.columns:
            print("Warning: filter key '{}' not in columns; skipping.".format(k))
            continue
        df = df[df[k] == v]
    return df

def pivot_agg_e1(df):
    # E1 axes: train_frac (columns) x weight_decay (rows)
    # Compute per-cell:
    #   - grokking_rate: fraction of seeds with Tg >= 0
    #   - median_Tg: median of Tg over seeds that grokked
    #   - median_best_acc: median of best_test_acc over seeds
    keys = ['train_frac', 'weight_decay']
    group = df.groupby(keys)
    # fraction grokked
    def grok_rate(g):
        vals = (g['grokking_time_steps'] >= 0).astype(float)
        return float(vals.mean()) if len(vals) > 0 else np.nan
    # median Tg over grokked
    def med_tg(g):
        vals = g.loc[g['grokking_time_steps'] >= 0, 'grokking_time_steps']
        return float(vals.median()) if len(vals) > 0 else np.nan
    # median best acc
    def med_best(g):
        vals = g['best_test_acc']
        return float(vals.median()) if len(vals) > 0 else np.nan

    rate = group.apply(grok_rate).reset_index(name='grokking_rate')
    medtg = group.apply(med_tg).reset_index(name='median_Tg')
    best  = group.apply(med_best).reset_index(name='median_best_acc')

    # pivot to 2D tables
    cols = sorted(df['train_frac'].unique())
    rows = sorted(df['weight_decay'].unique())
    # put 0 first then ascending
    rows = sorted(rows, key=lambda x: (x!=0, x))
    rate_p = rate.pivot(index='weight_decay', columns='train_frac', values='grokking_rate').reindex(index=rows, columns=cols)
    medtg_p = medtg.pivot(index='weight_decay', columns='train_frac', values='median_Tg').reindex(index=rows, columns=cols)
    best_p  = best.pivot(index='weight_decay', columns='train_frac', values='median_best_acc').reindex(index=rows, columns=cols)
    return rate_p, medtg_p, best_p

def pivot_agg_e2(df):
    # E2 axes: width (rows) x train_frac (columns)
    keys = ['width', 'train_frac']
    group = df.groupby(keys)
    def grok_rate(g):
        vals = (g['grokking_time_steps'] >= 0).astype(float)
        return float(vals.mean()) if len(vals) > 0 else np.nan
    def med_tg(g):
        vals = g.loc[g['grokking_time_steps'] >= 0, 'grokking_time_steps']
        return float(vals.median()) if len(vals) > 0 else np.nan
    def med_best(g):
        vals = g['best_test_acc']
        return float(vals.median()) if len(vals) > 0 else np.nan
    rate = group.apply(grok_rate).reset_index(name='grokking_rate')
    medtg = group.apply(med_tg).reset_index(name='median_Tg')
    best  = group.apply(med_best).reset_index(name='median_best_acc')

    cols = sorted(df['train_frac'].unique())
    rows = sorted(df['width'].unique())
    rate_p = rate.pivot(index='width', columns='train_frac', values='grokking_rate').reindex(index=rows, columns=cols)
    medtg_p = medtg.pivot(index='width', columns='train_frac', values='median_Tg').reindex(index=rows, columns=cols)
    best_p  = best.pivot(index='width', columns='train_frac', values='median_best_acc').reindex(index=rows, columns=cols)
    return rate_p, medtg_p, best_p

def save_tables(outdir, prefix, rate_p, medtg_p, best_p):
    ensure_dir(outdir)
    rate_p.to_csv(os.path.join(outdir, "{}_grokking_rate.csv".format(prefix)))
    medtg_p.to_csv(os.path.join(outdir, "{}_median_Tg.csv".format(prefix)))
    best_p.to_csv(os.path.join(outdir, "{}_median_best_acc.csv".format(prefix)))

def draw_heatmap(ax, Z, xticks, yticks, title, cmap='viridis', vmin=None, vmax=None, log=False, annotate=False, fmt="{:.2f}"):
    if log:
        # Using LogNorm for positive data; mask nonpositive
        data = np.array(Z, dtype=np.float64)
        mask = ~np.isfinite(data) | (data <= 0)
        data_masked = np.ma.array(data, mask=mask)
        im = ax.imshow(data_masked, aspect='auto', origin='lower',
                       norm=LogNorm(vmin=vmin or np.nanmin(data_masked), vmax=vmax or np.nanmax(data_masked)),
                       cmap=cmap)
    else:
        im = ax.imshow(Z, aspect='auto', origin='lower', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(len(xticks)))
    ax.set_xticklabels([str(x) for x in xticks], rotation=45, ha='right')
    ax.set_yticks(np.arange(len(yticks)))
    ax.set_yticklabels([str(y) for y in yticks])
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate:
        for i in range(len(yticks)):
            for j in range(len(xticks)):
                val = Z[i, j]
                if np.isfinite(val):
                    ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=7, color='white' if not log else 'black')

def compute_boundary(rate_p, boundary_frac):
    # For each row (fixed y), find smallest column where rate >= boundary_frac
    # Return lists of x and y index positions to plot a curve.
    y_vals = list(rate_p.index.values)
    x_vals = list(rate_p.columns.values)
    curve_x, curve_y = [], []
    for yi, yv in enumerate(y_vals):
        row = rate_p.iloc[yi].values
        pos = None
        for xi, rv in enumerate(row):
            try:
                if float(rv) >= boundary_frac:
                    pos = xi
                    break
            except Exception:
                continue
        if pos is not None:
            curve_x.append(pos)
            curve_y.append(yi)
    return curve_x, curve_y

def main():
    ap = argparse.ArgumentParser(description="Aggregate CSVs into phase maps")
    ap.add_argument("--results_root", type=str, default="results")
    ap.add_argument("--exp", type=str, choices=["E1","E2"], required=True)
    ap.add_argument("--M", type=int, default=None, help="modulus; if omitted, first matching dir is used")
    ap.add_argument("--outdir", type=str, default="results/plots")
    ap.add_argument("--filter", type=str, default="", help="optional filters like 'optimizer=adamw;representation=embed'")
    ap.add_argument("--boundary_frac", type=float, default=0.6, help="phase boundary threshold on grokking rate")
    ap.add_argument("--annot", action="store_true", help="annotate heatmap cells with numbers")
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    summary_csv, run_dir = find_summary(args.results_root, args.exp, args.M)
    df = pd.read_csv(summary_csv)

    # types
    for col in ["train_frac","weight_decay","width","grokking_time_steps","best_test_acc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # apply optional filters to isolate a subfamily (e.g., optimizer, representation, lr)
    filt = parse_filters(args.filter)
    if filt:
        df = apply_filters(df, filt)

    ensure_dir(args.outdir)

    if args.exp == 'E1':
        # pivot
        rate_p, medtg_p, best_p = pivot_agg_e1(df)
        save_tables(args.outdir, "E1_M{}".format(df['M'].iloc[0] if 'M' in df.columns else "unknown"),
                    rate_p, medtg_p, best_p)

        # Heatmaps
        xticks = list(rate_p.columns.values)
        yticks = list(rate_p.index.values)
        rate_arr = rate_p.values.astype(np.float64)
        medtg_arr = medtg_p.values.astype(np.float64)
        best_arr  = best_p.values.astype(np.float64)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        draw_heatmap(axs[0], rate_arr, xticks, yticks, title="E1: Grokking rate", cmap='viridis', vmin=0.0, vmax=1.0, annotate=args.annot, fmt="{:.2f}")
        finite_tg = medtg_arr[np.isfinite(medtg_arr) & (medtg_arr > 0)]
        vmin = np.nanmin(finite_tg) if finite_tg.size else 1.0
        vmax = np.nanmax(finite_tg) if finite_tg.size else 1.0
        draw_heatmap(axs[1], medtg_arr, xticks, yticks, title="E1: Median T_g (steps)", cmap='plasma', vmin=vmin, vmax=vmax, log=True, annotate=False)
        draw_heatmap(axs[2], best_arr, xticks, yticks, title="E1: Median best test acc", cmap='magma', vmin=0.0, vmax=1.0, annotate=args.annot, fmt="{:.2f}")

        # boundary curve overlay (on grokking rate)
        bx, by = compute_boundary(rate_p, boundary_frac=args.boundary_frac)
        axs[0].plot(bx, by, color='white', lw=2, marker='o', ms=3)
        axs[0].set_ylabel("weight_decay")
        for ax in axs[1:]:
            ax.set_yticklabels([])
        axs[0].set_xlabel("train_frac"); axs[1].set_xlabel("train_frac"); axs[2].set_xlabel("train_frac")

        out_png = os.path.join(args.outdir, "E1_phase_maps.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=args.dpi)
        print("Saved plot:", out_png)

    elif args.exp == 'E2':
        rate_p, medtg_p, best_p = pivot_agg_e2(df)
        save_tables(args.outdir, "E2_M{}".format(df['M'].iloc[0] if 'M' in df.columns else "unknown"),
                    rate_p, medtg_p, best_p)

        xticks = list(rate_p.columns.values)
        yticks = list(rate_p.index.values)
        rate_arr = rate_p.values.astype(np.float64)
        medtg_arr = medtg_p.values.astype(np.float64)
        best_arr  = best_p.values.astype(np.float64)

        fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
        draw_heatmap(axs[0], rate_arr, xticks, yticks, title="E2: Grokking rate", cmap='viridis', vmin=0.0, vmax=1.0, annotate=args.annot, fmt="{:.2f}")
        finite_tg = medtg_arr[np.isfinite(medtg_arr) & (medtg_arr > 0)]
        vmin = np.nanmin(finite_tg) if finite_tg.size else 1.0
        vmax = np.nanmax(finite_tg) if finite_tg.size else 1.0
        draw_heatmap(axs[1], medtg_arr, xticks, yticks, title="E2: Median T_g (steps)", cmap='plasma', vmin=vmin, vmax=vmax, log=True, annotate=False)
        draw_heatmap(axs[2], best_arr, xticks, yticks, title="E2: Median best test acc", cmap='magma', vmin=0.0, vmax=1.0, annotate=args.annot, fmt="{:.2f}")

        # boundary curve (width x train_frac)
        bx, by = compute_boundary(rate_p, boundary_frac=args.boundary_frac)
        axs[0].plot(bx, by, color='white', lw=2, marker='o', ms=3)
        axs[0].set_ylabel("width")
        for ax in axs[1:]:
            ax.set_yticklabels([])
        axs[0].set_xlabel("train_frac"); axs[1].set_xlabel("train_frac"); axs[2].set_xlabel("train_frac")

        out_png = os.path.join(args.outdir, "E2_phase_maps.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=args.dpi)
        print("Saved plot:", out_png)

if __name__ == "__main__":
    main()
