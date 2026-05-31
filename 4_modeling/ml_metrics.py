"""아파트 가격 예측 — 공통 회귀 평가 지표 (발표·노트북 11/12/13 공용)."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── 4지표 정의 ──────────────────────────────────────────────
METRIC_COLS = ['MAE_만원', 'RMSE_만원', 'R2_log', 'MAPE_pct']

METRIC_LABELS = {
    'MAE_만원': 'MAE (만원)',
    'RMSE_만원': 'RMSE (만원)',
    'R2_log': 'R² (log)',
    'MAPE_pct': 'MAPE (%)',
}

METRIC_FMT = {
    'MAE_만원': '{:,.0f}',
    'RMSE_만원': '{:,.0f}',
    'R2_log': '{:.3f}',
    'MAPE_pct': '{:.1f}%',
}

METRIC_ROUND = {
    'MAE_만원': 0,
    'RMSE_만원': 0,
    'R2_log': 4,
    'MAPE_pct': 1,
}

METRIC_BETTER = {
    'MAE_만원': 'lower',
    'RMSE_만원': 'lower',
    'R2_log': 'higher',
    'MAPE_pct': 'lower',
}


def eval_regression(actual_won, pred_log, y_log):
    """log 학습 → expm1 복원 후 4지표 dict 반환."""
    pred_won = np.expm1(pred_log)
    actual = np.asarray(actual_won)
    return {
        'MAE_만원': mean_absolute_error(actual, pred_won),
        'RMSE_만원': np.sqrt(mean_squared_error(actual, pred_won)),
        'R2_log': r2_score(y_log, pred_log),
        'MAPE_pct': (np.abs(actual - pred_won) / np.clip(actual, 1, None)).mean() * 100,
    }


def format_metric_line(metrics, sep=' | '):
    """한 줄 요약 — 학습 루프 print용."""
    return sep.join(
        f"{METRIC_LABELS[c].split(' ')[0]} {METRIC_FMT[c].format(metrics[c])}"
        for c in METRIC_COLS
    )


def round_metrics(df, extra=None):
    rnd = dict(METRIC_ROUND)
    if extra:
        rnd.update(extra)
    return df.round(rnd)


def build_err_df(actual, pred_log, y_log, cluster_name=None):
    pred_won = np.expm1(pred_log)
    actual = np.asarray(actual)
    df = pd.DataFrame({
        'actual': actual,
        'pred': pred_won,
        'pred_log': pred_log,
        'y_log': y_log,
        'abs_err': np.abs(actual - pred_won),
        'sq_err': (actual - pred_won) ** 2,
        'ape': np.abs(actual - pred_won) / np.clip(actual, 1, None) * 100,
    })
    if cluster_name is not None:
        df['cluster_name'] = cluster_name
    return df


def cluster_metrics(err_df):
    def _agg(g):
        return pd.Series({
            'MAE_만원': g['abs_err'].mean(),
            'RMSE_만원': np.sqrt(g['sq_err'].mean()),
            'R2_log': r2_score(g['y_log'], g['pred_log']),
            'MAPE_pct': g['ape'].mean(),
            '건수': len(g),
        })

    return (
        err_df.groupby('cluster_name')
        .apply(_agg, include_groups=False)
        .reset_index()
        .sort_values('MAE_만원')
    )


def plot_metrics_bars(df, x_col, title='', colors=None):
    """모델 비교용 — 4지표 가로 막대 (1행×4열)."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    x_labels = df[x_col].tolist()
    for ax, col in zip(axes, METRIC_COLS):
        vals = df[col].values
        c = colors if colors and len(colors) == len(df) else '#457b9d'
        ax.barh(x_labels, vals, color=c, edgecolor='white')
        ax.set_xlabel(METRIC_LABELS[col])
        ax.set_title(METRIC_LABELS[col], fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        fmt = METRIC_FMT[col]
        for i, v in enumerate(vals):
            ax.text(v, i, f' {fmt.format(v)}', va='center', fontsize=8)
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def plot_cluster_metrics_2x2(cluster_df, title='', cluster_colors=None):
    """클러스터별 4지표 2×2 막대."""
    if cluster_colors is None:
        cluster_colors = {'고광도': '#e63946', '중광도': '#f4a261', '저광도': '#457b9d'}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    bar_colors = [cluster_colors.get(c, 'gray') for c in cluster_df['cluster_name']]

    for ax, col in zip(axes.ravel(), METRIC_COLS):
        vals = cluster_df[col].values
        ax.bar(cluster_df['cluster_name'], vals, color=bar_colors, edgecolor='black')
        ax.set_ylabel(METRIC_LABELS[col])
        ax.set_title(METRIC_LABELS[col])
        fmt = METRIC_FMT[col]
        for i, v in enumerate(vals):
            ax.text(i, v, fmt.format(v), ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()


def summary_table(metrics_dict, label=''):
    """발표용 1행 4지표 DataFrame."""
    row = dict(metrics_dict)
    if label:
        row = {'구분': label, **row}
    return pd.DataFrame([row])


def print_metrics_delta(before, after, title=''):
    """튜닝·모델 비교 — 4지표 변화 한 번에 출력."""
    if title:
        print(title)
    for col in METRIC_COLS:
        b, a = before[col], after[col]
        diff = b - a if METRIC_BETTER[col] == 'lower' else a - b
        pct = diff / abs(b) * 100 if b else float('nan')
        sign = '+' if diff >= 0 else ''
        print(f"  {METRIC_LABELS[col]}: {METRIC_FMT[col].format(b)} → "
              f"{METRIC_FMT[col].format(a)}  ({sign}{pct:.2f}% 개선)")


def presentation_row(notebook, stage, eval_set, model, metrics, n=None):
    """발표용 요약 1행."""
    row = {
        '노트북': notebook,
        '단계': stage,
        '평가셋': eval_set,
        '모델': model,
        **{c: metrics[c] for c in METRIC_COLS},
    }
    if n is not None:
        row['건수'] = n
    return row


def save_presentation_summary(rows, path='ml_results_summary.csv', notebook=None):
    """발표용 CSV — 동일 노트북 재실행 시 해당 행만 갱신."""
    new_df = round_metrics(pd.DataFrame(rows))
    if Path(path).exists():
        old = pd.read_csv(path, encoding='utf-8-sig')
        if notebook and '노트북' in old.columns:
            old = old[old['노트북'] != notebook]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'발표용 요약 저장: {path} ({len(out)}행)')
    return out
