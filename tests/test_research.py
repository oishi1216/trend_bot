from __future__ import annotations

from pathlib import Path

from app.research_dashboard import load_research_dashboard


def _write_csv(path: Path, rows: list[tuple[str, float, float, float, float, float, float]]) -> None:
    path.write_text(
        'date,open,high,low,close,volume,value\n' + '\n'.join(','.join(map(str, row)) for row in rows),
        encoding='utf-8',
    )


def test_research_dashboard_builds_backtest(tmp_path):
    data = tmp_path / 'data'
    data.mkdir()
    rows = []
    for i in range(150):
        d = f'2026-01-{(i % 28) + 1:02d}'
        base = 100 + i * 0.5
        rows.append((d, base, base + 2, base - 2, base + 1, 1000, 100_000_000))
    _write_csv(data / '1301.csv', rows)
    dash = load_research_dashboard(str(data), str(tmp_path / 'cache.json'))
    assert dash.status['mode'] == 'annual_20_research_v0.6.0'
    assert dash.status['loaded'] is True
    assert dash.status['universe_size'] == 1
