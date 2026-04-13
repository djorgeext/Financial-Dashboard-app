"""
Static safety checks for dashboard front-end rendering.
"""
import re
from pathlib import Path


DASHBOARD_PATH = Path(__file__).resolve().parent.parent / "dashboard.html"


def test_draw_news_avoids_innerhtml_for_external_fields():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    match = re.search(r"function drawNews\(d\) \{([\s\S]*?)\n\s*\}\n\s*</script>", html)
    assert match is not None, "drawNews function not found"

    body = match.group(1)
    assert "innerHTML = html" not in body
    assert "textContent" in body
    assert "createElement" in body


def test_draw_options_avoids_innerhtml_string_building_for_rows():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    match = re.search(r"function drawOptions\(d\) \{([\s\S]*?)\n\s*\}\n\s*function drawNews", html)
    assert match is not None, "drawOptions function not found"

    body = match.group(1)
    assert "let html =" not in body
    assert "innerHTML = html" not in body
    assert "html += '<td>'" not in body
    assert "createElement('table')" in body
    assert "createElement('td')" in body
    assert "textContent" in body
    assert "table-wrap" in body
    assert "results-table" in body
    assert "suggested_options" in body
    assert "rowsToRender" in body
    assert "!rows.length && !suggestedOptions.length" in body
    assert "Barrera a 10 horas" in body
    assert "'Barrera'," not in body


def test_dashboard_neutral_forecast_state_hides_chart_and_table_with_banner_message():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "function isNeutralForecastOrUnavailable(d)" in html
    assert "function showNeutralOrUnavailableState(message)" in html
    assert "showNeutralOrUnavailableState('Neutral forecast or not available');" in html
    assert "neutral.className = 'no-data';" in html
    assert "neutral.textContent = 'Neutral forecast or not available';" in html


def test_chart_uses_non_fragmented_continuous_axis_with_day_hour_ticks():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "rangebreaks" not in html
    assert "type: 'linear'" in html
    assert "tickmode: 'array'" in html
    assert "timeZone: 'America/New_York'" in html
    assert "function formatChartTickLabel(timestamp)" in html
    assert "return day + ' ' + month + '\\n' + hour + ':' + minute;" in html
    assert "nticks: 4" not in html


def test_chart_uses_initial_tick_strategy_and_adaptive_zoom_densification():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "const initialTickConfig = buildChartTickConfig(indexLabels, null, null, 4);" in html
    assert "function buildAdaptiveTickTarget(visibleCount)" in html
    assert "graph.on('plotly_relayout', relayoutData => {" in html
    assert "const hasExplicitRange = Object.prototype.hasOwnProperty.call(relayoutData || {}, 'xaxis.range[0]')" in html
    assert "Plotly.relayout(graph, {" in html


def test_chart_draws_ohlc_average_and_connects_forecast_from_last_average_point():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "const historicalAvgLine = {" in html
    assert "ohlcAvg: (open + high + low + close) / 4" in html
    assert "const forecastX = [lastHistoricalIndex].concat(forecastIndices);" in html
    assert "const forecastY = [lastHistoricalPoint.ohlcAvg].concat(forecastPoints.map(point => point.forecast));" in html
    assert "line: { color: '#22d3ee', width: 2, dash: 'dash' }" in html


def test_last_updated_uses_new_york_timezone_formatter():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "const lastUpdatedFormatter = new Intl.DateTimeFormat('en-US', {" in html
    assert "timeZone: 'America/New_York'" in html
    assert "const formattedLastUpdate = lastUpdatedFormatter.format(new Date());" in html
    assert "new Date().toLocaleTimeString()" not in html


def test_dashboard_uses_white_background_and_black_text_for_table_and_news_summary():
    html = DASHBOARD_PATH.read_text(encoding="utf-8")

    assert "background: #ffffff;" in html
    assert "color: #000000;" in html
    assert ".results-table th { background: #e2e8f0; color: #000000; position: sticky; top: 0; }" in html
    assert "#newsContainer { color: #000000; }" in html
    assert "#newsContainer .no-data { color: #000000; }" in html
    assert ".news-list, .news-item { color: #000000; }" in html
