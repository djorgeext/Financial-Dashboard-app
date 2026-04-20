"""
Static safety checks for dashboard front-end rendering.
"""
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_PATHS = [
    PROJECT_ROOT / "frontend" / "templates" / "dashboard.html",
    PROJECT_ROOT / "dashboard.html",
]
CSS_PATH = PROJECT_ROOT / "frontend" / "static" / "css" / "dashboard.css"


def _read_dashboard_html() -> str:
    for dashboard_path in DASHBOARD_PATHS:
        if dashboard_path.exists():
            return dashboard_path.read_text(encoding="utf-8")
    raise AssertionError("dashboard.html not found in expected locations")


def _read_dashboard_css() -> str:
    assert CSS_PATH.exists(), f"CSS file not found: {CSS_PATH}"
    return CSS_PATH.read_text(encoding="utf-8")


def test_draw_news_avoids_innerhtml_for_external_fields():
    html = _read_dashboard_html()

    match = re.search(r"function drawNews\(d\) \{([\s\S]*?)\n\s*\}\n\s*</script>", html)
    assert match is not None, "drawNews function not found"

    body = match.group(1)
    assert "innerHTML = html" not in body
    assert "textContent" in body
    assert "createElement" in body


def test_draw_options_avoids_innerhtml_string_building_for_rows():
    html = _read_dashboard_html()

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
    html = _read_dashboard_html()

    assert "function isNeutralForecastOrUnavailable(d)" in html
    assert "function showNeutralOrUnavailableState(message)" in html
    assert "showNeutralOrUnavailableState('Neutral forecast or not available');" in html
    assert "neutral.className = 'no-data';" in html
    assert "neutral.textContent = 'Neutral forecast or not available';" in html


def test_chart_uses_non_fragmented_continuous_axis_with_day_hour_ticks():
    html = _read_dashboard_html()

    assert "rangebreaks" not in html
    assert "type: 'linear'" in html
    assert "tickmode: 'array'" in html
    assert "timeZone: 'America/New_York'" in html
    assert "function formatChartTickLabel(timestamp)" in html
    assert "return day + ' ' + month + '\\n' + hour + ':' + minute;" in html
    assert "nticks: 4" not in html


def test_chart_uses_initial_tick_strategy_and_adaptive_zoom_densification():
    html = _read_dashboard_html()

    assert "const initialTickConfig = buildChartTickConfig(indexLabels, null, null, 4);" in html
    assert "function buildAdaptiveTickTarget(visibleCount)" in html
    assert "graph.on('plotly_relayout', relayoutData => {" in html
    assert "const hasExplicitRange = Object.prototype.hasOwnProperty.call(relayoutData || {}, 'xaxis.range[0]')" in html
    assert "Plotly.relayout(graph, {" in html


def test_chart_draws_ohlc_average_and_connects_forecast_from_last_average_point():
    html = _read_dashboard_html()

    assert "const historicalAvgLine = {" in html
    assert "ohlcAvg: (open + high + low + close) / 4" in html
    assert "const forecastX = [lastHistoricalIndex].concat(forecastIndices);" in html
    assert "const forecastY = [lastHistoricalPoint.ohlcAvg].concat(forecastPoints.map(point => point.forecast));" in html
    assert "line: { color: '#22d3ee', width: 2, dash: 'dash' }" in html


def test_last_updated_uses_new_york_timezone_formatter():
    html = _read_dashboard_html()

    assert "const lastUpdatedFormatter = new Intl.DateTimeFormat('en-US', {" in html
    assert "timeZone: 'America/New_York'" in html
    assert "const formattedLastUpdate = lastUpdatedFormatter.format(new Date());" in html
    assert "new Date().toLocaleTimeString()" not in html


def test_dashboard_uses_external_css_and_no_embedded_style_block():
    html = _read_dashboard_html()

    assert '<link rel="stylesheet" href="/static/css/dashboard.css">' in html
    assert "<style>" not in html


def test_dashboard_css_uses_white_background_black_text_and_light_gray_buttons():
    css = _read_dashboard_css()

    assert "body {" in css
    assert "background: #ffffff;" in css
    assert "color: #000000;" in css
    assert "--button-top: #d9d9d9;" in css
    assert "--button-bottom: #bfbfbf;" in css
    assert "button," in css
    assert "box-shadow: var(--button-shadow);" in css
    assert ".run-button:hover" in css
    assert ".run-button:active" in css


def test_dashboard_css_aligns_main_content_frames_with_shared_page_width():
    css = _read_dashboard_css()

    assert ".header," in css
    assert ".status-bar," in css
    assert ".controls," in css
    assert ".section {" in css
    assert "width: max(100%, var(--content-max-width));" in css
    assert "#chartContainer" in css


def test_dashboard_css_maintains_black_text_for_table_and_news_sections():
    css = _read_dashboard_css()

    assert ".results-table th" in css
    assert "background: #e2e8f0;" in css
    assert "#newsContainer {" in css
    assert "#newsContainer .no-data {" in css
    assert ".news-list," in css


def test_dashboard_css_typography_uses_global_html_scale_and_non_inflated_sections():
    css = _read_dashboard_css()

    assert "html {" in css
    assert "font-size: 106.25%;" in css

    css_values = re.findall(r"[\\w-]+\\s*:\\s*([^;{}]+);", css)
    px_value_declarations = [
        value.strip()
        for value in css_values
        if re.search(r"(?<![\\w-])-?\\d*\\.?\\d+px\\b", value, flags=re.IGNORECASE)
    ]
    assert not px_value_declarations, f"Found CSS values with px units: {px_value_declarations}"

    header_title_block = re.search(r"\.header h1\s*\{([\s\S]*?)\}", css)
    assert header_title_block is not None, ".header h1 rule not found"

    header_title_styles = header_title_block.group(1)
    assert "font-size: clamp(1.694rem, 2.8vw, 2.212rem);" in header_title_styles

    assert ".section { font-size: 2.0rem; }" not in css


def test_dashboard_template_files_are_consistent_when_both_exist():
    frontend_template = PROJECT_ROOT / "frontend" / "templates" / "dashboard.html"
    root_template = PROJECT_ROOT / "dashboard.html"

    if not (frontend_template.exists() and root_template.exists()):
        return

    assert frontend_template.read_text(encoding="utf-8") == root_template.read_text(encoding="utf-8")
