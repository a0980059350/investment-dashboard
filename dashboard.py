import os, re, time
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.transforms import IdentityTransform


TZ = ZoneInfo('Asia/Taipei')
OUTPUT = 'wallpaper.png'


FUNDS = [
    {
        'name': '安聯科技',
        'url': 'https://www.moneydj.com/funddj/ya/yp010000.djhtm?a=ACDD04'
    },
    {
        'name': '統一科技',
        'url': 'https://www.moneydj.com/funddj/ya/yp010000.djhtm?a=ACPS38'
    }
]


ETFS = [
    {
        'name': '00631L',
        'ticker': '00631L.TW',
        'ema': 32
    },
    {
        'name': '00830',
        'ticker': '00830.TW',
        'ema': 42
    }
]


# ---- 黑金配色 ----
BG = '#050505'
PANEL = '#111111'
PANEL_LO = '#0a0a0a'
PANEL_HI = '#161616'

GOLD = '#d4af37'
GOLD_BRIGHT = '#f4d160'
GOLD_LIGHT = '#e8cf7c'
GOLD_DIM = '#8a7326'

UP = '#e5c85c'
DOWN = '#555555'

TEXT = '#e8d9a8'
TEXT_DIM = '#c9b979'

LIGHT_GREEN = '#3ddc84'
LIGHT_GREEN_EDGE = '#7cffb5'

LIGHT_RED = '#ff4d4d'
LIGHT_RED_EDGE = '#ffb3b3'


_PANEL_CMAP = LinearSegmentedColormap.from_list(
    'panel',
    [PANEL_LO, PANEL_HI]
)


def setup_font():
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf',
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/truetype/arphic/ukai.ttc',
        '/usr/share/fonts/truetype/arphic/uming.ttc'
    ]

    for path in font_paths:
        if os.path.exists(path):
            font_manager.fontManager.addfont(path)
            font_name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams['font.family'] = font_name
            break

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.color'] = TEXT
    plt.rcParams['axes.edgecolor'] = GOLD
    plt.rcParams['axes.labelcolor'] = TEXT
    plt.rcParams['xtick.color'] = TEXT_DIM
    plt.rcParams['ytick.color'] = TEXT_DIM


def flatten_columns(columns):
    if isinstance(columns, pd.MultiIndex):
        result = []
        for column in columns:
            parts = [
                str(part).strip()
                for part in column
                if str(part).strip() not in ('', 'nan', 'None')
            ]
            result.append(' '.join(parts))
        return result

    return [str(column).strip() for column in columns]


def parse_fund_date(value):
    text = str(value).strip()
    text = re.sub(r'\s+', '', text)
    text = text.replace('.', '/').replace('-', '/')

    if not text or text.lower() in ('nan', 'none'):
        return pd.NaT

    current_year = datetime.now(TZ).year
    current_month = datetime.now(TZ).month

    # 月/日
    if re.fullmatch(r'\d{1,2}/\d{1,2}', text):
        month = int(text.split('/')[0])
        year = current_year - 1 if month > current_month else current_year
        text = f'{year}/{text}'

    # 民國年/月/日，例如 113/07/18
    match = re.fullmatch(r'(\d{2,3})/(\d{1,2})/(\d{1,2})', text)
    if match:
        year = int(match.group(1))
        if year < 1911:
            text = f'{year + 1911}/{match.group(2)}/{match.group(3)}'

    return pd.to_datetime(text, errors='coerce')


def parse_fund_value(value):
    text = str(value).strip()
    text = text.replace(',', '')
    text = re.sub(r'[^\d.\-]', '', text)
    return pd.to_numeric(text, errors='coerce')


EXCLUDE_VALUE_KEYWORDS = ('累計', '指數', '報酬', '成長', '規模', '配息')


def is_valid_value_header(text):
    return '淨值' in text and not any(
        keyword in text for keyword in EXCLUDE_VALUE_KEYWORDS
    )


def clean_table(table):
    table = table.copy()
    table.columns = flatten_columns(table.columns)

    date_column = next(
        (column for column in table.columns if '日期' in column),
        None
    )

    value_column = next(
        (column for column in table.columns if is_valid_value_header(column)),
        None
    )

    # 有些網頁把標題放在資料列，不在欄名
    if date_column is None or value_column is None:
        for row_index in range(min(8, len(table))):
            row_text = [str(value).strip() for value in table.iloc[row_index].tolist()]

            date_pos = next(
                (index for index, value in enumerate(row_text) if '日期' in value),
                None
            )
            value_pos = next(
                (index for index, value in enumerate(row_text) if is_valid_value_header(value)),
                None
            )

            if date_pos is not None and value_pos is not None:
                table = table.iloc[row_index + 1:, [date_pos, value_pos]].copy()
                table.columns = ['Date', 'Value']
                break
        else:
            raise ValueError('找不到日期/淨值欄位')
    else:
        table = table[[date_column, value_column]].copy()
        table.columns = ['Date', 'Value']

    table['Date'] = table['Date'].map(parse_fund_date)
    table['Value'] = table['Value'].map(parse_fund_value)

    return (
        table
        .dropna(subset=['Date', 'Value'])
        .drop_duplicates('Date')
        .sort_values('Date')
    )


def fetch_fund(url):
    last_error = None
    response = None

    for attempt in range(3):
        try:
            response = requests.get(
                url,
                headers={
                    'User-Agent': (
                        'Mozilla/5.0 '
                        '(Linux; Android 13) '
                        'AppleWebKit/537.36 '
                        '(KHTML, like Gecko) '
                        'Chrome/126.0 Mobile Safari/537.36'
                    ),
                    'Referer': 'https://www.moneydj.com/',
                    'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
                },
                timeout=60
            )
            response.raise_for_status()

            if not response.encoding or response.encoding.lower() == 'iso-8859-1':
                response.encoding = response.apparent_encoding

            break

        except Exception as error:
            last_error = error
            print(
                f'基金抓取失敗，第 {attempt + 1} 次：',
                repr(error)
            )
            time.sleep(3)
    else:
        raise RuntimeError(
            f'基金網站連線失敗：{last_error}'
        )

    try:
        tables = pd.read_html(StringIO(response.text))
    except Exception as error:
        raise RuntimeError(
            f'基金網頁表格解析失敗：{error}'
        )

    candidates = []

    for table in tables:
        try:
            cleaned = clean_table(table)
            if len(cleaned) >= 2:
                candidates.append(cleaned)
        except Exception:
            continue

    if not candidates:
        raise RuntimeError(
            f'基金資料抓取失敗：共找到 {len(tables)} 個表格，但無法辨識日期與淨值'
        )

    data = max(candidates, key=len).sort_values('Date')

    start_date = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.Timedelta(days=380)
    )

    data = (
        data[data['Date'] >= start_date]
        .tail(270)
    )

    if len(data) < 2:
        raise RuntimeError('基金資料筆數不足')

    high_1y = extract_high_1y_from_tables(tables)

    return data, high_1y


def extract_high_1y_from_tables(tables):
    """
    直接從基金頁面本身既有的摘要表格
    （淨值日期／最新淨值／每日變化／最高淨值(年)／最低淨值(年)）
    取得官方公布的「最高淨值(年)」。

    這比另外呼叫 m.moneydj.com 手機版頁面更可靠，
    因為手機版頁面時常被 CDN／伺服器端快取住，
    抓到的可能是好幾週前的舊快照（例如已經漲破前高之後，
    手機版頁面卻還停留在漲破之前的數字），
    導致「最高淨值(年)」被系統性低估。
    改用同一次請求、同一個頁面裡的摘要表格，
    可以確保跟每日淨值走勢圖用的是同一份、同一時間點的資料。
    """
    for table in tables:
        columns = flatten_columns(table.columns)

        # 情況一：標題本身就是欄名
        for col_index, col in enumerate(columns):
            if '最高' in col and '淨值' in col and '年' in col:
                for _, row in table.iterrows():
                    value = parse_fund_value(row.iloc[col_index])
                    if pd.notna(value):
                        return float(value)

        # 情況二：標題放在資料列裡，數值在下一列同一欄位
        for row_index in range(min(8, len(table) - 1)):
            row_text = [str(v).strip() for v in table.iloc[row_index].tolist()]

            label_pos = next(
                (
                    i for i, v in enumerate(row_text)
                    if '最高' in v and '淨值' in v and '年' in v
                ),
                None
            )

            if label_pos is not None:
                value = parse_fund_value(table.iloc[row_index + 1, label_pos])
                if pd.notna(value):
                    return float(value)

    return None




def fetch_etf(ticker):
    data = pd.DataFrame()
    raw_data = pd.DataFrame()
    last_error = None

    for attempt in range(3):
        try:
            data = yf.download(
                ticker,
                period='18mo',
                interval='1d',
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30
            )

            raw_data = yf.download(
                ticker,
                period='18mo',
                interval='1d',
                auto_adjust=False,
                progress=False,
                threads=False,
                timeout=30
            )

            if not data.empty and not raw_data.empty:
                break

        except Exception as error:
            last_error = error
            print(
                f'ETF抓取失敗 {ticker}，'
                f'第 {attempt + 1} 次：',
                repr(error)
            )

        time.sleep(3)

    if data.empty or raw_data.empty:
        if last_error is not None:
            raise RuntimeError(
                f'{ticker} 無資料：{last_error}'
            )

        raise RuntimeError(
            f'{ticker} 無資料'
        )

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    required_columns = ['Open', 'High', 'Low', 'Close']

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f'{ticker} 缺少欄位：{missing_columns}'
        )

    data = data[required_columns].dropna()
    daily_raw_close = raw_data['Close'].dropna()

    weekly_data = pd.DataFrame({
        'Open': data['Open'].resample('W-FRI').first(),
        'High': data['High'].resample('W-FRI').max(),
        'Low': data['Low'].resample('W-FRI').min(),
        'Close': data['Close'].resample('W-FRI').last()
    })

    return {
        'weekly': weekly_data.dropna().tail(53),
        'daily_adj': data['Close'],
        'daily_raw': daily_raw_close
    }



def is_week_complete(period_end):
    if getattr(period_end, 'tzinfo', None) is not None:
        period_end_date = period_end.tz_convert(TZ).date()
    else:
        period_end_date = period_end.date()

    today = datetime.now(TZ).date()
    return period_end_date <= today


def stats(series):
    series = series.dropna()

    latest = float(series.iloc[-1])
    high = float(series.max())
    drawdown = latest / high - 1
    return_rate = latest / float(series.iloc[0]) - 1

    return latest, high, drawdown, return_rate


def date_based_stats(dates, values):
    """
    以「去年今天」到「今天」為基準計算報酬率與回撤。
    若去年今天當天不是交易日，自動改用前一個交易日的資料。
    """
    series = (
        pd.Series(list(values), index=pd.DatetimeIndex(dates))
        .sort_index()
    )
    series = series[~series.index.duplicated(keep='last')].dropna()

    latest_date = series.index[-1]
    latest = float(series.iloc[-1])

    one_year_ago = latest_date - pd.DateOffset(years=1)

    base_slice = series[series.index <= one_year_ago]

    if not base_slice.empty:
        base_value = float(base_slice.iloc[-1])
        window_start = base_slice.index[-1]
    else:
        base_value = float(series.iloc[0])
        window_start = series.index[0]

    window = series[series.index >= window_start]
    high = float(window.max())

    return_rate = latest / base_value - 1
    drawdown = latest / high - 1

    return latest, high, drawdown, return_rate


def latest_and_high(dates, values):
    """近一年（今天往前一年）的最新值與最高值，用於顯示一般（未還原）價格。"""
    series = (
        pd.Series(list(values), index=pd.DatetimeIndex(dates))
        .sort_index()
    )
    series = series[~series.index.duplicated(keep='last')].dropna()

    latest_date = series.index[-1]
    latest = float(series.iloc[-1])

    window_start = latest_date - pd.DateOffset(years=1)
    window = series[series.index >= window_start]
    high = float(window.max())

    return latest, high


def card_backdrop(ax):
    ax.imshow(
        np.linspace(0, 1, 256).reshape(-1, 1),
        cmap=_PANEL_CMAP,
        extent=(0, 1, 0, 1),
        transform=ax.transAxes,
        aspect='auto',
        origin='lower',
        zorder=-5
    )


def corner_brackets(ax, frac=0.045, lw=2.6, color=GOLD_BRIGHT):
    corners = [
        (0, 0, 1, 1),
        (1, 0, -1, 1),
        (0, 1, 1, -1),
        (1, 1, -1, -1)
    ]

    for x, y, dx, dy in corners:
        ax.plot(
            [x, x + dx * frac],
            [y, y],
            transform=ax.transAxes,
            color=color,
            lw=lw,
            solid_capstyle='round',
            zorder=12,
            clip_on=False
        )

        ax.plot(
            [x, x],
            [y, y + dy * frac],
            transform=ax.transAxes,
            color=color,
            lw=lw,
            solid_capstyle='round',
            zorder=12,
            clip_on=False
        )


def draw_signal_light(fig, ax, up, label=None, x=0.92, y=0.965, r_px=20):
    fill = LIGHT_GREEN if up else LIGHT_RED
    edge = LIGHT_GREEN_EDGE if up else LIGHT_RED_EDGE

    if label:
        ax.text(
            x - 0.07,
            y,
            label,
            transform=ax.transAxes,
            ha='right',
            va='center',
            fontsize=24,
            fontweight='bold',
            color=fill,
            zorder=31,
            clip_on=False
        )

    x_display, y_display = ax.transAxes.transform((x, y))

    fig.add_artist(
        Circle(
            (x_display, y_display),
            r_px,
            transform=IdentityTransform(),
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.6,
            zorder=30,
            clip_on=False
        )
    )


def style_card(ax):
    ax.set_facecolor('none')
    card_backdrop(ax)

    for side in ['top', 'right', 'left', 'bottom']:
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color(GOLD_DIM)
        ax.spines[side].set_linewidth(1.3)

    corner_brackets(ax)


def plot_fund(ax, name, data, high_1y, fig):
    x = np.arange(len(data))
    latest, local_high, _, _ = date_based_stats(
        data['Date'],
        data['Value']
    )

    high = high_1y if high_1y is not None else local_high
    drawdown = latest / high - 1
    add_price = high * 0.8

    style_card(ax)

    ax.plot(
        x,
        data['Value'],
        lw=2.6,
        color=GOLD_BRIGHT,
        solid_capstyle='round',
        zorder=5
    )

    y_min, y_max = ax.get_ylim()
    data_range = local_high - y_min
    ax.set_ylim(
        y_min - data_range * 0.12,
        local_high + data_range * 0.14
    )

    is_up = abs(drawdown) > 0.20
    fund_status = '可以加碼' if is_up else '暫停加碼'

    ax.set_title(
        name,
        loc='left',
        fontsize=34,
        fontweight='bold',
        pad=14,
        color=GOLD
    )

    ax.text(
        0.97,
        0.06,
        (
            f'最新淨值 {latest:.2f}\n'
            f'最高淨值 {high:.2f}\n'
            f'加碼價 {add_price:.2f}\n'
            f'回撤 {drawdown:.1%}'
        ),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=24,
        color=TEXT,
        linespacing=1.7,
        bbox=dict(
            boxstyle='round,pad=.5',
            facecolor=BG,
            edgecolor=GOLD_DIM,
            linewidth=1,
            alpha=0.8
        )
    )

    draw_signal_light(fig, ax, is_up, label=fund_status)

    ax.grid(alpha=0.08, color=GOLD_DIM, lw=0.6)
    ax.set_xlim(0, max(1, len(x) - 1))
    ax.tick_params(labelbottom=False)


def plot_etf(ax, name, etf_bundle, ema_period, fig):
    data = etf_bundle['weekly']
    x = np.arange(len(data))

    ema = (
        data['Close']
        .ewm(span=ema_period, adjust=False)
        .mean()
    )

    latest, high = latest_and_high(
        etf_bundle['daily_raw'].index,
        etf_bundle['daily_raw'].values
    )

    _, _, drawdown, _ = date_based_stats(
        etf_bundle['daily_adj'].index,
        etf_bundle['daily_adj'].values
    )

    stop = high * 0.8

    style_card(ax)

    for i, (_, row) in enumerate(data.iterrows()):
        open_price = float(row['Open'])
        high_price = float(row['High'])
        low_price = float(row['Low'])
        close_price = float(row['Close'])

        candle_up = close_price >= open_price
        candle_color = UP if candle_up else DOWN

        ax.vlines(
            i,
            low_price,
            high_price,
            lw=0.7,
            color=candle_color,
            zorder=4
        )

        body_bottom = min(open_price, close_price)
        body_height = max(
            abs(close_price - open_price),
            max(close_price, open_price) * 0.001
        )

        ax.add_patch(
            Rectangle(
                (i - 0.21, body_bottom),
                0.42,
                body_height,
                facecolor=candle_color,
                edgecolor=GOLD_DIM,
                linewidth=0.5,
                alpha=0.92,
                zorder=5
            )
        )

    ax.plot(
        x,
        ema.values,
        lw=2.1,
        label=f'EMA{ema_period}',
        color=GOLD_BRIGHT,
        solid_capstyle='round',
        zorder=6
    )

    y_min, y_max = ax.get_ylim()
    data_range = high - y_min
    ax.set_ylim(
        y_min - data_range * 0.12,
        high + data_range * 0.14
    )

    week_complete = is_week_complete(data.index[-1])
    signal_index = -1 if week_complete else -2

    signal_close = float(data['Close'].iloc[signal_index])
    signal_ema = float(ema.iloc[signal_index])

    is_up = signal_close > signal_ema
    status = (
        f'站上{ema_period}週線'
        if is_up
        else f'跌破{ema_period}週線'
    )

    ax.set_title(
        name,
        loc='left',
        fontsize=34,
        fontweight='bold',
        pad=14,
        color=GOLD
    )

    ax.text(
        0.97,
        0.06,
        (
            f'最新價 {latest:.2f}\n'
            f'最高價 {high:.2f}\n'
            f'停損價 {stop:.2f}\n'
            f'回撤 {drawdown:.1%}'
        ),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=24,
        color=TEXT,
        linespacing=1.65,
        bbox=dict(
            boxstyle='round,pad=.5',
            facecolor=BG,
            edgecolor=GOLD_DIM,
            linewidth=1,
            alpha=0.8
        )
    )

    draw_signal_light(fig, ax, is_up, label=status)

    ax.grid(alpha=0.08, color=GOLD_DIM, lw=0.6)
    ax.set_xlim(-1, len(x))
    ax.tick_params(labelbottom=False)


def add_vignette(fig):
    ax = fig.add_axes([0, 0, 1, 1], zorder=-20)
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ny = 300
    nx = 140

    yy, xx = np.mgrid[0:ny, 0:nx]

    cx = (nx - 1) / 2
    cy = (ny - 1) / 2

    distance = np.sqrt(
        ((xx - cx) / cx) ** 2
        + ((yy - cy) / cy) ** 2
    )

    distance = np.clip(distance, 0, 1)
    alpha = (distance ** 2.2) * 0.5

    rgba = np.zeros((ny, nx, 4))
    rgba[..., 3] = alpha

    ax.imshow(
        rgba,
        extent=(0, 1, 0, 1),
        aspect='auto',
        origin='lower'
    )


def main():
    setup_font()

    fig = plt.figure(figsize=(10.8, 23.4), dpi=100)
    fig.patch.set_facecolor(BG)

    add_vignette(fig)

    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[0.42, 4.79, 4.79],
        hspace=0.10,
        wspace=0.06,
        left=0.03,
        right=0.97,
        top=0.985,
        bottom=0.010
    )

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis('off')
    title_ax.set_xlim(0, 1)
    title_ax.set_ylim(0, 1)

    title_ax.text(
        1,
        0.78,
        (
            '更新時間：'
            f"{datetime.now(TZ).strftime('%Y/%m/%d %H:%M')}"
        ),
        fontsize=20,
        ha='right',
        va='center',
        color=TEXT_DIM,
        alpha=0.85
    )

    fund_axes = [
        fig.add_subplot(grid[1, 0]),
        fig.add_subplot(grid[1, 1])
    ]

    etf_axes = [
        fig.add_subplot(grid[2, 0]),
        fig.add_subplot(grid[2, 1])
    ]

    for ax, fund in zip(fund_axes, FUNDS):
        try:
            fund_data, high_1y = fetch_fund(fund['url'])

            if high_1y is None:
                print(
                    '官方最高淨值(年)抓取失敗，改用近期資料估算:',
                    fund['name']
                )

            plot_fund(
                ax,
                fund['name'],
                fund_data,
                high_1y,
                fig
            )

        except Exception as error:
            print(
                '基金錯誤:',
                fund['name'],
                repr(error)
            )

            style_card(ax)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.text(
                0.04,
                0.65,
                fund['name'],
                fontsize=34,
                fontweight='bold',
                color=GOLD,
                transform=ax.transAxes
            )

            ax.text(
                0.04,
                0.42,
                (
                    '資料更新失敗\n'
                    f'{type(error).__name__}: {error}'
                ),
                fontsize=24,
                color=TEXT_DIM,
                transform=ax.transAxes
            )

    for ax, etf in zip(etf_axes, ETFS):
        try:
            etf_data = fetch_etf(etf['ticker'])

            plot_etf(
                ax,
                etf['name'],
                etf_data,
                etf['ema'],
                fig
            )

        except Exception as error:
            print(
                'ETF錯誤:',
                etf['name'],
                repr(error)
            )

            style_card(ax)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.text(
                0.04,
                0.65,
                etf['name'],
                fontsize=34,
                fontweight='bold',
                color=GOLD,
                transform=ax.transAxes
            )

            ax.text(
                0.04,
                0.42,
                (
                    '資料更新失敗\n'
                    f'{type(error).__name__}: {error}'
                ),
                fontsize=24,
                color=TEXT_DIM,
                transform=ax.transAxes
            )

    plt.savefig(
        OUTPUT,
        dpi=100,
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print('已產生', OUTPUT)


if __name__ == '__main__':
    main()




