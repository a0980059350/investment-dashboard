import os, re, time
from datetime import datetime
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
        'name': '安聯台灣科技基金',
        'url': 'https://fund.hncb.com.tw/w/wr/wr02_ACDD04-005003.djhtm'
    },
    {
        'name': '統一全球新科技基金',
        'url': 'https://fund.hncb.com.tw/w/wr/wr02_ACPS38-009022.djhtm'
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
        '/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf'
    ]

    for path in font_paths:

        if os.path.exists(path):

            font_name = font_manager.FontProperties(
                fname=path
            ).get_name()

            plt.rcParams['font.family'] = font_name

            break

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['text.color'] = TEXT
    plt.rcParams['axes.edgecolor'] = GOLD
    plt.rcParams['axes.labelcolor'] = TEXT
    plt.rcParams['xtick.color'] = TEXT_DIM
    plt.rcParams['ytick.color'] = TEXT_DIM


def clean_table(table):

    table = table.copy()

    table.columns = [
        str(column).strip()
        for column in table.columns
    ]

    date_column = next(
        (
            column
            for column in table.columns
            if '日期' in column
        ),
        None
    )

    value_column = next(
        (
            column
            for column in table.columns
            if '淨值' in column
        ),
        None
    )

    if not date_column or not value_column:
        raise ValueError('找不到日期/淨值欄位')

    output = table[
        [date_column, value_column]
    ].copy()

    output.columns = [
        'Date',
        'Value'
    ]

    current_year = datetime.now(TZ).year

    def parse_date(value):

        text = str(value).strip()

        if re.fullmatch(
            r'\d{1,2}/\d{1,2}',
            text
        ):
            text = f'{current_year}/{text}'

        return pd.to_datetime(
            text,
            errors='coerce'
        )

    output['Date'] = output[
        'Date'
    ].map(parse_date)

    output['Value'] = pd.to_numeric(
        output['Value']
        .astype(str)
        .str.replace(',', '', regex=False),
        errors='coerce'
    )

    return (
        output
        .dropna()
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
                    'Referer': 'https://fund.hncb.com.tw/',
                    'Accept-Language': 'zh-TW,zh;q=0.9,en;q=0.8'
                },
                timeout=60
            )

            response.raise_for_status()

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

    candidates = []

    try:

        tables = pd.read_html(
            response.text
        )

    except Exception as error:

        raise RuntimeError(
            f'基金網頁表格解析失敗：{error}'
        )

    for table in tables:

        columns_text = ' '.join(
            map(str, table.columns)
        )

        if (
            '日期' in columns_text
            and '淨值' in columns_text
        ):

            try:

                candidates.append(
                    clean_table(table)
                )

            except Exception:

                pass

    if not candidates:
        raise RuntimeError('基金資料抓取失敗')

    data = (
        pd.concat(candidates)
        .drop_duplicates('Date')
        .sort_values('Date')
    )

    start_date = (
        pd.Timestamp.now()
        - pd.Timedelta(days=380)
    )

    return (
        data[
            data['Date'] >= start_date
        ]
        .tail(270)
    )


def fetch_etf(ticker):

    data = pd.DataFrame()
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

            if not data.empty:
                break

        except Exception as error:

            last_error = error

            print(
                f'ETF抓取失敗 {ticker}，'
                f'第 {attempt + 1} 次：',
                repr(error)
            )

        time.sleep(3)

    if data.empty:

        if last_error is not None:

            raise RuntimeError(
                f'{ticker} 無資料：{last_error}'
            )

        raise RuntimeError(
            f'{ticker} 無資料'
        )

    if isinstance(
        data.columns,
        pd.MultiIndex
    ):

        data.columns = (
            data.columns
            .get_level_values(0)
        )

    required_columns = [
        'Open',
        'High',
        'Low',
        'Close'
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f'{ticker} 缺少欄位：'
            f'{missing_columns}'
        )

    data = (
        data[
            required_columns
        ]
        .dropna()
    )

    weekly_data = pd.DataFrame({
        'Open': data['Open']
        .resample('W-FRI')
        .first(),

        'High': data['High']
        .resample('W-FRI')
        .max(),

        'Low': data['Low']
        .resample('W-FRI')
        .min(),

        'Close': data['Close']
        .resample('W-FRI')
        .last()
    })

    return (
        weekly_data
        .dropna()
        .tail(53)
    )


def stats(series):

    series = series.dropna()

    latest = float(
        series.iloc[-1]
    )

    high = float(
        series.max()
    )

    drawdown = (
        latest / high
        - 1
    )

    return_rate = (
        latest
        / float(series.iloc[0])
        - 1
    )

    return (
        latest,
        high,
        drawdown,
        return_rate
    )


def card_backdrop(ax):

    ax.imshow(
        np.linspace(
            0,
            1,
            256
        ).reshape(-1, 1),
        cmap=_PANEL_CMAP,
        extent=(0, 1, 0, 1),
        transform=ax.transAxes,
        aspect='auto',
        origin='lower',
        zorder=-5
    )


def corner_brackets(
    ax,
    frac=0.045,
    lw=2.6,
    color=GOLD_BRIGHT
):

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


def draw_signal_light(
    fig,
    ax,
    up,
    x=0.92,
    y=0.965,
    r_px=10
):

    fill = (
        LIGHT_GREEN
        if up
        else LIGHT_RED
    )

    edge = (
        LIGHT_GREEN_EDGE
        if up
        else LIGHT_RED_EDGE
    )

    x_display, y_display = (
        ax.transAxes.transform(
            (x, y)
        )
    )

    fig.add_artist(
        Circle(
            (
                x_display,
                y_display
            ),
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

    for side in [
        'top',
        'right',
        'left',
        'bottom'
    ]:

        ax.spines[
            side
        ].set_visible(True)

        ax.spines[
            side
        ].set_color(GOLD_DIM)

        ax.spines[
            side
        ].set_linewidth(1.3)

    corner_brackets(ax)


def plot_fund(
    ax,
    name,
    data,
    fig
):

    x = np.arange(
        len(data)
    )

    latest, high, drawdown, return_rate = stats(
        data['Value']
    )

    style_card(ax)

    ax.plot(
        x,
        data['Value'],
        lw=2.6,
        color=GOLD_BRIGHT,
        solid_capstyle='round',
        zorder=5
    )

    ax.axhline(
        high,
        lw=1.1,
        ls='--',
        color=GOLD_DIM,
        zorder=3
    )

    ax.text(
        len(x) - 1,
        high,
        f' 最高 {high:.2f}',
        va='bottom',
        ha='right',
        fontsize=11,
        color=TEXT_DIM
    )

    is_up = (
        abs(drawdown) > 0.20
    )

    fund_status = (
        '可以加碼'
        if is_up
        else '暫不加碼'
    )

    ax.set_title(
        name,
        loc='left',
        fontsize=17,
        fontweight='bold',
        pad=14,
        color=GOLD
    )

    ax.text(
        0.97,
        0.93,
        (
            f'{fund_status}\n'
            f'最新淨值 {latest:.2f}\n'
            f'近一年報酬 {return_rate:+.1%}\n'
            f'回撤 {drawdown:.1%}'
        ),
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=12,
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

    draw_signal_light(
        fig,
        ax,
        is_up
    )

    ax.grid(
        alpha=0.08,
        color=GOLD_DIM,
        lw=0.6
    )

    ax.set_xlim(
        0,
        max(
            1,
            len(x) - 1
        )
    )

    ax.tick_params(
        labelbottom=False
    )


def plot_etf(
    ax,
    name,
    data,
    ema_period,
    fig
):

    x = np.arange(
        len(data)
    )

    ema = (
        data['Close']
        .ewm(
            span=ema_period,
            adjust=False
        )
        .mean()
    )

    latest, high, drawdown, return_rate = stats(
        data['Close']
    )

    stop = high * 0.8

    style_card(ax)

    for i, (_, row) in enumerate(
        data.iterrows()
    ):

        open_price = float(
            row['Open']
        )

        high_price = float(
            row['High']
        )

        low_price = float(
            row['Low']
        )

        close_price = float(
            row['Close']
        )

        candle_up = (
            close_price >= open_price
        )

        candle_color = (
            UP
            if candle_up
            else DOWN
        )

        ax.vlines(
            i,
            low_price,
            high_price,
            lw=0.7,
            color=candle_color,
            zorder=4
        )

        body_bottom = min(
            open_price,
            close_price
        )

        body_height = max(
            abs(
                close_price
                - open_price
            ),
            max(
                close_price,
                open_price
            ) * 0.001
        )

        ax.add_patch(
            Rectangle(
                (
                    i - 0.21,
                    body_bottom
                ),
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

    ax.axhline(
        high,
        lw=1.1,
        ls='--',
        color=GOLD_DIM,
        zorder=3
    )

    ax.axhline(
        stop,
        lw=1.3,
        ls=':',
        color=TEXT_DIM,
        zorder=3
    )

    ax.text(
        len(x) - 1,
        high,
        f' 最高 {high:.2f}',
        va='bottom',
        ha='right',
        fontsize=11,
        color=TEXT_DIM
    )

    ax.text(
        len(x) - 1,
        stop,
        f' 停損價 {stop:.2f}',
        va='bottom',
        ha='right',
        fontsize=11,
        color=TEXT_DIM
    )

    is_up = (
        latest
        > float(
            ema.iloc[-1]
        )
    )

    status = (
        f'站上週EMA{ema_period}'
        if is_up
        else f'跌破週EMA{ema_period}'
    )

    ax.set_title(
        name,
        loc='left',
        fontsize=17,
        fontweight='bold',
        pad=14,
        color=GOLD
    )

    ax.text(
        0.97,
        0.94,
        (
            f'最新價 {latest:.2f}\n'
            f'近一年報酬 {return_rate:+.1%}\n'
            f'回撤 {drawdown:.1%}\n'
            f'{status}'
        ),
        transform=ax.transAxes,
        ha='right',
        va='top',
        fontsize=12,
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

    draw_signal_light(
        fig,
        ax,
        is_up
    )

    ax.grid(
        alpha=0.08,
        color=GOLD_DIM,
        lw=0.6
    )

    ax.set_xlim(
        -1,
        len(x)
    )

    ax.tick_params(
        labelbottom=False
    )


def add_vignette(fig):

    ax = fig.add_axes(
        [0, 0, 1, 1],
        zorder=-20
    )

    ax.axis('off')

    ax.set_xlim(
        0,
        1
    )

    ax.set_ylim(
        0,
        1
    )

    ny = 300
    nx = 140

    yy, xx = np.mgrid[
        0:ny,
        0:nx
    ]

    cx = (
        nx - 1
    ) / 2

    cy = (
        ny - 1
    ) / 2

    distance = np.sqrt(
        (
            (xx - cx) / cx
        ) ** 2
        +
        (
            (yy - cy) / cy
        ) ** 2
    )

    distance = np.clip(
        distance,
        0,
        1
    )

    alpha = (
        distance ** 2.2
    ) * 0.5

    rgba = np.zeros(
        (
            ny,
            nx,
            4
        )
    )

    rgba[..., 3] = alpha

    ax.imshow(
        rgba,
        extent=(0, 1, 0, 1),
        aspect='auto',
        origin='lower'
    )


def main():

    setup_font()

    fig = plt.figure(
        figsize=(10.8, 24),
        dpi=100
    )

    fig.patch.set_facecolor(
        BG
    )

    add_vignette(fig)

    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[
            0.5,
            4.75,
            4.75
        ],
        hspace=0.34,
        wspace=0.30,
        left=0.075,
        right=0.955,
        top=0.972,
        bottom=0.028
    )

    title_ax = fig.add_subplot(
        grid[0, :]
    )

    title_ax.axis('off')

    title_ax.set_xlim(
        0,
        1
    )

    title_ax.set_ylim(
        0,
        1
    )

    title_ax.text(
        0,
        0.78,
        '科技四核心投資儀表板',
        fontsize=36,
        fontweight='bold',
        ha='left',
        va='center',
        color=GOLD
    )

    title_ax.text(
        0,
        0.32,
        '基金 × ETF｜近一年',
        fontsize=13,
        ha='left',
        va='center',
        color=GOLD_LIGHT,
        alpha=0.9
    )

    title_ax.plot(
        [0, 1],
        [0.05, 0.05],
        color=GOLD_DIM,
        lw=1,
        alpha=0.6,
        solid_capstyle='round'
    )

    title_ax.plot(
        [0, 0.22],
        [0.05, 0.05],
        color=GOLD_BRIGHT,
        lw=1.6,
        alpha=0.9,
        solid_capstyle='round'
    )

    fund_axes = [
        fig.add_subplot(
            grid[1, 0]
        ),
        fig.add_subplot(
            grid[1, 1]
        )
    ]

    etf_axes = [
        fig.add_subplot(
            grid[2, 0]
        ),
        fig.add_subplot(
            grid[2, 1]
        )
    ]

    for ax, fund in zip(
        fund_axes,
        FUNDS
    ):

        try:

            fund_data = fetch_fund(
                fund['url']
            )

            plot_fund(
                ax,
                fund['name'],
                fund_data,
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
                fontsize=17,
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
                fontsize=12,
                color=TEXT_DIM,
                transform=ax.transAxes
            )

    for ax, etf in zip(
        etf_axes,
        ETFS
    ):

        try:

            etf_data = fetch_etf(
                etf['ticker']
            )

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
                fontsize=17,
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
                fontsize=12,
                color=TEXT_DIM,
                transform=ax.transAxes
            )

    fig.text(
        0.955,
        0.014,
        (
            '更新時間：'
            f"{datetime.now(TZ).strftime('%Y/%m/%d %H:%M')}"
        ),
        fontsize=10,
        ha='right',
        color=TEXT_DIM,
        alpha=0.85
    )

    plt.savefig(
        OUTPUT,
        dpi=100,
        facecolor=fig.get_facecolor()
    )

    plt.close(fig)

    print(
        '已產生',
        OUTPUT
    )


if __name__ == '__main__':
    main()
