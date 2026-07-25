import os, re, time, json
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
HISTORY_FILE = 'history.json'


FUNDS = [
    {
        'name': '安聯科技',
        'display': '安聯',
        'url': 'https://www.moneydj.com/funddj/ya/yp010000.djhtm?a=ACDD04'
    },
    {
        'name': '統一科技',
        'display': '統一',
        'url': 'https://www.moneydj.com/funddj/ya/yp010000.djhtm?a=ACPS38'
    }
]


ETFS = [
    {
        'name': '00631L',
        'display': '正2',
        'ticker': '00631L.TW',
        'ema': 32
    },
    {
        'name': '00830',
        'display': '費半',
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

LIGHT_YELLOW = '#ffd23d'
LIGHT_YELLOW_EDGE = '#fff0a8'

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

    return data


def load_history():
    """
    讀取先前累積的每日淨值歷史紀錄（跨每次 workflow 執行持續保存）。
    找不到檔案或內容損毀時，回傳空字典，之後會自然從頭開始累積。
    """
    if not os.path.exists(HISTORY_FILE):
        return {}

    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as error:
        print('讀取歷史紀錄失敗，改用空白重新累積：', repr(error))
        return {}


def save_history(history):
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as file:
            json.dump(history, file, ensure_ascii=False)
    except Exception as error:
        print('寫入歷史紀錄失敗：', repr(error))


def update_history_and_get_high(history, fund_name, data):
    """
    把這次抓到的每日淨值併入長期歷史紀錄裡，
    並回傳「自從開始累積以來，最近一年內」看過的最高值。

    這個做法不依賴任何網站提供的「最高淨值(年)」欄位
    （那個欄位常常是網頁用 JavaScript 動態載入的，
    requests 抓不到），改成每次執行都把當下抓到的
    每日淨值記錄下來，自己長期滾動累積出正確的一年高點。
    """
    fund_history = history.get(fund_name, {})

    for _, row in data.iterrows():
        date_key = row['Date'].strftime('%Y-%m-%d')
        fund_history[date_key] = float(row['Value'])

    # 保留視窗從400天拉長到6年，跟ETF那邊 period='6y' 的邏輯一致。
    # 原本400天的cutoff會把超過1年多以前的淨值資料整批刪掉，
    # 導致「3年/5年報酬率」永遠找不到夠久以前的基準值，
    # 只能退回用「現有資料最早一筆」頂替，算出來的3年、5年數字會一樣、也不準。
    cutoff = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.DateOffset(years=6)
    )

    fund_history = {
        date_key: value
        for date_key, value in fund_history.items()
        if pd.to_datetime(date_key) >= cutoff
    }

    history[fund_name] = fund_history

    one_year_ago = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.Timedelta(days=365)
    )

    recent_values = [
        value
        for date_key, value in fund_history.items()
        if pd.to_datetime(date_key) >= one_year_ago
    ]

    return max(recent_values) if recent_values else None


def history_to_chart_data(fund_history):
    """
    把累積的歷史紀錄轉換成可以直接拿去畫圖的 DataFrame。

    一開始（累積天數還不多）畫出來的線就只會是這幾天的資料，
    但隨著 workflow 每天持續執行，累積的天數會越來越多，
    最多滾動保留約 400 天，畫出來的線會逐漸變成完整一年走勢圖，
    不再受限於網站本身只提供近30日資料的限制。
    """
    if not fund_history:
        return pd.DataFrame(columns=['Date', 'Value'])

    dates = pd.to_datetime(list(fund_history.keys()))
    values = list(fund_history.values())

    return (
        pd.DataFrame({'Date': dates, 'Value': values})
        .sort_values('Date')
        .reset_index(drop=True)
    )


def fetch_twse_realtime(ticker):
    """
    直接抓證交所(TWSE)公開的即時資訊API，跟券商APP同一組資料源。
    回傳 (最新價, 昨收盤價)，抓不到或格式不對就回傳 (None, None)，
    由呼叫端 fallback 回 yfinance 的做法。
    """
    symbol = ticker.replace('.TW', '').replace('.TWO', '')
    ex_ch = f'tse_{symbol}.tw'

    url = (
        'https://mis.twse.com.tw/stock/api/getStockInfo.jsp'
        f'?ex_ch={ex_ch}&json=1&delay=0'
    )

    try:
        response = requests.get(
            url,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Linux; Android 13) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/126.0 Mobile Safari/537.36'
                ),
                'Referer': 'https://mis.twse.com.tw/stock/index.jsp'
            },
            timeout=10
        )
        response.raise_for_status()
        payload = response.json()

        rows = payload.get('msgArray') or []
        if not rows:
            return None, None

        row = rows[0]
        raw_price = row.get('z')
        raw_prev_close = row.get('y')

        # 'z' 在非交易時間或尚無成交時會是 '-'，視為抓取失敗
        if not raw_price or raw_price == '-':
            return None, None
        if not raw_prev_close or raw_prev_close == '-':
            return None, None

        return float(raw_price), float(raw_prev_close)

    except Exception as error:
        print(
            f'{ticker} TWSE即時報價抓取失敗，改用yfinance：',
            repr(error)
        )
        return None, None


def fetch_taiex_realtime():
    """加權指數即時點位+昨收，跟個股同一組TWSE API，代碼固定用 t00。"""
    url = (
        'https://mis.twse.com.tw/stock/api/getStockInfo.jsp'
        '?ex_ch=tse_t00.tw&json=1&delay=0'
    )
    try:
        response = requests.get(
            url,
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Linux; Android 13) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/126.0 Mobile Safari/537.36'
                ),
                'Referer': 'https://mis.twse.com.tw/stock/index.jsp'
            },
            timeout=10
        )
        response.raise_for_status()
        payload = response.json()
        rows = payload.get('msgArray') or []
        if not rows:
            return None, None

        row = rows[0]
        raw_price = row.get('z') or row.get('ip')
        raw_prev_close = row.get('y')

        if not raw_price or raw_price == '-':
            return None, None
        if not raw_prev_close or raw_prev_close == '-':
            return None, None

        return float(raw_price), float(raw_prev_close)

    except Exception as error:
        print('加權指數即時報價抓取失敗：', repr(error))
        return None, None


def fetch_market_margin_ratio():
    """
    大盤融資維持率 = Σ(個股融資今日餘額(股) × 收盤價) / 大盤融資金額今日餘額(元)

    分母：www.twse.com.tw 舊版 MI_MARGN?selectType=MS（集中市場信用交易統計彙總，
         'tables'結構包著'融資金額(仟元)'的今日餘額，支援date參數查歷史）。

    分子：openapi.twse.com.tw/v1/exchangeReport/MI_MARGN——這才是真正「每檔個股」
         的融資餘額（舊版selectType=ALL其實仍是集中市場加總，不是個股，踩了一次坑）。
         OpenAPI版本不支援查歷史日期，永遠只回傳最新一個交易日。
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Linux; Android 13) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/126.0 Mobile Safari/537.36'
        )
    }

    total_margin_amount = None
    matched_date = None

    # ---- 分母：找最近一個有資料的交易日的大盤融資金額今日餘額 ----
    for back_days in range(6):
        try:
            query_date = (
                datetime.now(TZ) - pd.Timedelta(days=back_days)
            ).strftime('%Y%m%d')

            ms_resp = requests.get(
                'https://www.twse.com.tw/exchangeReport/MI_MARGN',
                params={'response': 'json', 'date': query_date, 'selectType': 'MS'},
                headers=headers,
                timeout=20
            )
            ms_resp.raise_for_status()
            ms_payload = ms_resp.json()

            ms_tables = ms_payload.get('tables') or []
            if not ms_tables:
                print(f'[融資維持率] {query_date} MS 沒有tables，可能非交易日')
                continue

            credit_fields = ms_tables[0].get('fields') or []
            credit_list = ms_tables[0].get('data') or []

            balance_col = next(
                (i for i, f in enumerate(credit_fields) if '今日餘額' in f),
                None
            )
            amount_row = next(
                (row for row in credit_list if '融資金額' in str(row[0])),
                None
            )

            if balance_col is None or amount_row is None:
                print('[融資維持率] MI_MARGN(MS) 欄位對不上，實際欄位：', credit_fields)
                continue

            amount = float(str(amount_row[balance_col]).replace(',', '')) * 1000

            if amount > 0:
                total_margin_amount = amount
                matched_date = query_date
                print(f'[融資維持率] 分母取自 {query_date}，金額={amount}')
                break

        except Exception as error:
            print(f'[融資維持率] 分母抓取失敗({back_days}天前)：', repr(error))

    if total_margin_amount is None:
        print('[融資維持率] 找不到有效分母，放棄')
        return None

    # ---- 分子：openapi 每檔個股融資今日餘額 × 收盤價 ----
    try:
        margin_resp = requests.get(
            'https://openapi.twse.com.tw/v1/exchangeReport/MI_MARGN',
            headers=headers,
            timeout=30
        )
        margin_resp.raise_for_status()
        margin_rows = margin_resp.json()

        print('[融資維持率] openapi MI_MARGN 資料筆數：', len(margin_rows))
        if margin_rows:
            print('[融資維持率] openapi MI_MARGN 第一筆範例：', margin_rows[0])

        margin_shares = {}
        for row in margin_rows:
            try:
                code = str(row.get('股票代號', '')).strip()
                shares = float(str(row.get('融資今日餘額', '')).replace(',', '')) * 1000
                if shares > 0:
                    margin_shares[code] = shares
            except (ValueError, TypeError):
                continue

        print('[融資維持率] 有效融資個股數：', len(margin_shares))

        if not margin_shares:
            return None

        price_resp = requests.get(
            'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL',
            headers=headers,
            timeout=30
        )
        price_resp.raise_for_status()
        price_rows = price_resp.json()

        numerator_date = None
        if price_rows:
            numerator_date = price_rows[0].get('Date')
        print(f'[融資維持率] 分子日期(STOCK_DAY_ALL) = {numerator_date}，分母日期(MS) = {matched_date}')
        if numerator_date and matched_date and str(numerator_date) != str(matched_date):
            print('[融資維持率] 警告：分子與分母不是同一天，計算結果可能有落差')

        close_prices = {}
        for row in price_rows:
            try:
                code = str(row.get('Code', '')).strip()
                close = float(str(row.get('ClosingPrice', '')).replace(',', ''))
                close_prices[code] = close
            except (ValueError, TypeError):
                continue

        margin_value = sum(
            shares * close_prices[code]
            for code, shares in margin_shares.items()
            if code in close_prices
        )

        print(
            f'[融資維持率] margin_value={margin_value}, '
            f'total_margin_amount={total_margin_amount}（分母日期 {matched_date}）'
        )

        if margin_value <= 0:
            return None

        return margin_value / total_margin_amount * 100

    except Exception as error:
        print('[融資維持率] 分子抓取失敗：', repr(error))
        return None


def update_market_pe_history(history, pe_value):
    """
    把每次算出來的大盤本益比累積進歷史紀錄，滾動計算近5年平均值與標準差。
    這是用「自己累積」的方式做的，不是抓現成的5年歷史本益比資料庫（免費資料源沒有這個）。
    剛開始累積天數還不夠5年時，樣本數會偏少，回傳的平均/標準差僅供參考，
    等 workflow 累積跑得夠久（理論上要滿5年）數字才會真正穩定。
    """
    key = '大盤本益比'
    pe_hist = history.get(key, {})

    if pe_value is not None:
        today_str = datetime.now(TZ).strftime('%Y-%m-%d')
        pe_hist[today_str] = float(pe_value)

    # 保留視窗設6年（比5年統計窗多留1年緩衝）
    cutoff = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.DateOffset(years=6)
    )
    pe_hist = {
        date_key: value
        for date_key, value in pe_hist.items()
        if pd.to_datetime(date_key) >= cutoff
    }
    history[key] = pe_hist

    five_year_cutoff = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.DateOffset(years=5)
    )
    recent_values = [
        value
        for date_key, value in pe_hist.items()
        if pd.to_datetime(date_key) >= five_year_cutoff
    ]

    # 樣本數太少（例如剛開始累積）時，平均值/標準差不具參考性
    if len(recent_values) < 30:
        return None, None, len(recent_values)

    array = np.array(recent_values)
    return float(array.mean()), float(array.std()), len(recent_values)


def fetch_market_overview(history):
    """
    大盤總覽：加權指數+漲跌幅、大盤本益比(簡單平均，非市值加權，僅供參考)、
    大盤波動率(TAIEX近20日年化歷史波動率)、大盤融資維持率。
    任何一項抓不到都顯示 N/A，不會讓整支程式失敗。
    """
    result = {
        'taiex_price': None,
        'taiex_change_pct': None,
        'market_pe': None,
        'market_vol': None,
        'margin_ratio': None
    }

    # ---- 加權指數 ----
    try:
        live_price, live_prev_close = fetch_taiex_realtime()
        if live_price is None:
            info = yf.Ticker('^TWII').fast_info
            live_price = float(info['last_price'])
            live_prev_close = float(info['previous_close'])

        result['taiex_price'] = live_price
        if live_prev_close:
            result['taiex_change_pct'] = live_price / live_prev_close - 1
    except Exception as error:
        print('加權指數抓取失敗：', repr(error))

    # ---- 大盤波動率（20日年化歷史波動率，log return）----
    try:
        twii_hist = yf.download(
            '^TWII', period='3mo', interval='1d',
            auto_adjust=True, progress=False,
            threads=False, timeout=30
        )
        if isinstance(twii_hist.columns, pd.MultiIndex):
            twii_hist.columns = twii_hist.columns.get_level_values(0)

        twii_ret = np.log(
            twii_hist['Close'] / twii_hist['Close'].shift(1)
        ).dropna()

        result['market_vol'] = float(
            twii_ret.tail(20).std() * np.sqrt(252) * 100
        )
    except Exception as error:
        print('大盤波動率計算失敗：', repr(error))

    # ---- 大盤本益比（改用市值加權：已發行股數 × 收盤價 當權重，接近官方算法）----
    try:
        pe_resp = requests.get(
            'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL',
            timeout=20
        )
        pe_resp.raise_for_status()
        pe_rows = pe_resp.json()

        print('[大盤本益比] openapi BWIBBU_ALL 資料筆數：', len(pe_rows))

        pe_by_code = {}
        for row in pe_rows:
            try:
                code = str(row.get('Code', '')).strip()
                pe = float(str(row.get('PEratio', '')).replace(',', ''))
                # 排除異常極端值（例如剛轉盈、獲利極低的公司本益比會飆到數百倍），
                # 避免少數暴衝股票把加權平均值拉爆。100倍以上視為異常濾除。
                if 0 < pe <= 100:
                    pe_by_code[code] = pe
            except (ValueError, TypeError):
                continue

        if pe_by_code:
            price_resp = requests.get(
                'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL',
                timeout=30
            )
            price_resp.raise_for_status()
            price_rows = price_resp.json()

            print('[大盤本益比] openapi STOCK_DAY_ALL 資料筆數：', len(price_rows))

            close_by_code = {}
            for row in price_rows:
                try:
                    code = str(row.get('Code', '')).strip()
                    close = float(str(row.get('ClosingPrice', '')).replace(',', ''))
                    if close > 0:
                        close_by_code[code] = close
                except (ValueError, TypeError):
                    continue

            shares_resp = requests.get(
                'https://openapi.twse.com.tw/v1/opendata/t187ap03_L',
                timeout=30
            )
            shares_resp.raise_for_status()
            shares_rows = shares_resp.json()

            print('[大盤本益比] openapi t187ap03_L(已發行股數) 資料筆數：', len(shares_rows))

            total_weight = 0.0
            weighted_sum = 0.0

            for row in shares_rows:
                try:
                    code = str(row.get('公司代號', '')).strip()
                    if code not in pe_by_code or code not in close_by_code:
                        continue

                    shares = float(
                        str(row.get('已發行普通股數或TDR原股發行股數', ''))
                        .replace(',', '')
                    )
                    if shares <= 0:
                        continue

                    market_cap = shares * close_by_code[code]
                    weighted_sum += pe_by_code[code] * market_cap
                    total_weight += market_cap
                except (ValueError, TypeError):
                    continue

            if total_weight > 0:
                result['market_pe'] = weighted_sum / total_weight
                print('[大盤本益比] 市值加權計算結果：', result['market_pe'])
    except Exception as error:
        print('大盤本益比抓取失敗：', repr(error))

    pe_mean, pe_std, pe_sample = update_market_pe_history(history, result['market_pe'])
    result['market_pe_mean'] = pe_mean
    result['market_pe_std'] = pe_std
    result['market_pe_sample'] = pe_sample
    print(f'[大盤本益比] 近5年統計：平均={pe_mean}, 標準差={pe_std}, 樣本數={pe_sample}')

    # ---- 大盤融資維持率 ----
    try:
        result['margin_ratio'] = fetch_market_margin_ratio()
    except Exception as error:
        print('大盤融資維持率整體流程失敗：', repr(error))

    return result


def fetch_etf(ticker):
    data = pd.DataFrame()
    raw_data = pd.DataFrame()
    last_error = None

    # 期間拉長到6年（原本只抓18個月），
    # 這樣「1/3/5年報酬率」裡的3年、5年才有真正的歷史資料可以算，
    # 不會因為資料不夠長而退回用「現有資料最早一筆」頂替，導致報酬率失真。
    # 週線圖表顯示範圍不受影響，仍然只取最近53週（weekly_data.tail(53)）。
    for attempt in range(3):
        try:
            data = yf.download(
                ticker,
                period='6y',
                interval='1d',
                auto_adjust=True,
                progress=False,
                threads=False,
                timeout=30
            )

            raw_data = yf.download(
                ticker,
                period='6y',
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

    # ---- 即時報價（跟 Yahoo 網頁上看到的一致，不受歷史K棒延遲影響） ----
    # 優先用證交所(TWSE)的即時資訊API直接抓「最新價/昨收」，
    # 這是跟券商APP同一組資料源，不用透過yfinance轉手。
    # 抓不到才 fallback 用 fast_info，再不行才退回歷史K棒最後一筆。
    # EMA、最高價、回撤、停損價這些仍然用歷史K棒 (daily_adj) 算，不受影響。
    live_price, live_prev_close = fetch_twse_realtime(ticker)

    if live_price is None:
        try:
            live_price = float(yf.Ticker(ticker).fast_info['last_price'])
        except Exception as error:
            print(
                f'{ticker} 即時報價抓取失敗，改用歷史K棒最後一筆：',
                repr(error)
            )

    if live_price is None:
        # fallback：抓不到即時報價時，退回原本用歷史K棒最後一筆的做法
        live_price = float(daily_raw_close.iloc[-1])

    if live_prev_close is None:
        # 「昨收」要抓「即時價那個交易日」的前一個交易日收盤價才對。
        # 比對歷史K棒最後一筆收盤價，是否已經等於即時報價──
        #   - 如果相等，代表歷史K棒已經追上即時價那個交易日，昨收要用「倒數第二筆」。
        #   - 如果不相等，代表歷史K棒還沒追上（延遲），最後一筆本身就是正確的昨收。
        if len(daily_raw_close) >= 2 and abs(
            float(daily_raw_close.iloc[-1]) - live_price
        ) < 1e-6:
            live_prev_close = float(daily_raw_close.iloc[-2])
        else:
            live_prev_close = float(daily_raw_close.iloc[-1])

    return {
        'weekly': weekly_data.dropna().tail(53),
        'daily_adj': data['Close'],
        'daily_raw': daily_raw_close,
        'live_price': live_price,
        'live_prev_close': live_prev_close
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


def multi_year_return_text(dates, values, years_list=(1, 3, 5)):
    """
    計算「N年前的今天」到「今天」的報酬率（N分別取 years_list），
    若N年前的今天不是交易日，自動改用前一個交易日的資料（跟 date_based_stats 邏輯一致）。
    顯示實際數字，不加「1/3/5年」字樣。
    """
    series = (
        pd.Series(list(values), index=pd.DatetimeIndex(dates))
        .sort_index()
    )
    series = series[~series.index.duplicated(keep='last')].dropna()

    latest_date = series.index[-1]
    latest = float(series.iloc[-1])

    labels = []
    for years in years_list:
        target_date = latest_date - pd.DateOffset(years=years)
        base_slice = series[series.index <= target_date]

        if not base_slice.empty:
            base_value = float(base_slice.iloc[-1])
        else:
            base_value = float(series.iloc[0])

        return_pct = (latest / base_value - 1) * 100
        labels.append(f'{return_pct:+.0f}%')

    return f"報酬率 {'／'.join(labels)}"


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


def draw_signal_light(fig, ax, state, label=None, x=0.92, y=0.965, r_px=20):
    colors = {
        'green': (LIGHT_GREEN, LIGHT_GREEN_EDGE),
        'yellow': (LIGHT_YELLOW, LIGHT_YELLOW_EDGE),
        'red': (LIGHT_RED, LIGHT_RED_EDGE)
    }

    fill, edge = colors[state]

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


def quarterly_month_ticks(dates):
    """
    找出「當月」以及往前每隔3個月的月份，回傳可以直接
    丟給 ax.set_xticks / set_xticklabels 用的（位置, 標籤）。

    例如資料最新到7月，就會標 7月、4月、1月、去年10月...
    一路往前推到資料的起始月份為止。
    """
    dates = pd.DatetimeIndex(pd.to_datetime(dates))

    if len(dates) == 0:
        return [], []

    periods = dates.to_period('M')
    earliest_period = periods.min()

    positions = []
    labels = []

    target_period = periods[-1]

    while target_period >= earliest_period:
        matches = np.where(periods == target_period)[0]

        if len(matches) > 0:
            positions.append(int(matches[0]))
            labels.append(str(target_period.month))

        target_period -= 3

    positions.reverse()
    labels.reverse()

    return positions, labels


def plot_fund(ax, name, data, high_1y, fig):
    latest, local_high, _, _ = date_based_stats(
        data['Date'],
        data['Value']
    )

    multi_return_line = multi_year_return_text(data['Date'], data['Value'])

    high = high_1y if high_1y is not None else local_high
    drawdown = latest / high - 1
    add_price = high * 0.8

    if len(data) >= 2:
        change_pct = data['Value'].iloc[-1] / data['Value'].iloc[-2] - 1
    else:
        change_pct = 0.0

    # 走勢圖只畫近一年（1/3/5年報酬率的計算仍然用上面的完整歷史 data，不受影響）
    latest_date = data['Date'].max()
    chart_data = (
        data[data['Date'] >= latest_date - pd.DateOffset(years=1)]
        .reset_index(drop=True)
    )
    x = np.arange(len(chart_data))

    style_card(ax)

    ax.plot(
        x,
        chart_data['Value'],
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

    abs_drawdown = abs(drawdown)

    if abs_drawdown > 0.20:
        fund_state = 'green'
        fund_status = '可以加碼'
    elif abs_drawdown > 0.10:
        fund_state = 'yellow'
        fund_status = '觀察加碼'
    else:
        fund_state = 'red'
        fund_status = '暫停加碼'

    ax.text(
        0.04,
        0.93,
        name,
        transform=ax.transAxes,
        fontsize=30,
        fontweight='bold',
        color=GOLD,
        ha='left',
        va='top',
        zorder=20
    )

    ax.text(
        0.97,
        0.06,
        (
            f'漲跌幅 {change_pct:+.2%}\n'
            f'最新價 {latest:.2f}\n'
            f'最高價 {high:.2f}\n'
            f'加碼價 {add_price:.2f}\n'
            f'回撤 {drawdown:.1%}\n'
            f'{multi_return_line}'
        ),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=24,
        color=TEXT,
        linespacing=1.7
    )

    draw_signal_light(fig, ax, fund_state, label=fund_status)

    ax.grid(alpha=0.08, color=GOLD_DIM, lw=0.6)
    ax.set_xlim(0, max(1, len(x) - 1))

    tick_positions, tick_labels = quarterly_month_ticks(chart_data['Date'])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(
        axis='x',
        labelbottom=True,
        labelsize=20,
        colors=TEXT_DIM,
        length=0
    )


def plot_etf(ax, name, etf_bundle, ema_period, fig):
    data = etf_bundle['weekly']
    x = np.arange(len(data))

    ema = (
        data['Close']
        .ewm(span=ema_period, adjust=False)
        .mean()
    )

    # 「最高價」改用還原週線的盤中最高點 (週K棒的High)，不是收盤價 (Close)。
    # data 本身已經是 fetch_etf() 裡取近一年 (tail 53週) 的還原週線，
    # 所以這裡直接抓這段期間內週K棒的最高High，範圍跟原本一致，只是換了欄位。
    high = float(data['High'].max())

    # 「最新價/漲跌幅」改用即時報價 (fast_info)，跟Yahoo網頁/券商顯示的一致，
    # 不受歷史K棒 (daily_adj) 有時延遲一個交易日才更新的影響。
    latest = etf_bundle['live_price']
    live_prev_close = etf_bundle['live_prev_close']

    drawdown = latest / high - 1

    stop = high * 0.8

    if live_prev_close:
        change_pct = latest / live_prev_close - 1
    else:
        change_pct = 0.0

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

    above_ema = signal_close > signal_ema
    abs_drawdown = abs(drawdown)

    if not above_ema:
        etf_state = 'red'
        status = f'跌破{ema_period}週線'
    elif abs_drawdown > 0.20:
        etf_state = 'yellow'
        status = '暫時離場'
    else:
        etf_state = 'green'
        status = f'站上{ema_period}週線'

    ax.text(
        0.04,
        0.93,
        name,
        transform=ax.transAxes,
        fontsize=30,
        fontweight='bold',
        color=GOLD,
        ha='left',
        va='top',
        zorder=20
    )

    # 「1／3／5年報酬率」用還原日線收盤價算，N年前的今天→今天(遇非交易日自動用最後交易日)
    multi_return_line = multi_year_return_text(
        etf_bundle['daily_adj'].index,
        etf_bundle['daily_adj'].values
    )

    ax.text(
        0.97,
        0.06,
        (
            f'漲跌幅 {change_pct:+.2%}\n'
            f'最新價 {latest:.2f}\n'
            f'最高價 {high:.2f}\n'
            f'停損價 {stop:.2f}\n'
            f'回撤 {drawdown:.1%}\n'
            f'{multi_return_line}'
        ),
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontsize=24,
        color=TEXT,
        linespacing=1.65
    )

    draw_signal_light(fig, ax, etf_state, label=status)

    ax.grid(alpha=0.08, color=GOLD_DIM, lw=0.6)
    ax.set_xlim(-1, len(x))

    tick_positions, tick_labels = quarterly_month_ticks(data.index)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.tick_params(
        axis='x',
        labelbottom=True,
        labelsize=20,
        colors=TEXT_DIM,
        length=0
    )


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
        height_ratios=[1.25, 4.475, 4.475],
        hspace=0.04,
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

    history = load_history()

    market = fetch_market_overview(history)

    def fmt(value, suffix='', digits=2):
        if value is None:
            return 'N/A'
        return f'{value:,.{digits}f}{suffix}'

    def metric_state(kind, value, pe_mean=None, pe_std=None):
        if kind == 'pe':
            if pe_mean is None or pe_std is None or pe_std <= 0:
                return 'yellow'  # 樣本數不足5年統計，無法判斷，先顯示中性黃燈
            if value > pe_mean + pe_std:
                return 'red'
            if value < pe_mean - pe_std:
                return 'green'
            return 'yellow'
        if kind == 'margin':
            if value >= 160:
                return 'green'
            if value >= 130:
                return 'yellow'
            return 'red'
        if kind == 'vol':
            if value < 20:
                return 'green'
            if value <= 30:
                return 'yellow'
            return 'red'
        return 'yellow'

    if market['taiex_change_pct'] is not None:
        taiex_line = (
            f"加權指數 {fmt(market['taiex_price'], digits=2)} "
            f"({market['taiex_change_pct']*100:+.2f}%)"
        )
    else:
        taiex_line = f"加權指數 {fmt(market['taiex_price'], digits=2)}"

    title_ax.text(
        0.03,
        0.95,
        taiex_line,
        fontsize=30,
        fontweight='bold',
        ha='left',
        va='top',
        color=GOLD,
        alpha=0.95
    )

    metric_rows = [
        ('大盤本益比', market['market_pe'], '', 'pe', ''),
        ('大盤波動率', market['market_vol'], '%', 'vol', ''),
        ('大盤融資維持率', market['margin_ratio'], '%', 'margin', ''),
    ]

    row_ys = [0.63, 0.36, 0.09]

    for (label, value, suffix, kind, note), y in zip(metric_rows, row_ys):
        if value is not None:
            draw_signal_light(
                fig, title_ax,
                metric_state(
                    kind, value,
                    pe_mean=market.get('market_pe_mean'),
                    pe_std=market.get('market_pe_std')
                ),
                x=0.03, y=y, r_px=13
            )

        title_ax.text(
            0.06,
            y,
            f"{label} {fmt(value, suffix=suffix)}{note}",
            fontsize=24,
            ha='left',
            va='center',
            color=TEXT_DIM,
            alpha=0.95
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
            fund_data = fetch_fund(fund['url'])

            high_1y = update_history_and_get_high(
                history,
                fund['name'],
                fund_data
            )

            chart_data = history_to_chart_data(
                history[fund['name']]
            )

            plot_fund(
                ax,
                fund['display'],
                chart_data,
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
                fund['display'],
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

    save_history(history)

    for ax, etf in zip(etf_axes, ETFS):
        try:
            etf_data = fetch_etf(etf['ticker'])

            plot_etf(
                ax,
                etf['display'],
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
                etf['display'],
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







