import os, re, time, json
import io
import zipfile
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import pdfplumber

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
MARKET_PE_CSV = 'market_pe_history.csv'


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
        'ema': 32,
        'stop_days': 40
    },
    {
        'name': '00830',
        'display': '費半',
        'ticker': '00830.TW',
        'ema': 42,
        'stop_days': 60
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


def format_date_suffix(date_value):
    """
    把各種日期格式(西元8碼如 20260728、民國7碼如 1150728)統一轉成
    畫面上要顯示的「（MM/DD）」後綴，方便標示每項數據實際對應的資料日期。
    無法辨識或空值時回傳空字串，不影響原本畫面排版。
    """
    if not date_value:
        return ''

    digits = re.sub(r'\D', '', str(date_value))

    try:
        if len(digits) == 7:
            month, day = digits[3:5], digits[5:7]
        elif len(digits) == 8:
            month, day = digits[4:6], digits[6:8]
        else:
            return ''
        return f'（{month}/{day}）'
    except Exception:
        return ''


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


def load_market_pe_csv():
    """
    讀取大盤本益比的完整每日歷史紀錄(CSV，欄位：date, pe)。
    這份檔案用「全部歷史」(不像history.json裡的舊邏輯只保留近5~6年)，
    平均值/標準差/百分位都是用這份CSV的全部資料算出來的。
    找不到檔案就回傳空的DataFrame，之後會自然從頭開始累積。
    """
    if not os.path.exists(MARKET_PE_CSV):
        return pd.DataFrame(columns=['date', 'pe'])

    try:
        df = pd.read_csv(MARKET_PE_CSV, dtype={'date': str})
        df['pe'] = pd.to_numeric(df['pe'], errors='coerce')
        df = df.dropna(subset=['pe'])
        df = df.drop_duplicates(subset='date', keep='last')
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as error:
        print('讀取market_pe_history.csv失敗，改用空白重新累積：', repr(error))
        return pd.DataFrame(columns=['date', 'pe'])


def save_market_pe_csv(df):
    try:
        df.sort_values('date').to_csv(MARKET_PE_CSV, index=False)
    except Exception as error:
        print('寫入market_pe_history.csv失敗：', repr(error))


def append_or_update_market_pe(date_str, pe_value):
    """
    今天已存在就更新，不存在就新增一筆，不會重複。
    回傳更新後的完整DataFrame。
    """
    df = load_market_pe_csv()

    if pe_value is None:
        return df

    if (df['date'] == date_str).any():
        df.loc[df['date'] == date_str, 'pe'] = pe_value
    else:
        df = pd.concat(
            [df, pd.DataFrame([{'date': date_str, 'pe': pe_value}])],
            ignore_index=True
        )

    df = df.drop_duplicates(subset='date', keep='last').sort_values('date').reset_index(drop=True)
    save_market_pe_csv(df)
    return df


def compute_market_pe_stats(df, today_pe):
    """
    用CSV裡的「全部歷史資料」算：
    - mean, std（方法一的紅綠燈門檻用）
    - z-score（今日PE距離平均值幾個標準差）
    - percentile（今日PE在全部歷史資料中的百分位排名，0~100）
    樣本數太少(<30筆)時，統計量不具參考性，回傳None。
    """
    if today_pe is None or df.empty or len(df) < 30:
        return None, None, None, None, len(df) if not df.empty else 0

    values = df['pe'].to_numpy()
    mean = float(values.mean())
    std = float(values.std())

    z_score = (today_pe - mean) / std if std > 0 else None
    percentile = float((values <= today_pe).mean() * 100)

    return mean, std, z_score, percentile, len(df)


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


def fetch_foreign_net_sell():
    """
    外資（不含投信、自營商）當日買賣超金額(元)。
    資料來源：www.twse.com.tw/fund/BFI82U（三大法人買賣金額統計表），
    這支報表裡有分開列出外資、投信、自營商各自的買賣差額，
    只取「外資」那一列，不是三大法人合計。
    格式跟MI_MARGN舊版一樣可能包在'tables'裡，也支援date參數查歷史。
    這支URL/欄位名稱沒辦法連線驗證，抓不到會印出實際回傳內容方便校正。
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Linux; Android 13) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/126.0 Mobile Safari/537.36'
        )
    }

    for back_days in range(6):
        try:
            query_date = (
                datetime.now(TZ) - pd.Timedelta(days=back_days)
            ).strftime('%Y%m%d')

            resp = requests.get(
                'https://www.twse.com.tw/fund/BFI82U',
                params={'response': 'json', 'dayDate': query_date, 'type': 'day'},
                headers=headers,
                timeout=20
            )
            resp.raise_for_status()
            payload = resp.json()

            print(f'[外資賣超] {query_date} 回傳內容：', payload)

            tables = payload.get('tables') or []
            if tables:
                fields = tables[0].get('fields') or []
                data_rows = tables[0].get('data') or []
            else:
                fields = payload.get('fields') or []
                data_rows = payload.get('data') or []

            if not fields or not data_rows:
                print(f'[外資賣超] {query_date} 沒有資料，可能非交易日')
                continue

            print(f'[外資賣超] {query_date} 欄位：', fields)

            diff_col = next(
                (i for i, f in enumerate(fields) if '買賣差額' in f or '買賣超' in f),
                None
            )

            if diff_col is None:
                print(f'[外資賣超] {query_date} 找不到買賣差額欄位，實際欄位如上')
                continue

            # 精確鎖定「外資及陸資」這一列。注意：這一列的完整名稱是
            # 「外資及陸資(不含外資自營商)」，本身就含有「自營商」三個字
            # (用來說明它不包含什麼)，不能用「不含自營商」當排除條件，
            # 只要判斷開頭是「外資及陸資」就好，不會跟單獨的「外資自營商」那列搞混。
            foreign_row = next(
                (
                    row for row in data_rows
                    if str(row[0]).strip().startswith('外資及陸資')
                ),
                None
            )

            if foreign_row is None:
                print(f'[外資賣超] {query_date} 找不到外資及陸資那一列，實際資料：', data_rows)
                continue

            net_value = float(str(foreign_row[diff_col]).replace(',', ''))
            print(f'[外資賣超] {query_date} 外資買賣差額(元)：{net_value}')
            return net_value, query_date

        except Exception as error:
            print(f'[外資賣超] 抓取失敗({back_days}天前)：', repr(error))

    return None, None


def fetch_foreign_futures_net_oi():
    """
    外資在臺股期貨的未平倉「淨部位」口數(多方未平倉 - 空方未平倉)。
    負數代表淨空單，正數代表淨多單，跟各券商/看盤軟體顯示的「外資未平倉」數字定義一致
    (不是空方未平倉的總口數，那是完全不同、恆為正值的另一個欄位)。
    資料來源：openapi.taifex.com.tw/v1/MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate
    （三大法人-區分各期貨契約-依日期），只回傳最新一個交易日，不支援查歷史日期。
    這支API的ContractCode/Item實際文字內容沒辦法連線驗證，抓不到會印出完整資料方便校正。
    """
    try:
        resp = requests.get(
            'https://openapi.taifex.com.tw/v1/MarketDataOfMajorInstitutionalTradersDetailsOfFuturesContractsBytheDate',
            timeout=30
        )
        resp.raise_for_status()
        rows = resp.json()

        print('[外資期貨未平倉] 資料筆數：', len(rows))
        if rows:
            print('[外資期貨未平倉] 第一筆範例：', rows[0])

        if not rows:
            return None, None

        target_row = next(
            (
                row for row in rows
                if '臺股期貨' in str(row.get('ContractCode', ''))
                and '外資' in str(row.get('Item', ''))
            ),
            None
        )

        if target_row is None:
            print('[外資期貨未平倉] 找不到臺股期貨+外資的資料列，實際欄位範例：', rows[0] if rows else None)
            return None, None

        net_oi = float(str(target_row.get('OpenInterest(Net)', '')).replace(',', ''))
        query_date = str(target_row.get('Date', '')).strip()

        print(f'[外資期貨未平倉] {query_date} 淨部位口數：{net_oi}')
        return net_oi, query_date

    except Exception as error:
        print('[外資期貨空單] 抓取失敗：', repr(error))
        return None, None


def fetch_institutional_net_buy():
    """
    三大法人(外資、投信、自營商)合計買賣超金額(元)。
    資料來源：www.twse.com.tw/fund/BFI82U（三大法人買賣金額統計表），
    格式跟MI_MARGN舊版一樣可能包在'tables'裡，也支援date參數查歷史。
    這支URL/欄位名稱沒辦法連線驗證，抓不到會印出實際回傳內容方便校正。
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Linux; Android 13) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/126.0 Mobile Safari/537.36'
        )
    }

    for back_days in range(6):
        try:
            query_date = (
                datetime.now(TZ) - pd.Timedelta(days=back_days)
            ).strftime('%Y%m%d')

            resp = requests.get(
                'https://www.twse.com.tw/fund/BFI82U',
                params={'response': 'json', 'dayDate': query_date, 'type': 'day'},
                headers=headers,
                timeout=20
            )
            resp.raise_for_status()
            payload = resp.json()

            print(f'[法人買賣超] {query_date} 回傳內容：', payload)

            tables = payload.get('tables') or []
            if tables:
                fields = tables[0].get('fields') or []
                data_rows = tables[0].get('data') or []
            else:
                fields = payload.get('fields') or []
                data_rows = payload.get('data') or []

            if not fields or not data_rows:
                print(f'[法人買賣超] {query_date} 沒有資料，可能非交易日')
                continue

            print(f'[法人買賣超] {query_date} 欄位：', fields)

            diff_col = next(
                (i for i, f in enumerate(fields) if '買賣差額' in f or '買賣超' in f),
                None
            )

            if diff_col is None:
                print(f'[法人買賣超] {query_date} 找不到買賣差額欄位，實際欄位如上')
                continue

            total_net = 0.0
            for row in data_rows:
                try:
                    total_net += float(str(row[diff_col]).replace(',', ''))
                except (ValueError, IndexError, TypeError):
                    continue

            print(f'[法人買賣超] {query_date} 合計淨額(元)：{total_net}')
            return total_net, query_date

        except Exception as error:
            print(f'[法人買賣超] 抓取失敗({back_days}天前)：', repr(error))

    return None, None


def fetch_market_margin_ratio():
    """
    大盤融資維持率 = Σ(個股融資今日餘額(股) × 收盤價) / 大盤融資金額今日餘額(元)

    分子：openapi.twse.com.tw/v1/exchangeReport/MI_MARGN——真正「每檔個股」的
         融資餘額（舊版selectType=ALL其實仍是集中市場加總，不是個股，踩了一次坑）。
         OpenAPI版本不支援查歷史日期，永遠只回傳最新一個交易日，這天是多少
         由TWSE自己的更新進度決定，我們沒辦法指定。

    分母：www.twse.com.tw 舊版 MI_MARGN?selectType=MS（集中市場信用交易統計彙總，
         'tables'結構包著'融資金額(仟元)'的今日餘額）。這支支援date參數查歷史，
         所以改成「分子有哪一天的資料，就去抓同一天的分母」，而不是各自抓各自
         最新的，避免兩邊日期對不上時algorithm混算出一個兩邊都不是的假數字
         (之前173%/180%失真的根因就是這裡)。
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Linux; Android 13) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/126.0 Mobile Safari/537.36'
        )
    }

    def to_western_yyyymmdd(date_value):
        """把民國年(如1150724，7碼)或西元年(如20260724，8碼)統一轉成西元8碼字串。"""
        digits = str(date_value).strip()
        if len(digits) == 7:
            roc_year = int(digits[:3])
            return f'{roc_year + 1911}{digits[3:]}'
        return digits

    # ---- 分子：openapi 每檔個股融資今日餘額 × 收盤價（這天由TWSE決定，我們只能接受）----
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
            return None, None

        price_resp = requests.get(
            'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL',
            headers=headers,
            timeout=30
        )
        price_resp.raise_for_status()
        price_rows = price_resp.json()

        numerator_date = price_rows[0].get('Date') if price_rows else None
        if not numerator_date:
            print('[融資維持率] STOCK_DAY_ALL 沒有日期欄位，放棄')
            return None, None

        target_date = to_western_yyyymmdd(numerator_date)
        print(f'[融資維持率] 分子日期(STOCK_DAY_ALL) = {numerator_date} → {target_date}')

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

        if margin_value <= 0:
            return None, None

    except Exception as error:
        print('[融資維持率] 分子抓取失敗：', repr(error))
        return None, None

    # ---- 分母：強制抓「跟分子同一天」的大盤融資金額今日餘額，兩邊日期保證對齊 ----
    # 保留小幅往前找的容錯（最多3天），只為了應付當天MS報表暫時抓取失敗
    # (例如502)的狀況，而不是為了對齊日期去找別的日期。
    total_margin_amount = None
    matched_date = None

    for back_days in range(3):
        query_date = (
            pd.Timestamp(target_date) - pd.Timedelta(days=back_days)
        ).strftime('%Y%m%d')

        try:
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
                print(f'[融資維持率] {query_date} MS 沒有tables，重試更早的日期')
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
            print(f'[融資維持率] 分母抓取失敗({query_date})：', repr(error))

    if total_margin_amount is None or matched_date != target_date:
        print(
            f'[融資維持率] 找不到與分子同一天({target_date})的分母資料，'
            '放棄計算(避免跨日期混算出失真數字)'
        )
        return None, None

    print(
        f'[融資維持率] margin_value={margin_value}, '
        f'total_margin_amount={total_margin_amount}（日期 {matched_date}，分子分母已對齊）'
    )

    return margin_value / total_margin_amount * 100, matched_date


def fetch_otc_margin_ratio():
    """
    上櫃(TPEx)大盤融資維持率，算法跟fetch_market_margin_ratio()一樣：
    Σ(個股融資今日餘額(股) × 收盤價) / 上櫃融資金額今日餘額(元)

    TPEx目前openapi文件版本可能會變動，這裡先用最可能的端點嘗試，
    抓不到就印出實際回傳內容/欄位，方便照著校正。
    """
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Linux; Android 13) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/126.0 Mobile Safari/537.36'
        )
    }

    # ---- 分子：上櫃個股融資今日餘額 × 收盤價 ----
    try:
        margin_resp = requests.get(
            'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_trading_status',
            headers=headers,
            timeout=30
        )
        margin_resp.raise_for_status()
        margin_rows = margin_resp.json()

        print('[上櫃融資維持率] API資料筆數：', len(margin_rows) if margin_rows else 0)
        if margin_rows:
            print('[上櫃融資維持率] 第一筆範例：', margin_rows[0])

        margin_shares = {}
        close_prices = {}
        for row in margin_rows:
            try:
                code = str(row.get('SecuritiesCompanyCode') or row.get('Code') or '').strip()
                shares_raw = row.get('MarginPurchaseTodayBalance') or row.get('TodayBalance')
                close_raw = row.get('Close') or row.get('ClosingPrice')
                if shares_raw is None or close_raw is None:
                    continue
                shares = float(str(shares_raw).replace(',', '')) * 1000
                close = float(str(close_raw).replace(',', ''))
                if shares > 0:
                    margin_shares[code] = shares
                    close_prices[code] = close
            except (ValueError, TypeError):
                continue

        print('[上櫃融資維持率] 有效融資個股數：', len(margin_shares))

        if not margin_shares:
            print('[上櫃融資維持率] 抓不到個股資料，實際欄位：',
                  list(margin_rows[0].keys()) if margin_rows else '（空）')
            return None, None

        margin_value = sum(
            shares * close_prices[code]
            for code, shares in margin_shares.items()
            if code in close_prices
        )

        if margin_value <= 0:
            return None, None

    except Exception as error:
        print('[上櫃融資維持率] 分子抓取失敗：', repr(error))
        return None, None

    # ---- 分母：上櫃融資金額今日餘額彙總 ----
    try:
        summary_resp = requests.get(
            'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_trading_summary',
            headers=headers,
            timeout=30
        )
        summary_resp.raise_for_status()
        summary_data = summary_resp.json()

        print('[上櫃融資維持率] 彙總API回傳：', summary_data)

        row = summary_data[0] if isinstance(summary_data, list) else summary_data
        amount_raw = row.get('MarginPurchaseTodayBalanceAmount') or row.get('MarginBalanceAmount')
        query_date = row.get('Date') or row.get('date')

        if amount_raw is None:
            print('[上櫃融資維持率] 彙總欄位對不上，實際欄位：', list(row.keys()))
            return None, None

        total_margin_amount = float(str(amount_raw).replace(',', '')) * 1000

    except Exception as error:
        print('[上櫃融資維持率] 分母抓取失敗：', repr(error))
        return None, None

    if total_margin_amount <= 0:
        return None, None

    print(f'[上櫃融資維持率] margin_value={margin_value}, total_margin_amount={total_margin_amount}')

    return margin_value / total_margin_amount * 100, query_date


def update_market_metric_history(history, key, value):
    """
    通用版：把每次算出來的大盤指標(本益比、股淨比等)累積進歷史紀錄，
    滾動計算近5年平均值與標準差。這是用「自己累積」的方式做的，
    不是抓現成的5年歷史資料庫（免費資料源沒有這個）。
    剛開始累積天數還不夠5年時，樣本數會偏少，回傳的平均/標準差僅供參考，
    等 workflow 累積跑得夠久（理論上要滿5年）數字才會真正穩定。
    """
    metric_hist = history.get(key, {})

    if value is not None:
        today_str = datetime.now(TZ).strftime('%Y-%m-%d')
        metric_hist[today_str] = float(value)

    # 保留視窗設6年（比5年統計窗多留1年緩衝）
    cutoff = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.DateOffset(years=6)
    )
    metric_hist = {
        date_key: v
        for date_key, v in metric_hist.items()
        if pd.to_datetime(date_key) >= cutoff
    }
    history[key] = metric_hist

    five_year_cutoff = (
        pd.Timestamp.now(tz=TZ).tz_localize(None)
        - pd.DateOffset(years=5)
    )
    recent_values = [
        v
        for date_key, v in metric_hist.items()
        if pd.to_datetime(date_key) >= five_year_cutoff
    ]

    # 樣本數太少（例如剛開始累積）時，平均值/標準差不具參考性
    if len(recent_values) < 30:
        return None, None, len(recent_values)

    array = np.array(recent_values)
    return float(array.mean()), float(array.std()), len(recent_values)


def update_market_pe_history(history, pe_value):
    """大盤本益比專用包裝(沿用通用函式)，保留原函式名稱給既有呼叫端使用。"""
    return update_market_metric_history(history, '大盤本益比', pe_value)


def update_market_pb_history(history, pb_value):
    """大盤股淨比專用包裝(沿用通用函式)。"""
    return update_market_metric_history(history, '大盤股淨比', pb_value)


def _normalize_period(value):
    """
    把各種可能出現的日期/年月格式，統一轉成西元YYYYMM整數，方便跨檔案合併排序。
    支援：
    - 6碼西元年月，例如 202506 -> 202506
    - 5碼民國年月，例如 11506 -> 202506 (115+1911=2026, 06月)
    - 4碼西元年(只有年沒有月，理論上不該出現在月資料，當作無法辨識)
    - 帶斜線或連字號的日期字串(如 2026/06、2026-06)，會先去除非數字字元再判斷
    無法辨識就回傳 None，這筆資料會被跳過，不會混進合併結果。
    """
    if value is None:
        return None
    digits = re.sub(r'\D', '', str(value))
    if len(digits) == 6:
        return int(digits)
    if len(digits) == 5:
        roc_year = int(digits[:3])
        month = int(digits[3:5])
        return (roc_year + 1911) * 100 + month
    if len(digits) == 7:
        # 極少數情況日期含日(YYYYMMDD)，只取年月
        return int(digits[:6])
    return None


def fetch_market_revenue_yoy():
    """
    上市公司整體當月營收年增率：Σ當月營收 / Σ去年當月營收 - 1
    資料來源：openapi.twse.com.tw/v1/opendata/t187ap05_L（上市公司每月營收彙總表）。
    這是月資料，每月中旬左右才會更新一次，不是每天都會變動，
    平常時間看到的數字會是「最近一次公布的那個月」，不是即時的。
    這支資料源單一、乾淨，不需要跨檔案合併，資料新鮮度也比國發會那份zip穩定。
    """
    try:
        resp = requests.get(
            'https://openapi.twse.com.tw/v1/opendata/t187ap05_L',
            timeout=30
        )
        resp.raise_for_status()
        rows = resp.json()

        print('[上市公司YoY] t187ap05_L 資料筆數：', len(rows))
        if rows:
            print('[上市公司YoY] 第一筆範例：', rows[0])

        if not rows:
            return None, None

        sample_keys = list(rows[0].keys())
        this_key = next(
            (k for k in sample_keys
             if '當月營收' in k and '去年' not in k and '累計' not in k),
            None
        )
        last_year_key = next(
            (k for k in sample_keys
             if '去年' in k and '當月' in k and '累計' not in k),
            None
        )
        period_key = next(
            (k for k in sample_keys if '年月' in k),
            None
        )

        if this_key is None or last_year_key is None:
            print('[上市公司YoY] 欄位對不上，實際欄位：', sample_keys)
            return None, None

        total_this = 0.0
        total_last = 0.0
        period = None

        for row in rows:
            try:
                this_val = float(str(row.get(this_key, '')).replace(',', ''))
                last_val = float(str(row.get(last_year_key, '')).replace(',', ''))
                if this_val > 0 and last_val > 0:
                    total_this += this_val
                    total_last += last_val
                    if period is None and period_key:
                        period = row.get(period_key)
            except (ValueError, TypeError):
                continue

        if total_last <= 0:
            print('[上市公司YoY] 加總後分母為0，放棄')
            return None, None

        yoy = (total_this / total_last - 1) * 100
        print(f'[上市公司YoY] 資料年月={period}，YoY={yoy}')
        return yoy, period

    except Exception as error:
        print('[上市公司YoY] 抓取失敗：', repr(error))
        return None, None


SOX_LIST_URL = 'https://www.nasdaq.com/docs/SOX'
SEC_TICKER_MAP_URL = 'https://www.sec.gov/files/company_tickers.json'

# SEC規定呼叫data.sec.gov必須附帶可辨識身份的User-Agent(含聯絡方式)，
# 否則容易被判定為未表明身份的爬蟲而擋掉。如需更換聯絡email，直接改這裡即可。
SEC_HEADERS = {
    'User-Agent': 'investment-dashboard a0980059350@github (personal wallpaper project)'
}

# 常見的營收XBRL標記，不同公司/不同準則(美國GAAP或國際IFRS，
# 外國發行人如ASML/ARM/TSM常用IFRS申報)打的tag名稱不一定相同，
# 依序嘗試，抓到第一個有資料的就用。
REVENUE_TAGS = [
    ('us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax'),
    ('us-gaap', 'RevenueFromContractWithCustomerIncludingAssessedTax'),
    ('us-gaap', 'Revenues'),
    ('us-gaap', 'SalesRevenueNet'),
    ('us-gaap', 'SalesRevenueGoodsNet'),
    ('ifrs-full', 'Revenue'),
    ('ifrs-full', 'RevenueFromContractsWithCustomers'),
]


def fetch_sox_constituents():
    """
    抓費半(SOX，PHLX Semiconductor Sector Index)完整30檔成分股與權重。
    資料來源：Nasdaq官方PDF(不需登入)。這份PDF網址固定，
    Nasdaq會定期更新內容反映目前最新成分股，不需要另外維護清單。

    回傳：[{'ticker': 'NVDA', 'weight': 10.23}, ...]，失敗回傳空list。
    """
    try:
        resp = requests.get(SOX_LIST_URL, timeout=30)
        resp.raise_for_status()

        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            full_text = '\n'.join(
                (page.extract_text() or '') for page in pdf.pages
            )

        constituents = []
        seen_tickers = set()
        for line in full_text.splitlines():
            tokens = line.split()
            if len(tokens) < 2:
                continue
            ticker_candidate = tokens[-2]
            weight_candidate = tokens[-1]
            if not re.fullmatch(r'[A-Z]{1,6}', ticker_candidate):
                continue
            if not re.fullmatch(r'\d{1,3}\.\d{1,2}', weight_candidate):
                continue
            if ticker_candidate in seen_tickers:
                continue
            seen_tickers.add(ticker_candidate)
            constituents.append({
                'ticker': ticker_candidate,
                'weight': float(weight_candidate)
            })

        print(f'[Semi YoY][SOX] 解析出成分股數：{len(constituents)}')
        return constituents

    except Exception as error:
        print('[Semi YoY][SOX] 成分股清單抓取/解析失敗：', repr(error))
        return []


def fetch_sec_ticker_cik_map():
    """
    抓SEC官方「股票代號 -> CIK」對照表(全市場，一次抓，重複利用)。
    """
    try:
        resp = requests.get(SEC_TICKER_MAP_URL, headers=SEC_HEADERS, timeout=30)
        resp.raise_for_status()
        raw = resp.json()

        ticker_to_cik = {}
        for entry in raw.values():
            ticker = str(entry.get('ticker', '')).upper()
            cik = entry.get('cik_str')
            if ticker and cik is not None:
                ticker_to_cik[ticker] = str(cik).zfill(10)

        print(f'[Semi YoY][SOX] SEC ticker/CIK對照表筆數：{len(ticker_to_cik)}')
        return ticker_to_cik

    except Exception as error:
        print('[Semi YoY][SOX] SEC ticker/CIK對照表抓取失敗：', repr(error))
        return {}


def fetch_company_revenue_yoy(cik):
    """
    抓單一公司(用CIK)的營收資料，自動判斷是「季報公司」還是「年報公司」：
    - 若該公司有10-Q(季報)資料 -> 用最新一季 vs 去年同一季
    - 若該公司只有10-K/20-F/40-F(年報)資料，沒有10-Q
      (常見於外國私人發行人，如ASML/ARM/台積電這類公司) -> 用最新一年 vs 去年同一年
    這個判斷是「看資料本身有沒有10-Q」，不是寫死哪幾家公司是外國發行人，
    所以未來SOX成分股更換時，同一套邏輯仍然適用，不需要另外維護名單。

    回傳 (yoy_pct, latest_period_end, duration_days, filer_type) 或
    None(抓不到可比較資料時回傳None)，filer_type為'季報'或'年報'。
    """
    try:
        resp = requests.get(
            f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json',
            headers=SEC_HEADERS, timeout=30
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as error:
        print(f'[Semi YoY][SOX] CIK{cik} companyfacts抓取失敗：', repr(error))
        return None

    facts = data.get('facts', {})

    entries = []
    for taxonomy, tag in REVENUE_TAGS:
        tag_data = facts.get(taxonomy, {}).get(tag)
        if not tag_data:
            continue
        units = tag_data.get('units', {})
        # 優先用USD，沒有的話就用該tag底下隨便一種幣別
        # (YoY是比率，同一家公司前後兩期幣別一致，不影響計算)
        unit_key = 'USD' if 'USD' in units else next(iter(units), None)
        if unit_key is None:
            continue
        for item in units[unit_key]:
            start = item.get('start')
            end = item.get('end')
            val = item.get('val')
            form = item.get('form', '')
            if not start or not end or val is None:
                continue
            if '10-K' not in form and '10-Q' not in form and '20-F' not in form and '40-F' not in form:
                continue
            entries.append({'start': start, 'end': end, 'val': val, 'form': form})
        if entries:
            break  # 抓到第一個有資料的tag就不再嘗試其他tag

    if not entries:
        return None

    # 依「這家公司有沒有10-Q(季報)資料」自動判斷是季報公司還是年報公司
    # (外國發行人如ASML/ARM/TSM通常只申報20-F年報，沒有10-Q，
    #  不寫死是哪幾家，未來成分股換了也能自動適用)
    quarterly_entries = [e for e in entries if '10-Q' in e['form']]
    is_quarterly_filer = len(quarterly_entries) > 0

    if is_quarterly_filer:
        candidate_entries = quarterly_entries
        expected_duration = 91  # 一季約91天
        duration_tolerance = 20
    else:
        candidate_entries = [e for e in entries if ('10-K' in e['form']) or ('20-F' in e['form']) or ('40-F' in e['form'])]
        expected_duration = 365  # 一年約365天
        duration_tolerance = 20

    if not candidate_entries:
        return None

    # 只保留期間長度符合預期(季報~91天/年報~365天)的資料，避免混入累計數字
    filtered = []
    for item in candidate_entries:
        try:
            item_start = datetime.strptime(item['start'], '%Y-%m-%d')
            item_end = datetime.strptime(item['end'], '%Y-%m-%d')
        except ValueError:
            continue
        duration = (item_end - item_start).days
        if abs(duration - expected_duration) <= duration_tolerance:
            filtered.append({**item, '_start_dt': item_start, '_end_dt': item_end, '_duration': duration})

    if not filtered:
        return None

    filtered.sort(key=lambda e: e['_end_dt'])
    latest = filtered[-1]
    duration_days = latest['_duration']

    # 找「大約一年前、期間長度相近」的那一筆
    target_end_low = latest['_end_dt'] - pd.Timedelta(days=380)
    target_end_high = latest['_end_dt'] - pd.Timedelta(days=350)

    prior_candidates = [
        item for item in filtered
        if target_end_low <= item['_end_dt'] <= target_end_high
    ]

    if not prior_candidates:
        return None

    prior = max(prior_candidates, key=lambda e: e['_end_dt'])

    if prior['val'] == 0:
        return None

    yoy = (latest['val'] / prior['val'] - 1) * 100
    filer_type = '季報' if is_quarterly_filer else '年報'
    return yoy, latest['end'], duration_days, filer_type


def fetch_semi_yoy():
    """
    費半(SOX，PHLX Semiconductor Sector Index)成分股「市值加權」營收年增率。
    資料來源：
      - 成分股清單：Nasdaq官方PDF(只取用來知道「目前是哪30家」，不使用其公布的權重)
      - 市值：yfinance即時報價
      - 個別公司營收：SEC EDGAR官方XBRL資料(data.sec.gov)

    每家公司各自判斷季報/年報並算出自己的YoY%，
    再用「即時市值」做加權平均(不是用Nasdaq公布的指數權重)，
    得到整體的市值加權YoY%。

    若成功取得的公司「市值」總和低於全部30家市值總和的50%，
    視為樣本不足，放棄本次結果，回傳(None, None)，
    由呼叫端沿用history.json裡上一次成功抓到的數值。

    成分股清單每次都重新抓取Nasdaq最新公告，若SOX調整成分股(增減公司)，
    下次執行會自動反映最新名單，不需要手動維護。
    """
    constituents = fetch_sox_constituents()
    if not constituents:
        return None, None

    ticker_to_cik = fetch_sec_ticker_cik_map()
    if not ticker_to_cik:
        return None, None

    market_caps = {}
    for company in constituents:
        ticker = company['ticker']
        try:
            info = yf.Ticker(ticker).fast_info
            cap = info.get('market_cap') if hasattr(info, 'get') else info['market_cap']
            if cap:
                market_caps[ticker] = float(cap)
        except Exception as error:
            print(f'[Semi YoY][SOX] {ticker} 市值抓取失敗：', repr(error))

    total_cap = sum(market_caps.values())
    if total_cap <= 0:
        print('[Semi YoY][SOX] 全部成分股市值都抓不到，放棄本次結果')
        return None, None

    matched_cap = 0.0
    weighted_yoy_sum = 0.0
    success_count = 0
    latest_ends = []

    for company in constituents:
        ticker = company['ticker']

        cap = market_caps.get(ticker)
        if cap is None:
            print(f'[Semi YoY][SOX] {ticker} 沒有市值資料，跳過')
            continue

        cik = ticker_to_cik.get(ticker)
        if cik is None:
            print(f'[Semi YoY][SOX] {ticker} 在SEC對照表裡找不到CIK，跳過')
            continue

        result = fetch_company_revenue_yoy(cik)
        time.sleep(0.15)  # 禮貌性間隔，SEC規定上限為每秒10次請求

        if result is None:
            print(f'[Semi YoY][SOX] {ticker} 抓不到可比較的營收資料，跳過')
            continue

        yoy, period_end, duration_days, filer_type = result
        weighted_yoy_sum += yoy * cap
        matched_cap += cap
        success_count += 1
        latest_ends.append(period_end)
        print(
            f'[Semi YoY][SOX] {ticker}({filer_type}) YoY={yoy:.2f}%，'
            f'期間長度={duration_days}天，資料截至={period_end}，市值={cap:,.0f}'
        )

    print(
        f'[Semi YoY][SOX] 成功家數：{success_count}/{len(constituents)}，'
        f'涵蓋市值：{matched_cap:,.0f}/{total_cap:,.0f}'
        f'（{matched_cap / total_cap * 100:.1f}%）'
    )

    if matched_cap < total_cap * 0.5:
        print('[Semi YoY][SOX] 涵蓋市值不足50%，樣本不足，放棄本次結果')
        return None, None

    weighted_yoy = weighted_yoy_sum / matched_cap

    # 用最常見的資料截止日期當作顯示用的期別標籤(各公司財報季度不一定對齊)
    if latest_ends:
        most_common_end = pd.Series(latest_ends).mode().iloc[0]
        period_label = most_common_end
    else:
        period_label = None

    print(f'[Semi YoY][SOX] 市值加權YoY結果：{weighted_yoy:.2f}%，期別標籤：{period_label}')
    return weighted_yoy, period_label


def fetch_ndc_business_indicators():
    """
    國發會《景氣指標及燈號》資料集(data.gov.tw dataset id 6099)是一份逐月時間序列，
    除了「景氣對策信號」燈號本身，裡面還有「海關出口值」「外銷訂單動向指數」等原始數值，
    可以直接拿同一份資料自己算出「出口年增率」「外銷訂單年增率」，不用再抓新的資料源。
    這整份都是月資料，國發會每月底才公布上個月數字，不是即時的。
    """
    result = {}

    try:
        meta_resp = requests.get(
            'https://data.gov.tw/api/v2/rest/dataset/6099',
            timeout=20
        )
        meta_resp.raise_for_status()
        meta = meta_resp.json()

        distribution = (
            (meta.get('result') or {}).get('distribution')
            or meta.get('distribution')
            or []
        )

        download_url = None
        for item in distribution:
            url = item.get('resourceDownloadUrl') or item.get('resourceDownloadURL')
            if url:
                download_url = url
                break

        if not download_url:
            print('[國發會景氣指標] 找不到下載連結，metadata：', meta)
            return result

        print('[國發會景氣指標] 下載連結：', download_url)

        file_resp = requests.get(download_url, timeout=60)
        file_resp.raise_for_status()
        content = file_resp.content

        merged_df = None
        try:
            zf = zipfile.ZipFile(io.BytesIO(content))
            names = zf.namelist()
            print('[國發會景氣指標] zip內檔案：', names)

            for name in names:
                try:
                    with zf.open(name) as f:
                        lower_name = name.lower()
                        if lower_name.endswith('.csv'):
                            candidate_df = pd.read_csv(f, encoding='utf-8-sig')
                        elif lower_name.endswith(('.xls', '.xlsx')):
                            candidate_df = pd.read_excel(f)
                        elif lower_name.endswith('.ods'):
                            candidate_df = pd.read_excel(f, engine='odf')
                        else:
                            continue

                    if candidate_df is None or candidate_df.empty:
                        continue

                    cols = [str(c) for c in candidate_df.columns]
                    print(f'[國發會景氣指標] {name} 欄位：', cols)

                    date_key = next(
                        (c for c in cols if str(c).strip() in ('Date', '年月', '日期')),
                        None
                    )

                    # zip裡有manifest.csv這類「說明檔案結構」的清單，也有拆成多個
                    # 檔案的原始資料(領先/同時/落後指標各自一份)，只有帶日期欄位
                    # 的才是真正的時間序列資料，其他一律跳過，全部依日期合併起來。
                    if date_key is None:
                        continue

                    print(
                        f'[國發會景氣指標] {name} 日期欄位原始範例(前3筆)：',
                        candidate_df[date_key].head(3).tolist(),
                        '(最後3筆)：',
                        candidate_df[date_key].tail(3).tolist()
                    )

                    candidate_df = candidate_df.rename(columns={date_key: '_date_key_'})
                    candidate_df['_date_key_'] = candidate_df['_date_key_'].map(_normalize_period)
                    candidate_df = candidate_df.dropna(subset=['_date_key_'])

                    if candidate_df.empty:
                        print(f'[國發會景氣指標] {name} 日期標準化後全部無法辨識，跳過')
                        continue

                    print(
                        f'[國發會景氣指標] {name} 標準化後日期範圍：',
                        int(candidate_df['_date_key_'].min()), '~',
                        int(candidate_df['_date_key_'].max())
                    )

                    if merged_df is None:
                        merged_df = candidate_df
                    else:
                        merged_df = merged_df.merge(
                            candidate_df,
                            on='_date_key_',
                            how='outer',
                            suffixes=('', '_dup')
                        )

                except Exception as inner_error:
                    print(f'[國發會景氣指標] 讀取{name}失敗：', repr(inner_error))
                    continue
        except zipfile.BadZipFile:
            merged_df = pd.read_csv(io.BytesIO(content), encoding='utf-8-sig')
            date_key = next(
                (c for c in merged_df.columns if str(c).strip() in ('Date', '年月', '日期')),
                merged_df.columns[0]
            )
            merged_df = merged_df.rename(columns={date_key: '_date_key_'})
            merged_df['_date_key_'] = merged_df['_date_key_'].map(_normalize_period)
            merged_df = merged_df.dropna(subset=['_date_key_'])

        if merged_df is None or merged_df.empty:
            print('[國發會景氣指標] 解壓/讀取後找不到可用的時間序列資料')
            return result

        merged_df = merged_df.sort_values('_date_key_').reset_index(drop=True)
        target_df = merged_df
        date_col = '_date_key_'

        print('[國發會景氣指標] 合併後欄位：', list(target_df.columns))

        # ---- 景氣對策信號 ----
        signal_col = next(
            (c for c in target_df.columns if '景氣對策信號' in c and '分數' not in c),
            None
        )
        score_col = next(
            (c for c in target_df.columns if '景氣對策信號' in c and '分數' in c),
            None
        )

        if signal_col is not None:
            valid = target_df.dropna(subset=[signal_col])
            if not valid.empty:
                last_row = valid.iloc[-1]
                result['business_cycle_signal'] = str(last_row[signal_col]).strip()
                result['business_cycle_period'] = str(int(last_row[date_col]))
                if score_col:
                    try:
                        result['business_cycle_score'] = float(last_row[score_col])
                    except (ValueError, TypeError):
                        pass

        # ---- 出口年增率 / 外銷訂單年增率：同一份時間序列自己算YoY(本月 vs 12個月前) ----
        export_col = next(
            (c for c in target_df.columns if '海關出口值' in c),
            None
        )
        order_col = next(
            (c for c in target_df.columns if '外銷訂單動向指數' in c),
            None
        )

        def calc_yoy(col_name):
            series = pd.to_numeric(target_df[col_name], errors='coerce').dropna()
            if len(series) < 13:
                return None
            latest_value = series.iloc[-1]
            year_ago_value = series.iloc[-13]
            if year_ago_value == 0:
                return None
            return (latest_value / year_ago_value - 1) * 100

        if export_col:
            result['export_yoy'] = calc_yoy(export_col)
            print('[國發會景氣指標] 出口年增率：', result.get('export_yoy'))

        if order_col:
            result['order_yoy'] = calc_yoy(order_col)
            print('[國發會景氣指標] 外銷訂單年增率：', result.get('order_yoy'))

        return result

    except Exception as error:
        print('[國發會景氣指標] 抓取失敗：', repr(error))
        return result


def fetch_market_overview(history):
    """
    大盤總覽：加權指數+漲跌幅、大盤本益比(簡單平均，非市值加權，僅供參考)、
    大盤波動率(TAIEX近20日年化歷史波動率)、大盤融資維持率。
    任何一項抓不到都顯示 N/A，不會讓整支程式失敗。
    """
    result = {
        'taiex_price': None,
        'taiex_change_pct': None,
        'taiex_date': None,
        'market_pe': None,
        'market_pe_date': None,
        'market_pb': None,
        'market_pb_date': None,
        'market_yield': None,
        'market_vol': None,
        'market_vol_date': None,
        'margin_ratio': None,
        'margin_ratio_date': None,
        'margin_ratio_otc': None,
        'margin_ratio_otc_date': None,
        'revenue_yoy': None,
        'revenue_period': None,
        'semi_yoy': None,
        'semi_yoy_month': None,
        'semi_yoy_update_time': None,
        'foreign_net_sell': None,
        'foreign_net_sell_date': None,
        'foreign_futures_net_oi': None,
        'foreign_futures_net_oi_date': None,
        'otc_price': None,
        'otc_change_pct': None,
        'otc_date': None,
        'otc_market_pe': None,
        'otc_market_pe_date': None,
        'otc_market_vol': None,
        'otc_market_vol_date': None,
        'taiex_drawdown': None,
        'otc_drawdown': None
    }

    # ---- 加權指數 ----
    try:
        live_price, live_prev_close = fetch_taiex_realtime()
        if live_price is None:
            info = yf.Ticker('^TWII').fast_info
            live_price = float(info['last_price'])
            live_prev_close = float(info['previous_close'])

        result['taiex_price'] = live_price
        result['taiex_date'] = datetime.now(TZ).strftime('%Y%m%d')
        if live_prev_close:
            result['taiex_change_pct'] = live_price / live_prev_close - 1
    except Exception as error:
        print('加權指數抓取失敗：', repr(error))

    # ---- 大盤波動率（20日年化歷史波動率，簡單報酬率pct_change，跟回測邏輯一致）----
    try:
        twii_hist = yf.download(
            '^TWII', period='3mo', interval='1d',
            auto_adjust=True, progress=False,
            threads=False, timeout=30
        )
        if isinstance(twii_hist.columns, pd.MultiIndex):
            twii_hist.columns = twii_hist.columns.get_level_values(0)

        twii_ret = twii_hist['Close'].pct_change().dropna()

        result['market_vol'] = float(
            twii_ret.tail(20).std() * np.sqrt(252) * 100
        )
        result['market_vol_date'] = twii_hist.index[-1].strftime('%Y%m%d')
    except Exception as error:
        print('大盤波動率計算失敗：', repr(error))

    # ---- 加權指數回撤（近一年高點）----
    try:
        twii_1y = yf.download(
            '^TWII', period='1y', interval='1d',
            auto_adjust=True, progress=False,
            threads=False, timeout=30
        )
        if isinstance(twii_1y.columns, pd.MultiIndex):
            twii_1y.columns = twii_1y.columns.get_level_values(0)

        taiex_high_1y = float(twii_1y['Close'].tail(252).max())
        if result.get('taiex_price') and taiex_high_1y:
            result['taiex_drawdown'] = result['taiex_price'] / taiex_high_1y - 1
            print(f'[加權回撤] 近一年高點={taiex_high_1y}，回撤={result["taiex_drawdown"]}')
    except Exception as error:
        print('加權回撤計算失敗：', repr(error))

    # ---- 櫃買指數（上櫃即時價格，改用TPEx官方 tpex_index，不再依賴yfinance的^TWOII）----
    otc_index_rows = None
    try:
        otc_index_resp = requests.get(
            'https://www.tpex.org.tw/openapi/v1/tpex_index',
            headers={
                'User-Agent': (
                    'Mozilla/5.0 (Linux; Android 13) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/126.0 Mobile Safari/537.36'
                )
            },
            timeout=30
        )
        otc_index_resp.raise_for_status()
        otc_index_rows = otc_index_resp.json()

        print('[櫃買指數] tpex_index 資料筆數：', len(otc_index_rows) if otc_index_rows else 0)
        if otc_index_rows:
            print('[櫃買指數] 最後一筆範例：', otc_index_rows[-1])

        if otc_index_rows:
            latest_row = otc_index_rows[-1]
            result['otc_price'] = float(str(latest_row.get('Close', '')).replace(',', ''))
            result['otc_date'] = datetime.now(TZ).strftime('%Y%m%d')

            try:
                change_value = float(str(latest_row.get('Change', '')).replace(',', ''))
                prev_close = result['otc_price'] - change_value
                if prev_close:
                    result['otc_change_pct'] = change_value / prev_close
                else:
                    raise ValueError('prev_close為0')
            except (ValueError, TypeError, ZeroDivisionError):
                if len(otc_index_rows) >= 2:
                    prev_close = float(str(otc_index_rows[-2].get('Close', '')).replace(',', ''))
                    if prev_close:
                        result['otc_change_pct'] = result['otc_price'] / prev_close - 1

            # 抓到新資料就存進history快取，供非交易日(空清單)時沿用
            history['櫃買指數快取'] = {
                'otc_price': result['otc_price'],
                'otc_change_pct': result.get('otc_change_pct'),
                'otc_date': result['otc_date']
            }
        else:
            # 非交易日等狀況，tpex_index會回傳空清單，沿用上一次抓到的快取值
            cached = history.get('櫃買指數快取')
            if cached:
                result['otc_price'] = cached.get('otc_price')
                result['otc_change_pct'] = cached.get('otc_change_pct')
                result['otc_date'] = cached.get('otc_date')
                print('[櫃買指數] 今日無資料(非交易日？)，沿用快取：', cached)
    except Exception as error:
        print('櫃買指數抓取失敗：', repr(error))
        cached = history.get('櫃買指數快取')
        if cached:
            result['otc_price'] = cached.get('otc_price')
            result['otc_change_pct'] = cached.get('otc_change_pct')
            result['otc_date'] = cached.get('otc_date')
            print('[櫃買指數] 抓取失敗，沿用快取：', cached)

    # ---- 上櫃波動率（20日年化歷史波動率，改用TPEx官方 tpex_index 歷史收盤價）----
    try:
        if otc_index_rows:
            otc_closes = []
            for row in otc_index_rows:
                try:
                    otc_closes.append(float(str(row.get('Close', '')).replace(',', '')))
                except (ValueError, TypeError):
                    continue

            otc_close_series = pd.Series(otc_closes)
            otc_ret = otc_close_series.pct_change().dropna()

            result['otc_market_vol'] = float(
                otc_ret.tail(20).std() * np.sqrt(252) * 100
            )
            result['otc_market_vol_date'] = result['otc_date']

            history['櫃買波動率快取'] = {
                'otc_market_vol': result['otc_market_vol'],
                'otc_market_vol_date': result['otc_market_vol_date']
            }
        else:
            cached_vol = history.get('櫃買波動率快取')
            if cached_vol:
                result['otc_market_vol'] = cached_vol.get('otc_market_vol')
                result['otc_market_vol_date'] = cached_vol.get('otc_market_vol_date')
                print('[上櫃波動率] 今日無資料，沿用快取：', cached_vol)
    except Exception as error:
        print('上櫃波動率計算失敗：', repr(error))

    # ---- 櫃買指數回撤（近一年高點；優先用yfinance拉一年資料，失敗則退回tpex_index現有天數）----
    try:
        otc_high_1y = None
        try:
            twoii_1y = yf.download(
                '^TWOII', period='1y', interval='1d',
                auto_adjust=True, progress=False,
                threads=False, timeout=30
            )
            if isinstance(twoii_1y.columns, pd.MultiIndex):
                twoii_1y.columns = twoii_1y.columns.get_level_values(0)
            if len(twoii_1y) > 0:
                otc_high_1y = float(twoii_1y['Close'].tail(252).max())
        except Exception as error:
            print('[櫃買回撤] yfinance抓一年資料失敗，改用tpex_index現有天數：', repr(error))

        if otc_high_1y is None and otc_index_rows:
            # 備援：tpex_index目前只回傳約一個月資料，並非真正的近一年高點，先用現有範圍內的最高值頂著
            fallback_closes = []
            for row in otc_index_rows:
                try:
                    fallback_closes.append(float(str(row.get('Close', '')).replace(',', '')))
                except (ValueError, TypeError):
                    continue
            if fallback_closes:
                otc_high_1y = max(fallback_closes)
                print('[櫃買回撤] 備援資料僅涵蓋', len(fallback_closes), '筆，非真正近一年高點，僅供暫時參考')

        if otc_high_1y and result.get('otc_price'):
            result['otc_drawdown'] = result['otc_price'] / otc_high_1y - 1
            print(f'[櫃買回撤] 高點={otc_high_1y}，回撤={result["otc_drawdown"]}')
    except Exception as error:
        print('櫃買回撤計算失敗：', repr(error))

    # ---- 大盤本益比／股價淨值比／殖利率（改用市值加權：已發行股數 × 收盤價 當權重，接近官方算法）----
    try:
        pe_resp = requests.get(
            'https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL',
            timeout=20
        )
        pe_resp.raise_for_status()
        pe_rows = pe_resp.json()

        print('[大盤估值指標] openapi BWIBBU_ALL 資料筆數：', len(pe_rows))

        pe_by_code = {}
        pb_by_code = {}
        yield_by_code = {}

        for row in pe_rows:
            code = str(row.get('Code', '')).strip()
            try:
                pe = float(str(row.get('PEratio', '')).replace(',', ''))
                # 排除異常極端值（例如剛轉盈、獲利極低的公司本益比會飆到數百倍），
                # 避免少數暴衝股票把加權平均值拉爆。100倍以上視為異常濾除。
                if 0 < pe <= 100:
                    pe_by_code[code] = pe
            except (ValueError, TypeError):
                pass
            try:
                pb = float(str(row.get('PBratio', '')).replace(',', ''))
                # 正常大盤股淨比大約落在1~5倍，50倍門檻太寬鬆，
                # 收緊到10倍，避免少數異常個股(例如淨值接近0的公司)把平均值拉爆。
                if 0 < pb <= 10:
                    pb_by_code[code] = pb
            except (ValueError, TypeError):
                pass
            try:
                dividend_yield = float(str(row.get('DividendYield', '')).replace(',', ''))
                if dividend_yield >= 0:
                    yield_by_code[code] = dividend_yield
            except (ValueError, TypeError):
                pass

        if pe_by_code or pb_by_code or yield_by_code:
            price_resp = requests.get(
                'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL',
                timeout=30
            )
            price_resp.raise_for_status()
            price_rows = price_resp.json()

            print('[大盤估值指標] openapi STOCK_DAY_ALL 資料筆數：', len(price_rows))

            valuation_date = price_rows[0].get('Date') if price_rows else None

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

            print('[大盤估值指標] openapi t187ap03_L(已發行股數) 資料筆數：', len(shares_rows))
            print('[大盤估值指標] pe_by_code 樣本數：', len(pe_by_code))
            print('[大盤估值指標] pb_by_code 樣本數：', len(pb_by_code))
            print('[大盤估值指標] close_by_code 樣本數：', len(close_by_code))

            pe_total_market_cap = pe_total_earnings = 0.0
            pb_total_market_cap = pb_total_book_value = 0.0
            yield_weighted_sum = yield_weight = 0.0
            matched = 0

            for row in shares_rows:
                try:
                    code = str(row.get('公司代號', '')).strip()
                    if code not in close_by_code:
                        continue

                    # 這支API的已發行股數單位是「仟股」，要乘1000才是實際股數。
                    # 這個單位誤差在市值加權「比率」算法裡分子分母會同倍縮放而互相抵消，
                    # 不影響最終PE/PB結果，但市值本身的絕對值要修正才正確。
                    shares = float(
                        str(row.get('已發行普通股數或TDR原股發行股數', ''))
                        .replace(',', '')
                    ) * 1000
                    if shares <= 0:
                        continue

                    matched += 1
                    market_cap = shares * close_by_code[code]

                    if code in pe_by_code:
                        # 本益比改用「總市值 ÷ 總獲利」算法，跟股淨比同一套邏輯，
                        # 兩個指標才一致。獲利用 市值/PE 反推，不用額外抓財報獲利數字。
                        earnings = market_cap / pe_by_code[code]
                        pe_total_market_cap += market_cap
                        pe_total_earnings += earnings
                    if code in pb_by_code:
                        # 股淨比改用「總市值 ÷ 總淨值」算法，比市值加權平均更接近
                        # 官方對「整體股淨比」的定義。淨值用 市值/PB 反推，
                        # 不用額外抓每股淨值(BVPS)資料。
                        book_value = market_cap / pb_by_code[code]
                        pb_total_market_cap += market_cap
                        pb_total_book_value += book_value
                    if code in yield_by_code:
                        yield_weighted_sum += yield_by_code[code] * market_cap
                        yield_weight += market_cap
                except (ValueError, TypeError, ZeroDivisionError):
                    continue

            print('[大盤估值指標] shares與收盤價配對成功家數：', matched)
            print('[大盤估值指標] pe_total_market_cap：', pe_total_market_cap, ' pe_total_earnings：', pe_total_earnings)
            print('[大盤估值指標] pb_total_market_cap：', pb_total_market_cap, ' pb_total_book_value：', pb_total_book_value)

            if pe_total_earnings > 0:
                result['market_pe'] = pe_total_market_cap / pe_total_earnings
                result['market_pe_date'] = valuation_date
                print('[大盤估值指標] 本益比(總市值/總獲利)結果：', result['market_pe'])
            if pb_total_book_value > 0:
                result['market_pb'] = pb_total_market_cap / pb_total_book_value
                result['market_pb_date'] = valuation_date
                print('[大盤估值指標] 股價淨值比(總市值/總淨值)結果：', result['market_pb'])
                print(
                    '[大盤估值指標] 股淨比樣本數：', len(pb_by_code),
                    '最大值：', max(pb_by_code.values()) if pb_by_code else None
                )
            if yield_weight > 0:
                result['market_yield'] = yield_weighted_sum / yield_weight
                print('[大盤估值指標] 殖利率市值加權結果：', result['market_yield'])
    except Exception as error:
        print('大盤估值指標抓取失敗：', repr(error))

    # ---- 上櫃本益比（市值加權：tpex_mainboard_peratio_analysis + tpex_mainboard_daily_close_quotes）----
    def fetch_json_with_retry(url, headers, timeout=30, retries=3):
        """
        TPEx部分端點資料量較大，偶爾會在傳輸中途斷線
        (ChunkedEncodingError/ProtocolError)，失敗時重試幾次再放棄。
        """
        last_error = None
        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as error:
                last_error = error
                print(f'[重試] {url} 第{attempt}次失敗：', repr(error))
        raise last_error

    try:
        otc_headers = {
            'User-Agent': (
                'Mozilla/5.0 (Linux; Android 13) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/126.0 Mobile Safari/537.36'
            )
        }

        otc_pe_rows = fetch_json_with_retry(
            'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_peratio_analysis',
            otc_headers
        )

        print('[上櫃本益比] openapi peratio_analysis 資料筆數：', len(otc_pe_rows))

        otc_pe_by_code = {}
        for row in otc_pe_rows:
            try:
                code = str(row.get('SecuritiesCompanyCode', '')).strip()
                pe = float(str(row.get('PriceEarningRatio', '')).replace(',', ''))
                # 排除異常極端值，跟上市本益比同樣用100倍當門檻濾除
                if 0 < pe <= 100:
                    otc_pe_by_code[code] = pe
            except (ValueError, TypeError):
                continue

        if otc_pe_by_code:
            otc_close_rows = fetch_json_with_retry(
                'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_daily_close_quotes',
                otc_headers
            )

            otc_valuation_date = otc_close_rows[0].get('Date') if otc_close_rows else None

            print('[上櫃本益比] openapi daily_close_quotes 資料筆數：', len(otc_close_rows))
            print('[上櫃本益比] otc_pe_by_code 樣本數：', len(otc_pe_by_code))

            otc_pe_total_market_cap = otc_pe_total_earnings = 0.0
            otc_matched = 0

            for row in otc_close_rows:
                try:
                    code = str(row.get('SecuritiesCompanyCode', '')).strip()
                    if code not in otc_pe_by_code:
                        continue

                    close = float(str(row.get('Close', '')).replace(',', ''))
                    # Capitals(發行股數)單位未確認，先假設是實際股數；
                    # 如果算出來的本益比數字明顯離譜，代表可能要*1000，屆時再校正。
                    shares = float(str(row.get('Capitals', '')).replace(',', ''))
                    if close <= 0 or shares <= 0:
                        continue

                    otc_matched += 1
                    market_cap = shares * close
                    earnings = market_cap / otc_pe_by_code[code]
                    otc_pe_total_market_cap += market_cap
                    otc_pe_total_earnings += earnings
                except (ValueError, TypeError, ZeroDivisionError):
                    continue

            print('[上櫃本益比] close與PE配對成功家數：', otc_matched)
            print(
                '[上櫃本益比] otc_pe_total_market_cap：', otc_pe_total_market_cap,
                ' otc_pe_total_earnings：', otc_pe_total_earnings
            )

            if otc_pe_total_earnings > 0:
                result['otc_market_pe'] = otc_pe_total_market_cap / otc_pe_total_earnings
                result['otc_market_pe_date'] = otc_valuation_date
                print('[上櫃本益比] 本益比(總市值/總獲利)結果：', result['otc_market_pe'])
    except Exception as error:
        print('上櫃本益比抓取失敗：', repr(error))

    # ---- 大盤本益比歷史統計：改用 market_pe_history.csv（全部歷史，不再只留5~6年）----
    pe_date_raw = result.get('market_pe_date')
    pe_date_key = None
    if pe_date_raw:
        digits = str(pe_date_raw).strip()
        if len(digits) == 7:
            roc_year = int(digits[:3])
            pe_date_key = f'{roc_year + 1911}{digits[3:]}'
        else:
            pe_date_key = digits

    if pe_date_key and result['market_pe'] is not None:
        pe_history_df = append_or_update_market_pe(pe_date_key, result['market_pe'])
    else:
        pe_history_df = load_market_pe_csv()

    pe_mean, pe_std, pe_zscore, pe_percentile, pe_sample = compute_market_pe_stats(
        pe_history_df, result['market_pe']
    )
    result['market_pe_mean'] = pe_mean
    result['market_pe_std'] = pe_std
    result['market_pe_zscore'] = pe_zscore
    result['market_pe_percentile'] = pe_percentile
    result['market_pe_sample'] = pe_sample

    print(
        f"[大盤本益比] PE：{result['market_pe']}  "
        f"平均：{pe_mean}  Std：{pe_std}  "
        f"Z-score：{pe_zscore}  Percentile：{pe_percentile}  樣本數：{pe_sample}"
    )

    otc_pe_mean, otc_pe_std, otc_pe_sample = update_market_metric_history(
        history, '上櫃本益比', result['otc_market_pe']
    )
    result['otc_market_pe_mean'] = otc_pe_mean
    result['otc_market_pe_std'] = otc_pe_std
    result['otc_market_pe_sample'] = otc_pe_sample
    print(f'[上櫃本益比] 近5年統計：平均={otc_pe_mean}, 標準差={otc_pe_std}, 樣本數={otc_pe_sample}')

    pb_mean, pb_std, pb_sample = update_market_pb_history(history, result.get('market_pb'))
    result['market_pb_mean'] = pb_mean
    result['market_pb_std'] = pb_std
    result['market_pb_sample'] = pb_sample
    print(f'[大盤股淨比] 近5年統計：平均={pb_mean}, 標準差={pb_std}, 樣本數={pb_sample}')

    # ---- 大盤融資維持率(上市) ----
    try:
        margin_ratio, margin_ratio_date = fetch_market_margin_ratio()
        result['margin_ratio'] = margin_ratio
        result['margin_ratio_date'] = margin_ratio_date
    except Exception as error:
        print('上市融資維持率整體流程失敗：', repr(error))

    # ---- 上櫃融資維持率：TPEx未公開市場加總的融資金額資料，無法計算，固定顯示「資料源缺失」----

    # ---- 上市公司當月營收年增率 ----
    try:
        yoy, period = fetch_market_revenue_yoy()
        result['revenue_yoy'] = yoy
        result['revenue_period'] = period
    except Exception as error:
        print('上市公司YoY整體流程失敗：', repr(error))

    # ---- 全球半導體營收YoY(SIA，失敗時沿用history.json裡上一次成功的數值) ----
    try:
        semi_yoy, semi_period = fetch_semi_yoy()
        if semi_yoy is not None:
            update_time = datetime.now(TZ).strftime('%Y-%m-%d %H:%M')
            result['semi_yoy'] = semi_yoy
            result['semi_yoy_month'] = semi_period
            result['semi_yoy_update_time'] = update_time
            history['semi_yoy_cache'] = {
                'semi_yoy': semi_yoy,
                'semi_yoy_month': semi_period,
                'semi_yoy_update_time': update_time
            }
        else:
            cached = history.get('semi_yoy_cache')
            if cached:
                result['semi_yoy'] = cached.get('semi_yoy')
                result['semi_yoy_month'] = cached.get('semi_yoy_month')
                result['semi_yoy_update_time'] = cached.get('semi_yoy_update_time')
                print('[Semi YoY] 本次抓取失敗，沿用history.json快取：', cached)
    except Exception as error:
        print('全球半導體YoY整體流程失敗：', repr(error))
        cached = history.get('semi_yoy_cache')
        if cached:
            result['semi_yoy'] = cached.get('semi_yoy')
            result['semi_yoy_month'] = cached.get('semi_yoy_month')
            result['semi_yoy_update_time'] = cached.get('semi_yoy_update_time')
            print('[Semi YoY] 例外狀況，沿用history.json快取：', cached)

    # ---- 外資賣超 ----
    try:
        net_value, net_date = fetch_foreign_net_sell()
        result['foreign_net_sell'] = net_value
        result['foreign_net_sell_date'] = net_date
    except Exception as error:
        print('外資賣超整體流程失敗：', repr(error))

    # ---- 外資期貨未平倉空單 ----
    try:
        net_oi, oi_date = fetch_foreign_futures_net_oi()
        result['foreign_futures_net_oi'] = net_oi
        result['foreign_futures_net_oi_date'] = oi_date
    except Exception as error:
        print('外資期貨空單整體流程失敗：', repr(error))

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
        'daily_low': data['Low'],
        'daily_high': data['High'],
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
            f'8折價 {add_price:.2f}\n'
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


def plot_etf(ax, name, etf_bundle, ema_period, stop_days, fig):
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

    # 停損價 = 還原日線近一年(近252個交易日)最高價 × (1 - 20%)
    high_1y = float(etf_bundle['daily_high'].tail(252).max())
    stop = high_1y * 0.8

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
            f'回撤 {drawdown:.1%}\n'
            f'20%停損價 {stop:.2f}\n'
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
        height_ratios=[2.6, 3.85, 3.85],
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

    history = load_history()

    market = fetch_market_overview(history)

    def fmt(value, suffix='', digits=2):
        if value is None:
            return 'N/A'
        return f'{value:,.{digits}f}{suffix}'

    def metric_state(kind, value, mean=None, std=None):
        if kind == 'pe':
            if mean is None or std is None or std <= 0:
                return 'yellow'  # 樣本數不足5年統計，無法判斷，先顯示中性黃燈
            if value > mean + std:
                return 'red'
            if value < mean - std:
                return 'green'
            return 'yellow'
        if kind == 'margin':
            if value < 150:
                return 'red'
            if value <= 170:
                return 'yellow'
            return 'green'
        if kind == 'vol':
            if value > 35:
                return 'red'
            if value >= 25:
                return 'yellow'
            return 'green'
        if kind == 'revenue_yoy':
            if value < 0:
                return 'red'
            if value <= 20:
                return 'yellow'
            return 'green'
        if kind == 'semi_yoy':
            if value < 0:
                return 'red'
            if value <= 20:
                return 'yellow'
            return 'green'
        if kind == 'pb':
            if mean is None or std is None or std <= 0:
                return 'yellow'  # 樣本數不足5年統計，無法判斷，先顯示中性黃燈
            if value > mean + std:
                return 'red'
            if value < mean - std:
                return 'green'
            return 'yellow'
        if kind == 'yield':
            if value > 4:
                return 'green'
            return 'red'
        if kind == 'foreign_sell':
            if value > 0:
                return 'green'
            return 'red'
        return 'yellow'

    revenue_note = ''
    raw_period = market.get('revenue_period')
    if raw_period:
        try:
            period_str = str(raw_period).strip()
            digits = re.sub(r'\D', '', period_str)
            if len(digits) == 6:
                year_4digit = int(digits[:4])
                month = int(digits[4:6])
                western_year_2digit = year_4digit % 100
            elif len(digits) == 5:
                roc_year = int(digits[:3])
                month = int(digits[3:5])
                western_year_2digit = (roc_year + 1911) % 100
            else:
                raise ValueError('無法辨識的日期格式')
            revenue_note = f"（{western_year_2digit:02d}／{month:02d}）"
        except (ValueError, IndexError):
            revenue_note = f"（{raw_period}）"

    # ---- 左欄：加權 / 漲跌幅 / 上市本益比 / 上市波動率 / 上市維持率 / 上市YoY ----
    left_x = 0.03

    taiex_date_note = format_date_suffix(market.get('taiex_date'))

    title_ax.text(
        left_x,
        0.92,
        f"加權 {fmt(market['taiex_price'], digits=2)}{taiex_date_note}",
        fontsize=26,
        fontweight='bold',
        ha='left',
        va='top',
        color=GOLD,
        alpha=0.95
    )

    if market['taiex_change_pct'] is not None:
        change_text = f"{market['taiex_change_pct']*100:+.2f}%"
    else:
        change_text = 'N/A'

    title_ax.text(
        left_x,
        0.72,
        f"漲跌幅 {change_text}{taiex_date_note}",
        fontsize=20,
        ha='left',
        va='center',
        color=TEXT_DIM,
        alpha=0.95
    )

    if market.get('taiex_drawdown') is not None:
        taiex_drawdown_text = f"{market['taiex_drawdown']*100:+.1f}%"
    else:
        taiex_drawdown_text = 'N/A'

    title_ax.text(
        left_x,
        0.63,
        f"回撤 {taiex_drawdown_text}",
        fontsize=18,
        ha='left',
        va='center',
        color=TEXT_DIM,
        alpha=0.85
    )

    pe_percentile = market.get('market_pe_percentile')
    pe_note = format_date_suffix(market.get('market_pe_date'))
    if pe_percentile is not None:
        pe_note = f"{pe_note}（{pe_percentile:.0f}%）"

    left_metric_rows = [
        ('上市本益比', market['market_pe'], '', 'pe', pe_note),
        ('上市波動率', market['market_vol'], '%', 'vol', format_date_suffix(market.get('market_vol_date'))),
        ('上市維持率', market['margin_ratio'], '%', 'margin', format_date_suffix(market.get('margin_ratio_date'))),
        ('上市YoY', market['revenue_yoy'], '%', 'revenue_yoy', revenue_note),
    ]
    left_row_ys = [0.51, 0.36, 0.21, 0.06]

    for (label, value, suffix, kind, note), y in zip(left_metric_rows, left_row_ys):
        if value is not None:
            if kind == 'pe':
                stat_mean, stat_std = market.get('market_pe_mean'), market.get('market_pe_std')
            else:
                stat_mean, stat_std = None, None

            draw_signal_light(
                fig, title_ax,
                metric_state(kind, value, mean=stat_mean, std=stat_std),
                x=left_x, y=y, r_px=12
            )

        title_ax.text(
            left_x + 0.03,
            y,
            f"{label} {fmt(value, suffix=suffix)}{note}",
            fontsize=20,
            ha='left',
            va='center',
            color=TEXT_DIM,
            alpha=0.95
        )

    # ---- 右欄：日期 / 櫃買 / 漲跌幅 / 上櫃本益比 / 上櫃波動率 / Semi YoY ----
    right_x = 0.66

    title_ax.text(
        right_x,
        0.985,
        (
            '日期：'
            f"{datetime.now(TZ).strftime('%Y/%m/%d %H:%M')}"
        ),
        fontsize=16,
        ha='left',
        va='top',
        color=TEXT_DIM,
        alpha=0.85
    )

    otc_date_note = format_date_suffix(market.get('otc_date'))

    title_ax.text(
        right_x,
        0.92,
        f"櫃買 {fmt(market.get('otc_price'), digits=2)}{otc_date_note}",
        fontsize=26,
        fontweight='bold',
        ha='left',
        va='top',
        color=GOLD,
        alpha=0.95
    )

    if market.get('otc_change_pct') is not None:
        otc_change_text = f"{market['otc_change_pct']*100:+.2f}%"
    else:
        otc_change_text = 'N/A'

    title_ax.text(
        right_x,
        0.72,
        f"漲跌幅 {otc_change_text}{otc_date_note}",
        fontsize=20,
        ha='left',
        va='center',
        color=TEXT_DIM,
        alpha=0.95
    )

    if market.get('otc_drawdown') is not None:
        otc_drawdown_text = f"{market['otc_drawdown']*100:+.1f}%"
    else:
        otc_drawdown_text = 'N/A'

    title_ax.text(
        right_x,
        0.63,
        f"回撤 {otc_drawdown_text}",
        fontsize=18,
        ha='left',
        va='center',
        color=TEXT_DIM,
        alpha=0.85
    )

    right_metric_rows = [
        ('上櫃本益比', market.get('otc_market_pe'), '', 'pe', format_date_suffix(market.get('otc_market_pe_date'))),
        ('上櫃波動率', market.get('otc_market_vol'), '%', 'vol', format_date_suffix(market.get('otc_market_vol_date'))),
    ]
    right_row_ys = [0.48, 0.33]

    for (label, value, suffix, kind, note), y in zip(right_metric_rows, right_row_ys):
        if value is not None:
            if kind == 'pe':
                stat_mean, stat_std = market.get('otc_market_pe_mean'), market.get('otc_market_pe_std')
            else:
                stat_mean, stat_std = None, None

            draw_signal_light(
                fig, title_ax,
                metric_state(kind, value, mean=stat_mean, std=stat_std),
                x=right_x, y=y, r_px=12
            )

        title_ax.text(
            right_x + 0.03,
            y,
            f"{label} {fmt(value, suffix=suffix)}{note}",
            fontsize=20,
            ha='left',
            va='center',
            color=TEXT_DIM,
            alpha=0.95
        )

    # ---- Semi YoY(全球半導體營收年增率，資料來源：SIA官方新聞) ----
    semi_yoy_value = market.get('semi_yoy')
    if semi_yoy_value is not None:
        semi_yoy_text = f"{semi_yoy_value:+.1f}%"
    else:
        semi_yoy_text = 'N/A'

    semi_yoy_month_text = market.get('semi_yoy_month') or ''

    if semi_yoy_value is not None:
        draw_signal_light(
            fig, title_ax,
            metric_state('semi_yoy', semi_yoy_value),
            x=right_x, y=0.10, r_px=12
        )

    title_ax.text(
        right_x + 0.03,
        0.10,
        f"Semi YoY {semi_yoy_text}" + (f"（{semi_yoy_month_text}）" if semi_yoy_month_text else ''),
        fontsize=20,
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
