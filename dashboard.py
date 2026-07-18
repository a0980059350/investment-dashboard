import os, re
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests, yfinance as yf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle

TZ=ZoneInfo('Asia/Taipei'); OUTPUT='wallpaper.png'
FUNDS=[
 {'name':'安聯台灣科技基金','url':'https://fund.hncb.com.tw/w/wr/wr02_ACDD04-005003.djhtm'},
 {'name':'統一全球新科技基金','url':'https://fund.hncb.com.tw/w/wr/wr02_ACPS38-009022.djhtm'}]
ETFS=[{'name':'00631L','ticker':'00631L.TW','ema':32},{'name':'00830','ticker':'00830.TW','ema':42}]

def setup_font():
    for p in ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc','/usr/share/fonts/opentype/noto/NotoSansCJKtc-Regular.otf']:
        if os.path.exists(p):
            plt.rcParams['font.family']=font_manager.FontProperties(fname=p).get_name(); break
    plt.rcParams['axes.unicode_minus']=False

def clean_table(t):
    t=t.copy(); t.columns=[str(c).strip() for c in t.columns]
    dc=next((c for c in t.columns if '日期' in c),None); vc=next((c for c in t.columns if '淨值' in c),None)
    if not dc or not vc: raise ValueError('找不到日期/淨值欄位')
    out=t[[dc,vc]].copy(); out.columns=['Date','Value']; y=datetime.now(TZ).year
    def pdte(x):
        s=str(x).strip(); s=f'{y}/{s}' if re.fullmatch(r'\d{1,2}/\d{1,2}',s) else s
        return pd.to_datetime(s,errors='coerce')
    out['Date']=out['Date'].map(pdte); out['Value']=pd.to_numeric(out['Value'].astype(str).str.replace(',','',regex=False),errors='coerce')
    return out.dropna().drop_duplicates('Date').sort_values('Date')

def fetch_fund(url):
    r=requests.get(url,headers={'User-Agent':'Mozilla/5.0'},timeout=30); r.raise_for_status()
    cs=[]
    for t in pd.read_html(r.text):
        if '日期' in ' '.join(map(str,t.columns)) and '淨值' in ' '.join(map(str,t.columns)):
            try: cs.append(clean_table(t))
            except: pass
    if not cs: raise RuntimeError('基金資料抓取失敗')
    d=pd.concat(cs).drop_duplicates('Date').sort_values('Date')
    return d[d['Date']>=pd.Timestamp.now()-pd.Timedelta(days=380)].tail(270)

def fetch_etf(ticker):
    d=yf.download(ticker,period='18mo',interval='1d',auto_adjust=True,progress=False,threads=False)
    if d.empty: raise RuntimeError(f'{ticker} 無資料')
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d=d[['Open','High','Low','Close']].dropna()
    return pd.DataFrame({'Open':d['Open'].resample('W-FRI').first(),'High':d['High'].resample('W-FRI').max(),'Low':d['Low'].resample('W-FRI').min(),'Close':d['Close'].resample('W-FRI').last()}).dropna().tail(53)

def stats(s):
    s=s.dropna(); latest=float(s.iloc[-1]); high=float(s.max()); return latest,high,latest/high-1,latest/float(s.iloc[0])-1

def plot_fund(ax,name,d):
    x=np.arange(len(d)); latest,high,dd,ret=stats(d['Value'])
    ax.plot(x,d['Value'],lw=2.5); ax.axhline(high,lw=1.4,ls='--'); ax.text(len(x)-1,high,f' 最高 {high:.2f}',va='bottom',ha='right',fontsize=12)
    ax.set_title(name,loc='left',fontsize=20,fontweight='bold',pad=12)
    ax.text(.99,.95,f'最新淨值 {latest:.2f}\n近一年報酬 {ret:+.1%}\n回撤 {dd:.1%}',transform=ax.transAxes,ha='right',va='top',fontsize=14,bbox=dict(boxstyle='round,pad=.45',alpha=.15))
    ax.grid(alpha=.2); ax.set_xlim(0,max(1,len(x)-1)); ax.tick_params(labelbottom=False); ax.spines[['top','right']].set_visible(False)

def plot_etf(ax,name,d,n):
    x=np.arange(len(d)); ema=d['Close'].ewm(span=n,adjust=False).mean(); latest,high,dd,ret=stats(d['Close']); stop=high*.8
    for i,(_,r) in enumerate(d.iterrows()):
        o,h,l,c=map(float,[r['Open'],r['High'],r['Low'],r['Close']]); ax.vlines(i,l,h,lw=1)
        ax.add_patch(Rectangle((i-.29,min(o,c)),.58,max(abs(c-o),max(c,o)*.001),fill=(c>=o),alpha=.65))
    ax.plot(x,ema.values,lw=2,label=f'EMA{n}'); ax.axhline(high,lw=1.3,ls='--'); ax.axhline(stop,lw=1.5,ls=':')
    ax.text(len(x)-1,high,f' 最高 {high:.2f}',va='bottom',ha='right',fontsize=12); ax.text(len(x)-1,stop,f' 停損價 {stop:.2f}',va='bottom',ha='right',fontsize=12)
    status=f'站上週EMA{n}' if latest>float(ema.iloc[-1]) else f'跌破週EMA{n}'
    ax.set_title(name,loc='left',fontsize=20,fontweight='bold',pad=12)
    ax.text(.99,.96,f'最新價格 {latest:.2f}\n近一年報酬 {ret:+.1%}\n回撤 {dd:.1%}\n{status}',transform=ax.transAxes,ha='right',va='top',fontsize=14,bbox=dict(boxstyle='round,pad=.45',alpha=.15))
    ax.grid(alpha=.2); ax.set_xlim(-1,len(x)); ax.tick_params(labelbottom=False); ax.spines[['top','right']].set_visible(False)

def main():
    setup_font(); fig=plt.figure(figsize=(10.8,24),dpi=100); fig.patch.set_facecolor('#f4f5f8')
    gs=fig.add_gridspec(5,1,height_ratios=[.55,2.1,2.1,2.45,2.45],hspace=.34,left=.08,right=.95,top=.97,bottom=.04)
    t=fig.add_subplot(gs[0]); t.axis('off'); t.text(0,.7,'投資儀表板',fontsize=30,fontweight='bold',ha='left',va='center'); t.text(0,.18,'科技四核心｜近一年',fontsize=15,ha='left',va='center',alpha=.7)
    axes=[fig.add_subplot(gs[i]) for i in range(1,5)]
    for ax in axes: ax.set_facecolor('white')
    for ax,c in zip(axes[:2],FUNDS):
        try: plot_fund(ax,c['name'],fetch_fund(c['url']))
        except Exception as e: ax.axis('off'); ax.text(.02,.65,c['name'],fontsize=20,fontweight='bold'); ax.text(.02,.42,f'資料更新失敗\n{type(e).__name__}: {e}',fontsize=14)
    for ax,c in zip(axes[2:],ETFS):
        try: plot_etf(ax,c['name'],fetch_etf(c['ticker']),c['ema'])
        except Exception as e: ax.axis('off'); ax.text(.02,.65,c['name'],fontsize=20,fontweight='bold'); ax.text(.02,.42,f'資料更新失敗\n{type(e).__name__}: {e}',fontsize=14)
    fig.text(.08,.015,f"更新時間：{datetime.now(TZ).strftime('%Y/%m/%d %H:%M')}",fontsize=13,alpha=.75)
    plt.savefig(OUTPUT,dpi=100,facecolor=fig.get_facecolor()); plt.close(fig); print('已產生',OUTPUT)
if __name__=='__main__': main()
