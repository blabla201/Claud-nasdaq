import io, json, os, re, time, urllib.parse, urllib.request
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None

# ===== MOTOR =====



# ---------- Göstergeler ----------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return (100 - 100 / (1 + au / ad.replace(0, np.nan))).fillna(50)

def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()

def vwap_roll(df, n=20):
    tp = (df.High + df.Low + df.Close) / 3
    return (tp * df.Volume).rolling(n).sum() / df.Volume.rolling(n).sum()

def session_vwap(intra):
    """Gün içi (5dk) veride her seansın kendi VWAP'ı."""
    tp = (intra.High + intra.Low + intra.Close) / 3
    day = intra.index.date
    return (tp * intra.Volume).groupby(day).cumsum() / intra.Volume.groupby(day).cumsum()

def prep(df):
    df = df.copy()
    c = df.Close
    df["e9"], df["e21"] = ema(c, 9), ema(c, 21)
    df["vw"], df["rsi"], df["atr"] = vwap_roll(df), rsi(c), atr(df)
    df["hh20"] = df.High.shift(1).rolling(20).max()
    df["ll10"] = df.Low.shift(1).rolling(10).min()
    return df.dropna()

# ---------- Stratejiler (sinyal: gün kapanışında 1=pozisyonda) ----------
def _latch(enter, exit_):
    out, p = [], 0
    for e, x in zip(enter.values, exit_.values):
        if p == 0 and e: p = 1
        elif p == 1 and x: p = 0
        out.append(p)
    return pd.Series(out, index=enter.index, dtype=float)

def _retest(d):
    """Kırılım sonrası seviyeye geri dönüp tutunursa gir; seviye kaybedilirse ya da EMA21 altına inerse çık."""
    c, l, lv, at, e21 = d.Close.values, d.Low.values, d.hh20.values, d.atr.values, d.e21.values
    out, st, lvl, bo = [], 0, 0.0, 0
    for i in range(len(d)):
        if st == 0 and c[i] > lv[i]:
            st, lvl, bo = 1, lv[i], i
        elif st == 1:
            if c[i] < lvl * 0.99 or i - bo > 10: st = 0
            elif i > bo and l[i] <= lvl * 1.015 and c[i] > lvl: st = 2
        elif st == 2 and (c[i] < lvl - at[i] or c[i] < e21[i]):
            st = 0
        out.append(1.0 if st == 2 else 0.0)
    return pd.Series(out, index=d.index)

NAMES = {
    "ema9": "EMA9 (fiyat > EMA9)",
    "x921": "EMA9 / EMA21 kesişimi",
    "vwap": "VWAP (20 gün) üstü",
    "combo": "Birleşik: EMA9+VWAP+RSI",
    "rsi": "RSI dönüş (35/55)",
    "brk": "20 gün kırılım",
    "retest": "Kırılım + geri test",
}
FUNCS = {
    "ema9": lambda d: (d.Close > d.e9).astype(float),
    "x921": lambda d: (d.e9 > d.e21).astype(float),
    "vwap": lambda d: (d.Close > d.vw).astype(float),
    "combo": lambda d: ((d.Close > d.e9) & (d.Close > d.vw) & (d.rsi > 50) & (d.rsi < 75)).astype(float),
    "rsi": lambda d: _latch(d.rsi < 35, d.rsi > 55),
    "brk": lambda d: _latch(d.Close > d.hh20, d.Close < d.ll10),
    "retest": _retest,
}

# ---------- Backtest ----------
def backtest(df, pos, cost=0.0005):
    """Sinyal t kapanışında oluşur, getiri t+1'den itibaren alınır (ileri bakış yok)."""
    ret = df.Close.pct_change().fillna(0)
    p = pos.shift(1).fillna(0)
    turn = p.diff().abs().fillna(p.abs())
    sret = p * ret - turn * cost
    eq, bh = (1 + sret).cumprod(), (1 + ret).cumprod()
    dd = eq / eq.cummax() - 1
    chg = pos.diff().fillna(pos)
    trades, entry = [], None
    for t, v in chg.items():
        if v == 1 and entry is None:
            entry = t
        elif v == -1 and entry is not None:
            trades.append((entry, t, df.Close[t] / df.Close[entry] - 1 - 2 * cost)); entry = None
    if entry is not None:
        trades.append((entry, df.index[-1], df.Close.iloc[-1] / df.Close[entry] - 1 - 2 * cost))
    tr = pd.DataFrame(trades, columns=["Giriş", "Çıkış", "Getiri"])
    w, l = tr[tr.Getiri > 0].Getiri, tr[tr.Getiri <= 0].Getiri
    sd = sret.std()
    tot = eq.iloc[-1] - 1
    return dict(
        eq=eq, bh=bh, dd=dd, trades=tr, tot=tot, bhr=bh.iloc[-1] - 1, mdd=dd.min(),
        n=len(tr), win=len(w) / len(tr) if len(tr) else 0,
        aw=w.mean() if len(w) else 0.0, al=l.mean() if len(l) else 0.0,
        pf=(w.sum() / -l.sum()) if len(l) and l.sum() != 0 else (9.99 if len(w) else 0.0),
        sh=(sret.mean() / sd * np.sqrt(252)) if sd else 0.0,
        rr=tot / abs(dd.min()) if dd.min() else 0.0, expo=p.mean(), sret=sret,
    )

def run_all(df, cost=0.0005):
    return {k: backtest(df, f(df), cost) for k, f in FUNCS.items()}

# ---------- Skor ve işlem planı ----------
def trendline(df, look=80, w=4):
    """Son iki alçalan tepeden geçen düşen trend çizgisi. Kırıldıysa (son 5 gün içinde yukarı) broken=True."""
    h, c, n = df.High.values, df.Close.values, len(df)
    piv = [i for i in range(max(w, n - look), n - w) if h[i] == h[i - w:i + w + 1].max()]
    for a in range(len(piv) - 1, 0, -1):
        i2, i1 = piv[a], piv[a - 1]
        if h[i2] < h[i1] and i2 - i1 >= 5:
            m = (h[i2] - h[i1]) / (i2 - i1)
            ln = lambda k: h[i1] + m * (k - i1)
            now = bool(c[n - 1] > ln(n - 1))
            was_below = any(c[k] <= ln(k) for k in range(n - 6, n - 1))
            return dict(i1=int(i1), y1=float(h[i1]), m=float(m), above=now, broken=now and was_below)
    return None

def oos(res, frac=0.6):
    """Veri dışı doğrulama: ilk %60'ta en iyi stratejiyi seç, hiç görmediği son %40'ta sına."""
    def sh(x):
        sd = x.std()
        return float(x.mean() / sd * np.sqrt(252)) if sd else 0.0
    n = len(next(iter(res.values()))["sret"]); cut = int(n * frac)
    tr = {k: sh(r["sret"].iloc[:cut]) for k, r in res.items()}
    k = max(tr, key=tr.get); r = res[k]; te = r["sret"].iloc[cut:]
    return dict(k=k, train_sh=tr[k], test_sh=sh(te), test_tot=float((1 + te).prod() - 1),
                test_bh=float(r["bh"].iloc[-1] / r["bh"].iloc[cut - 1] - 1))

def score(df, best_sharpe):
    c, last = df.Close, df.iloc[-1]
    r, add = [], lambda t, p: r.append((t, p))
    if last.Close > last.e9:
        add('Fiyat EMA9 üzerinde', 15)
    else:
        add('Fiyat EMA9 altında', -10)
    if last.e9 > last.e21:
        add('EMA9 > EMA21 (kısa trend yukarı)', 15)
    else:
        add('EMA9 < EMA21 (trend zayıf)', -10)
    if last.Close > last.vw:
        add('Fiyat 20g VWAP üzerinde', 15)
    else:
        add('Fiyat VWAP altında', -10)
    r20 = c.iloc[-1] / c.iloc[-21] - 1
    add(f"20 günlük getiri %{r20*100:+.1f}", 20 if r20 > .08 else 10 if r20 > 0 else -10)
    R = last.rsi
    if 50 <= R <= 70: add(f"RSI {R:.0f}: sağlıklı güç bölgesi", 15)
    elif R > 75: add(f"RSI {R:.0f}: aşırı alım riski", -10)
    elif R < 30: add(f"RSI {R:.0f}: aşırı satım, tepki olası", 5)
    else: add(f"RSI {R:.0f}: nötr/zayıf", 0)
    vr = df.Volume.tail(5).mean() / df.Volume.tail(20).mean()
    if vr > 1.2:
        add(f'Hacim ortalamanın {vr:.1f} katı', 10)
    else:
        add(f'Hacim normal ({vr:.1f}x)', 0)
    tl = trendline(df)
    if tl and tl["broken"]: add("Düşen trend çizgisi son 5 günde yukarı kırıldı", 10)
    if best_sharpe > 0.8:
        add(f'Veri dışı testte strateji Sharpe {best_sharpe:.1f}', 10)
    else:
        add('Veri dışı testte strateji zayıf', -5)
    return float(np.clip(50 + sum(p for _, p in r) / 1.5, 0, 100)), r

def _breakout(df, look=10):
    """Son `look` gün içindeki ilk kırılım: (konum, seviye) ya da None."""
    c, h, n = df.Close.values, df.hh20.values, len(df)
    for i in range(n - look, n):
        if c[i] > h[i]:
            return i, float(h[i])
    return None

def plan(df, sc, mkt_ok=True, maxatr=1.0):
    """Kırılımı kovalamaz: önce kırılımı bekler, sonra seviyeye geri dönüşü (geri test) bekler."""
    last = df.iloc[-1]
    c, a = float(last.Close), float(last.atr)
    atrp = a / c
    bo = _breakout(df)
    warn = []
    if bo is None:
        stage, lvl = "pre", float(last.hh20)
        entry = lvl * 1.002
        stop = entry - 1.5 * a
    else:
        i, lvl = bo
        if c < lvl * 0.99:
            stage = "fail"; entry = float(last.hh20) * 1.002; stop = entry - 1.5 * a
            warn.append("Sahte kırılım: fiyat kırılım seviyesinin altına döndü")
        else:
            stage = "retest" if c <= lvl * 1.02 else "extended"
            entry = c if stage == "retest" else lvl * 1.005
            stop = min(lvl - 1.0 * a, entry - 0.5 * a)
            if stage == "extended":
                warn.append("Kırılım gerçekleşti ama fiyat uzaklaştı: kovalamayın, seviyeye geri dönüşü (geri test) bekleyin")
            vol = float(df.Volume.iloc[i]); avg = float(df.Volume.iloc[max(0, i - 20):i].mean()) or 1.0
            if vol < avg:
                warn.append("Kırılım ortalamanın altında hacimle geldi: zayıf kırılım")
    R = entry - stop
    if last.rsi > 75: warn.append("RSI aşırı alım bölgesinde")
    if c < last.vw: warn.append("Fiyat VWAP altında")
    if c < 5: warn.append("Düşük fiyatlı hisse: spread ve oynaklık riski yüksek, küçük pozisyon kullanın")
    if R / entry > .06: warn.append(f"Stop mesafesi geniş (%{R/entry*100:.1f})")
    if atrp > maxatr: warn.append(f"Aşırı oynak: günlük ortalama hareket %{atrp*100:.1f}")
    if not mkt_ok: warn.append("Piyasa filtresi: QQQ düşüş trendinde, yeni alım önerilmez")
    blocked = atrp > maxatr or not mkt_ok
    strong = sc >= 70 and c > last.e21 and stage != "fail"
    kind = "AL" if strong and not blocked else "IZLE" if sc >= 50 else "UZAK"
    if stage == "fail":
        label = "SAHTE KIRILIM"
    elif kind == "AL":
        label = {"pre": f"KIRILIMI BEKLE (%{(lvl/c-1)*100:.1f} uzakta)",
                 "extended": f"GERİ TEST BEKLE (seviye ${lvl:.2f})",
                 "retest": "GERİ TEST BÖLGESİ: ALIM"}[stage]
    else:
        label = ("İZLE (filtre)" if strong and blocked else "İZLE") if kind == "IZLE" else "UZAK DUR"
    return dict(price=c, hh=lvl, entry=entry, stop=stop, R=R, t1=entry + 2 * R, t2=entry + 3 * R,
                kind=kind, label=label, warn=warn, done=stage in ("extended", "retest"), stage=stage, atrp=atrp, max_days=10)

def lots(capital, risk_pct, p, fractional=False):
    """Risk yüzdesine göre adet. fractional=True: parçalı (kesirli) hisse alınabilir."""
    if p["R"] <= 0:
        return 0
    q = min(capital * risk_pct / 100 / p["R"], capital / p["entry"])
    return round(q, 2) if fractional else int(q)


# ===== TARAYICI =====



HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "alerts_state.json")
LIQ = os.path.join(HERE, "liquid_cache.json")
LOGF = os.path.join(HERE, "signals_log.json")
RADARF = os.path.join(HERE, "radar_cache.json")
SURGEF = os.path.join(HERE, "surge_cache.json")
STATE2 = os.path.join(HERE, "surge_state.json")
COST = 0.0005
NDX_FALLBACK = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO COST NFLX AMD ADBE PEP CSCO TMUS INTC QCOM TXN AMGN INTU "
                "AMAT BKNG ISRG LRCX MU PANW ADI KLAC SBUX GILD MRVL CRWD ABNB ADP CDNS SNPS ORLY MELI PYPL").split()

# ---------- Evren ----------
def universe_ndx():
    try:
        t = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100", match="Ticker")[0]
        return [s.replace(".", "-") for s in t["Ticker"].astype(str)]
    except Exception:
        return NDX_FALLBACK

def universe_all():
    """Nasdaq'ın resmi sembol listesi: test/ETF/varant/hak/imtiyazlı hisseler elenir."""
    raw = urllib.request.urlopen("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", timeout=25).read().decode()
    d = pd.read_csv(io.StringIO(raw), sep="|")
    d = d[d["Symbol"].astype(str).str.match(r"^[A-Z]{1,5}$") & (d["Test Issue"] == "N") & (d["ETF"] == "N")]
    d = d[~d["Security Name"].str.contains(r"Warrant|Right|Unit|Preferred|Notes|Depositary", case=False, regex=True)]
    return d["Symbol"].tolist()

# ---------- Veri ----------
def load_daily(tickers, period="2y", chunk=100, minlen=150, progress=None):
    out, tickers = {}, list(tickers)
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, period=period, interval="1d", group_by="ticker",
                              auto_adjust=True, threads=True, progress=False)
        except Exception:
            continue
        if progress:
            progress(min(1.0, (i + chunk) / max(1, len(tickers))))
        for t in part:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) and t in raw.columns.get_level_values(0) else raw
                d = d.copy()
                if isinstance(d.columns, pd.MultiIndex):
                    d.columns = d.columns.get_level_values(0)
                d = d[["Open", "High", "Low", "Close", "Volume"]].dropna()
                if len(d) >= minlen:
                    out[t] = d
            except Exception:
                pass
    return out

def liquid_table(symbols, progress=None):
    """Tüm hisselerin son fiyatı ve 20 günlük ortalama $ hacmi (12 saat önbellek, tek indirme)."""
    key = f"tab-{len(symbols)}"
    try:
        c = json.load(open(LIQ))
        if c.get("key") == key and time.time() - c["t"] < 12 * 3600:
            return c["rows"]
    except Exception:
        pass
    rows = []
    for s, d in load_daily(symbols, "1mo", 300, 15, progress).items():
        try:
            rows.append([s, float(d.Close.iloc[-1]), float((d.Close * d.Volume).tail(20).mean())])
        except Exception:
            continue
    try:
        json.dump(dict(key=key, t=time.time(), rows=rows), open(LIQ, "w"))
    except Exception:
        pass
    return rows

def liquid(symbols, topn, minvol, pmin=5, pmax=1e9, progress=None):
    """Fiyatı pmin-pmax arasında ve ortalama $ hacmi yeterli olan en likit hisseler."""
    sel = [(s, dv) for s, p, dv in liquid_table(symbols, progress) if pmin <= p <= pmax and dv >= minvol]
    return [s for s, _ in sorted(sel, key=lambda x: -x[1])[:topn]]

# ---------- Uyarılar ----------
def sig_type(p):
    return {"pre": "kirilim", "extended": "bekle", "retest": "geri_test"}.get(p.get("stage"), "diger")

def fmt(sym, a, capital, risk, frac=False):
    p = a["plan"]
    q = lots(capital, risk, p, frac)
    w = (" ⚠ " + "; ".join(p["warn"])) if p["warn"] else ""
    return (f"🟢 {sym} — {p['label']}\nSkor {a['score']:.0f} · Fiyat ${p['price']:.2f}\n"
            f"Kırılım ${p['hh']:.2f} · Alış ${p['entry']:.2f} · Stop ${p['stop']:.2f}\n"
            f"Hedef ${p['t1']:.2f} / ${p['t2']:.2f} · Lot {q}{w}\n⏱ {p['max_days']} işlem günü içinde hedefe gitmezse çık. Stop'u aşağı çekme.\nYatırım tavsiyesi değildir.")

def detect_new(A, capital, risk, frac=False):
    cur = {f"{s}:{sig_type(a['plan'])}": s for s, a in A.items() if a["plan"]["kind"] == "AL"}
    try:
        old = set(json.load(open(STATE)))
    except Exception:
        old = set()
    try:
        json.dump(sorted(cur), open(STATE, "w"))
    except Exception:
        pass
    return [dict(sym=s, msg=fmt(s, A[s], capital, risk, frac), plan=A[s]["plan"], score=A[s]["score"],
                 date=str(A[s]["df"].index[-1].date())) for k, s in cur.items() if k not in old]

def telegram(token, chat, text):
    try:
        data = urllib.parse.urlencode({"chat_id": chat, "text": text}).encode()
        urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data, timeout=15)
        return True
    except Exception:
        return False

# ---------- Piyasa filtresi, bilanço, sinyal günlüğü ----------
def market_status():
    """QQQ 50 günlük ortalamanın üstündeyse piyasa yükselişte sayılır."""
    try:
        c = load_daily(["QQQ"], "1y", 100, 100)["QQQ"].Close
        ok = bool(c.iloc[-1] > c.rolling(50).mean().iloc[-1])
        return ok, f"QQQ {c.iloc[-1]:.2f}, 50 günlük ortalamanın {'üstünde (yükseliş)' if ok else 'altında (düşüş)'}"
    except Exception:
        return True, "QQQ verisi alınamadı, piyasa filtresi uygulanmadı"

def earnings_days(sym):
    """Bir sonraki bilançoya kaç gün var (bilinmiyorsa None)."""
    try:
        cal = yf.Ticker(sym).calendar
        ed = cal.get("Earnings Date") if isinstance(cal, dict) else None
        if ed:
            d = ed[0] if isinstance(ed, (list, tuple)) else ed
            return (pd.Timestamp(d).normalize() - pd.Timestamp.today().normalize()).days
    except Exception:
        pass
    return None

CLOSED = ("hedefe ulaştı", "stopa değdi", "süre doldu", "girilmedi")

def load_log():
    try:
        return json.load(open(LOGF, encoding="utf-8"))
    except Exception:
        return []

def save_log(L):
    try:
        json.dump(L, open(LOGF, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    except Exception:
        pass

def log_add(rec):
    L = load_log(); L.append(rec); save_log(L)

def merge_log(data):
    L = load_log()
    keys = {(r["sym"], r["tarih"], r["kaynak"], r["entry"]) for r in L}
    for r in data:
        if (r.get("sym"), r.get("tarih"), r.get("kaynak"), r.get("entry")) not in keys:
            L.append(r)
    save_log(L)

def log_new(new):
    L = load_log()
    for n in new:
        p = n["plan"]
        if any(r["sym"] == n["sym"] and r["tarih"] == n["date"] and r["kaynak"] == "program" for r in L):
            continue
        L.append(dict(sym=n["sym"], tarih=n["date"], kaynak="program", etiket=p["label"], skor=round(n["score"]),
                      entry=round(float(p["entry"]), 2), stop=round(float(p["stop"]), 2), t1=round(float(p["t1"]), 2),
                      max_days=p["max_days"], durum="bekliyor", giris_tarihi=None, cikis_tarihi=None, sonuc=None))
    save_log(L)

def evaluate(rec, df):
    """Kayıttan sonraki günlük barlara bakıp sonucu belirler (aynı gün stop ve hedef varsa stop sayılır)."""
    d = df[df.index > pd.Timestamp(rec["tarih"])]
    state = "açık" if rec["kaynak"] == "elle" else "bekliyor"
    gir = rec["tarih"] if state == "açık" else None
    cik = sonuc = None
    wait = opn = 0
    for t, r in d.iterrows():
        day = str(t.date())
        if state == "bekliyor":
            wait += 1
            if r.High >= rec["entry"]:
                state, gir = "açık", day
            elif wait >= 5:
                state, cik = "girilmedi", day
                break
            else:
                continue
        opn += 1
        if r.Low <= rec["stop"]:
            state, sonuc, cik = "stopa değdi", rec["stop"] / rec["entry"] - 1, day; break
        if r.High >= rec["t1"]:
            state, sonuc, cik = "hedefe ulaştı", rec["t1"] / rec["entry"] - 1, day; break
        if opn >= rec["max_days"]:
            state, sonuc, cik = "süre doldu", r.Close / rec["entry"] - 1, day; break
    if sonuc is not None:
        sonuc -= 2 * COST
    rec.update(durum=state, giris_tarihi=gir, cikis_tarihi=cik, sonuc=sonuc)

def update_log(A):
    L = load_log()
    need = [r for r in L if r["durum"] not in CLOSED]
    miss = sorted({r["sym"] for r in need if r["sym"] not in A})
    extra = load_daily(miss, "6mo", 100, 20) if miss else {}
    for r in need:
        df = A[r["sym"]]["df"] if r["sym"] in A else extra.get(r["sym"])
        if df is not None:
            try:
                evaluate(r, df)
            except Exception:
                pass
    save_log(L)
    return L

# ---------- Ani hareket radarı ve hisse inceleme ----------
def _tags(price, ch, vx, dollar):
    t = []
    if price < 1: t.append("1$ altı (çıkarılma riski)")
    elif price < 3: t.append("3$ altı")
    if ch > 1.0: t.append("tek günde %100+")
    if vx > 20: t.append("hacim 20x+")
    if dollar < 5e6: t.append("düşük likidite")
    return t

def radar(symbols, min_move=0.3, min_volx=5.0, min_dollar=1e6, progress=None):
    """Bugün fiyatı ve hacmi birlikte sıçrayan hisseler. Alım listesi değil, izleme listesidir."""
    rows = []
    for s, d in load_daily(symbols, "2mo", 300, 25, progress).items():
        try:
            c, v = d.Close, d.Volume
            ch = float(c.iloc[-1] / c.iloc[-2] - 1)
            vx = float(v.iloc[-1] / max(v.iloc[-21:-1].mean(), 1))
            dv = float(c.iloc[-1] * v.iloc[-1])
            if ch >= min_move and vx >= min_volx and dv >= min_dollar and c.iloc[-1] >= 0.3:
                rows.append(dict(Sembol=s, Fiyat=float(c.iloc[-1]), Bugün=ch, Gün5=float(c.iloc[-1] / c.iloc[-6] - 1),
                                 Hacimx=vx, GünlükDolar=dv, Tarih=str(d.index[-1].date()), Uyarı=", ".join(_tags(float(c.iloc[-1]), ch, vx, dv))))
        except Exception:
            continue
    return sorted(rows, key=lambda r: -r["Bugün"])

def run_radar(min_move, min_volx, progress=None):
    key = f"{min_move}-{min_volx}"
    try:
        c = json.load(open(RADARF, encoding="utf-8"))
        if c["key"] == key and time.time() - c["t"] < 15 * 60:
            return c["rows"]
    except Exception:
        pass
    rows = radar(universe_all(), min_move, min_volx, 1e6, progress)
    try:
        json.dump(dict(key=key, t=time.time(), rows=rows), open(RADARF, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass
    return rows

def inspect_symbol(sym):
    """Tek bir hisseyi incele: fiyat hareketi, risk işaretleri ve programın görüşü."""
    raw = load_daily([sym], "1y", 100, 30).get(sym)
    if raw is None:
        return None
    c, v = raw.Close, raw.Volume
    X = dict(sym=sym, raw=raw, price=float(c.iloc[-1]), ch1=float(c.iloc[-1] / c.iloc[-2] - 1),
             ch5=float(c.iloc[-1] / c.iloc[-6] - 1) if len(c) > 6 else 0.0,
             ch20=float(c.iloc[-1] / c.iloc[-21] - 1) if len(c) > 21 else 0.0,
             vx=float(v.iloc[-1] / max(v.iloc[-21:-1].mean(), 1)), hi=float(raw.High.tail(252).max()), lo=float(raw.Low.tail(252).min()),
             from_peak=float(c.iloc[-1] / c.tail(60).max() - 1), dollar=float(c.iloc[-1] * v.iloc[-1]))
    fl = []
    p = X["price"]
    if p < 1: fl.append("1$ altında: Nasdaq’tan çıkarılma (delisting) riski ve çok geniş alış-satış farkı var.")
    elif p < 3: fl.append("3$ altında: program normalde bu hisseleri önermez.")
    big = float((c.pct_change().tail(30)).max())
    if big > 1.0: fl.append(f"Son 30 günde tek günde %{big*100:.0f} sıçrama olmuş: spekülatif hareket, genelde sonrasında sert geri çekilme gelir.")
    if X["vx"] > 10: fl.append(f"Bugünkü hacim ortalamanın {X['vx']:.0f} katı.")
    if X["from_peak"] < -0.4: fl.append(f"Son 60 günün zirvesinden %{abs(X['from_peak'])*100:.0f} aşağıda: yükselişin büyük kısmı geri verilmiş.")
    if X["dollar"] < 5e5: fl.append("Günlük işlem hacmi düşük: istediğiniz fiyattan çıkamayabilirsiniz.")
    if X["ch1"] > 0.3: fl.append("Bugün çok sert yükselmiş: kâr çoktan oluşmuş olabilir, sonradan girenler çoğu zaman tepeden alır.")
    X["flags"] = fl
    try:
        df = prep(raw)
        if len(df) >= 60:
            bt = run_all(df, COST)
            o = oos(bt)
            sc, why = score(df, o["test_sh"])
            X.update(df=df, score=sc, why=why, plan=plan(df, sc, True, 1.0))
    except Exception:
        pass
    return X

# ---------- Ani sıçrama: kısa vade, geçmiş istatistikle karar ----------
SURGE = dict(minmove=.08, maxmove=.6, minvx=3.0, pos=.75)
S_SLIP, S_COST, S_GAP = .005, .01, .03
CFG_DAYS = (1, 2, 3, 5)
CFG_STOPS = (.04, .06, .08)
EMPTY = dict(n=0, win=0.0, exp=0.0, med=0.0, p10=0.0, train_exp=0.0, test_exp=0.0, test_n=0, days=1, stop=.06, tgt=.12, skipped=0, ok=False)

def surge_flags(d):
    """Erken sıçrama: %8-60 yükseliş, hacim >= 3x, günü zirveye yakın kapanış, 20 günlük zirveyi geçmiş."""
    c, h, l, v = d.Close, d.High, d.Low, d.Volume
    ch = c.pct_change()
    vx = v / v.shift(1).rolling(20).mean()
    pos = (c - l) / (h - l).replace(0, np.nan)
    hh = h.shift(1).rolling(20).max()
    f = (ch >= SURGE["minmove"]) & (ch <= SURGE["maxmove"]) & (vx >= SURGE["minvx"]) & (pos >= SURGE["pos"]) & (c > hh) & (c * v >= 1e6) & (c >= 1)
    return f.fillna(False)

def _outcome(d, i, stop_pct, days):
    """Ertesi gün açılıştan gir (açılış kapanışın %3'ünden fazla yukarıdaysa girme); stop, 2x hedef, zaman stopu; kayma+maliyet dahil.
    Aynı gün hem stop hem hedef görülürse stop sayılır (temkinli). None = girilmedi."""
    if float(d.Open.iloc[i + 1]) > float(d.Close.iloc[i]) * (1 + S_GAP):
        return None
    entry = float(d.Open.iloc[i + 1]) * (1 + S_SLIP)
    stop, tgt = entry * (1 - stop_pct), entry * (1 + 2 * stop_pct)
    last = min(i + days, len(d) - 1)
    for j in range(i + 1, last + 1):
        if d.Low.iloc[j] <= stop: return -stop_pct - S_COST
        if d.High.iloc[j] >= tgt: return 2 * stop_pct - S_COST
    return float(d.Close.iloc[last]) / entry - 1 - S_COST

def _stats(res):
    a = np.array([o for _, o in res], float); n = len(a)
    if n == 0:
        return dict(EMPTY)
    cut = int(n * 2 / 3)
    tr, te = a[:cut], a[cut:]
    return dict(n=n, win=float((a > 0).mean()), exp=float(a.mean()), med=float(np.median(a)), p10=float(np.percentile(a, 10)),
                train_exp=float(tr.mean()) if len(tr) else 0.0, test_exp=float(te.mean()) if len(te) else 0.0, test_n=int(len(te)))

def surge_study(data, days_allowed=CFG_DAYS):
    """Her (vade, stop) ayarını geçmiş olaylarda dener. Ayar seçimi ilk 2/3'te, doğrulama son 1/3'te yapılır."""
    events = []
    for s, d in data.items():
        try:
            f = surge_flags(d).values
            i, n = 25, len(d)
            while i < n - 6:
                if f[i]:
                    events.append((d.index[i], s, i)); i += 6
                else:
                    i += 1
        except Exception:
            continue
    events.sort(key=lambda e: e[0])
    cfgs = []
    for days in days_allowed:
        for sp in CFG_STOPS:
            res = []
            for dt, s, i in events:
                o = _outcome(data[s], i, sp, days)
                if o is not None:
                    res.append((dt, o))
            st_ = _stats(res)
            st_.update(days=days, stop=sp, tgt=2 * sp, skipped=len(events) - len(res))
            st_["ok"] = bool(st_["n"] >= 30 and st_["win"] >= .35 and st_["exp"] > .005 and st_["train_exp"] > 0
                             and st_["test_exp"] > 0 and st_["test_n"] >= 10)
            cfgs.append(st_)
    best = max(cfgs, key=lambda c: (c["ok"], c["train_exp"])) if cfgs else dict(EMPTY)
    return dict(cfgs=cfgs, best=best, n_events=len(events))

def surge_levels(price, stop_pct=.06):
    return dict(entry=price, limit=price * (1 + S_GAP), stop=price * (1 - stop_pct), t1=price * (1 + 2 * stop_pct), R=price * stop_pct)

def run_surge(symbols, pmax, days_allowed=(1,), progress=None):
    """Son 1 yılda benzer sıçramaların sonucunu ölçer, bugünkü adayları bulur. Sonuç 6 saat saklanır."""
    key = f"{pmax}-{len(symbols)}-{days_allowed}-{time.strftime('%Y%m%d')}-v10"
    try:
        c = json.load(open(SURGEF, encoding="utf-8"))
        if c["key"] == key and time.time() - c["t"] < 6 * 3600:
            return c["res"]
    except Exception:
        pass
    data = load_daily(symbols, "1y", 300, 60, progress)
    study = surge_study(data, days_allowed)
    rows = []
    for s, d in data.items():
        try:
            if surge_flags(d).iloc[-1] and d.Close.iloc[-1] <= pmax:
                c_, v_ = d.Close, d.Volume
                rows.append(dict(sym=s, price=float(c_.iloc[-1]), change=float(c_.iloc[-1] / c_.iloc[-2] - 1),
                                 volx=float(v_.iloc[-1] / v_.iloc[-21:-1].mean()), date=str(d.index[-1].date())))
        except Exception:
            continue
    rows.sort(key=lambda r: -r["volx"])
    best = study["best"]
    res = dict(stats=best, cfgs=study["cfgs"], rows=rows, ok=bool(best["ok"]), ts=time.strftime("%H:%M"),
               n_syms=len(data), n_events=study.get("n_events", 0))
    try:
        json.dump(dict(key=key, t=time.time(), res=res), open(SURGEF, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass
    return res

def surge_msg(r, stats, capital, risk, frac=False):
    lv = surge_levels(r["price"], stats["stop"])
    q = lots(capital, risk, dict(R=lv["R"], entry=lv["entry"]), frac)
    cikis = "Yarın kapanışta mutlaka çık." if stats["days"] == 1 else f"{stats['days']} işlem günü içinde hedefe gitmezse çık."
    return (f"🚀 {r['sym']} — erken sıçrama adayı\nFiyat ${r['price']:.2f} · bugün %{r['change']*100:+.0f} · hacim {r['volx']:.0f}x\n"
            f"Alış: yarın açılışta, en fazla ${lv['limit']:.2f} · Stop alıştan %{stats['stop']*100:.0f} aşağı (≈${lv['stop']:.2f}) · "
            f"Hedef %{stats['tgt']*100:.0f} yukarı (≈${lv['t1']:.2f}) · Lot {q}\n⏱ {cikis} Açılış bugünkü kapanışın %3'ünden fazla üstündeyse alma.\n"
            f"Geçmiş {stats['n']} benzer olay: kazanç %{stats['win']*100:.0f}, ortalama %{stats['exp']*100:+.1f}\nYatırım tavsiyesi değildir.")

def surge_alerts(SG, tok, chat, capital, risk, frac=False):
    """Yeni (bildirilmemiş) ve istatistiği pozitif sıçrama adaylarını Telegram'a gönderir ve günlüğe yazar."""
    if not (SG and SG["ok"]):
        return []
    try:
        old = set(json.load(open(STATE2)))
    except Exception:
        old = set()
    keys = {f"{r['sym']}:{r['date']}" for r in SG["rows"]}
    new = [r for r in SG["rows"] if f"{r['sym']}:{r['date']}" not in old]
    try:
        json.dump(sorted(old | keys), open(STATE2, "w"))
    except Exception:
        pass
    st_ = SG["stats"]
    for r in new:
        lv = surge_levels(r["price"], st_["stop"])
        log_add(dict(sym=r["sym"], tarih=r["date"], kaynak="sıçrama", etiket="erken sıçrama", skor=None, entry=round(r["price"], 2),
                     stop=round(lv["stop"], 2), t1=round(lv["t1"], 2), max_days=int(st_["days"]), durum="bekliyor",
                     giris_tarihi=None, cikis_tarihi=None, sonuc=None))
        if tok and chat:
            telegram(tok, chat, surge_msg(r, st_, capital, risk, frac))
    return new

# ---------- Ana tarama ----------
def run_scan(mode, manual, topn, minvol, cost, capital, risk, tok="", chat="", pmin=0, pmax=1e9, frac=False,
             maxatr=1.0, use_mkt=True, use_earn=True):
    if mode.startswith("Nasdaq-100"):
        syms, info = universe_ndx(), "Nasdaq-100 hisseleri tarandı"
    elif mode.startswith("Geniş"):
        allsyms = universe_all()
        syms = liquid(allsyms, topn, minvol, pmin, pmax)
        info = f"Nasdaq’ta {len(allsyms)} hisse bulundu, en likit {len(syms)} tanesi analiz edildi"
    else:
        syms, info = re.findall(r"[A-Za-z.\-]+", manual.upper()), "Elle girilen liste"
    mkt_ok, mkt_txt = market_status() if use_mkt else (True, "Piyasa filtresi kapalı")
    A = {}
    for t, raw in load_daily(dict.fromkeys(syms)).items():
        try:
            if not pmin <= raw.Close.iloc[-1] <= pmax:
                continue
            df = prep(raw)
            if len(df) < 100:
                continue
            bt = run_all(df, cost)
            o = oos(bt)
            sc, why = score(df, o["test_sh"])
            A[t] = dict(df=df, bt=bt, oos=o, score=sc, why=why, plan=plan(df, sc, mkt_ok, maxatr), trend=trendline(df),
                       ts=time.strftime("%d.%m %H:%M UTC", time.gmtime()))
        except Exception:
            continue
    if use_earn:
        for s in [s for s, a in A.items() if a["plan"]["kind"] == "AL"][:20]:
            d = earnings_days(s)
            if d is not None and 0 <= d <= 7:
                p = A[s]["plan"]
                p["kind"], p["label"] = "IZLE", "İZLE (bilanço yakın)"
                p["warn"].append(f"Bilanço {d} gün sonra: sinyal güvenilmez")
    new = detect_new(A, capital, risk, frac) if A else []
    if A:
        log_new(new)
        update_log(A)
    if tok and chat:
        for n in new:
            telegram(tok, chat, n["msg"])
    return A, new, info, (mkt_ok, mkt_txt)

# ===== ARAYÜZ =====
SURUM = "v10 (tek dosya)"



st.set_page_config("Lab Nasdaq", page_icon="📈", layout="wide")
st.markdown("<style>.block-container{padding-top:1.2rem}</style>", unsafe_allow_html=True)
_ph = st.empty()
_ph.caption("⏳ Program yükleniyor…")


DEFAULT = ("AAPL MSFT NVDA AMZN GOOGL META TSLA AVGO COST NFLX AMD ADBE PEP CSCO TMUS INTC "
           "QCOM TXN AMGN INTU AMAT BKNG ISRG LRCX MU PANW ADI KLAC SBUX GILD")

# ---------- Veri ----------
def _clean(d):
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d[["Open", "High", "Low", "Close", "Volume"]].dropna()

@st.cache_data(ttl=60)
def load_intra(sym):
    d = yf.download(sym, period="5d", interval="5m", auto_adjust=True, progress=False)
    d = _clean(d)
    d["svwap"] = session_vwap(d)
    return d

@st.cache_data(ttl=300, show_spinner="Hisseler taranıyor… (geniş taramada 1-3 dk sürebilir)")
def scan(mode, manual, topn, minvol, cost, capital, risk, tok, chat, pmin, pmax, frac, maxatr, use_mkt, use_earn):
    return run_scan(mode, manual, topn, minvol, cost, capital, risk, tok, chat, pmin, pmax, frac, maxatr, use_mkt, use_earn)

@st.cache_data(ttl=300, show_spinner="Hisse inceleniyor…")
def inspect(sym):
    return inspect_symbol(sym)

# ---------- Kenar çubuğu ----------
with st.sidebar:
    st.header("Ayarlar")
    mode = st.radio("Hisseleri nasıl bulsun?", ["Nasdaq-100 (hızlı)", "Geniş Nasdaq taraması", "Elle liste"], index=1)
    topn, minvol, manual = 0, 0, ""
    if mode == "Geniş Nasdaq taraması":
        topn = st.slider("Analiz edilecek en likit hisse sayısı", 30, 300, 120, 10)
        minvol = st.number_input("Min. günlük işlem hacmi (milyon $)", 1, 1000, 10, 1) * 1e6
    elif mode == "Elle liste":
        manual = st.text_area("Hisseler (boşlukla ayır)", DEFAULT, height=100)
    capital = st.number_input("Sermaye ($)", 10, 10_000_000, 100, 10)
    risk = st.number_input("İşlem başı risk (%)", 0.1, 10.0, 1.0, 0.1)
    cost = st.number_input("İşlem maliyeti (her yön, %)", 0.0, 1.0, 0.05, 0.01) / 100
    c1, c2 = st.columns(2)
    pmin = c1.number_input("Min. fiyat ($)", 0.5, 10_000.0, 3.0, 0.5)
    pmax = c2.number_input("Maks. fiyat ($)", 1.0, 100_000.0, 30.0, 1.0)
    frac = st.toggle("Parçalı (kesirli) hisse alabilirim", False)
    maxatr = st.slider("Maks. günlük oynaklık (ATR %)", 2, 20, 8) / 100
    use_mkt = st.toggle("Piyasa filtresi (QQQ düşüşteyse alım önerme)", True)
    use_earn = st.toggle("Bilanço yakınsa uyar", True)
    vade = st.selectbox("Ani sıçrama stratejisi vadesi", ["1 gün (alıp ertesi gün çık)", "1-3 gün", "Hepsi (en iyisini bul)"], index=0)
    days_allowed = {"1 gün (alıp ertesi gün çık)": (1,), "1-3 gün": (1, 2, 3)}.get(vade, (1, 2, 3, 5))
    with st.expander("🔔 Telegram uyarısı"):
        tok = st.text_input("Bot token", os.getenv("TELEGRAM_TOKEN", ""), type="password")
        chat = st.text_input("Chat ID", os.getenv("TELEGRAM_CHAT_ID", ""))
    auto = st.toggle("Otomatik tara (5 dk)", True)
    if st.button("🔄 Şimdi tara", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    st.caption("Veri: Yahoo Finance (yfinance), genelde ~15 dk gecikmeli. Telegram için @BotFather ile bot açıp token ve chat ID girin.")

st.title("📈 Lab Nasdaq")
st.caption("Sürüm: " + SURUM)
_ph.empty()
if not st.session_state.get("go"):
    st.info("Ayarlar sol üstteki **>>** menüsündedir (fiyat aralığı, sermaye, tarama türü). Hazırsanız taramayı başlatın.")
    if st.button("▶ Taramayı başlat", type="primary", use_container_width=True):
        st.session_state["go"] = True
        st.rerun()
    st.stop()
SG = None
mode_eff, manual_eff, info_pre = mode, manual, ""
if mode == "Geniş Nasdaq taraması":
    bar = st.progress(0.0, text="Nasdaq hisseleri taranıyor… (ilk seferde birkaç dakika sürebilir)")
    try:
        allsyms = universe_all()
        syms = liquid(allsyms, topn, minvol, pmin, pmax,
                      progress=lambda x: bar.progress(min(x, 1.0), text=f"Nasdaq hisseleri taranıyor… %{int(x * 100)}"))
    except Exception as e:
        bar.empty()
        st.error("Nasdaq hisse listesi alınamadı: " + str(e)[:150] + ". Birkaç dakika sonra yeniden deneyin ya da sol menüden Nasdaq-100’ü seçin.")
        st.stop()
    bar.empty()
    if not syms:
        st.error("Bu fiyat ve hacim şartlarına uyan hisse bulunamadı. Fiyat aralığını genişletin ya da hacim şartını düşürün.")
        st.stop()
    sbar = st.progress(0.0, text="Ani sıçrama adayları ve geçmiş sonuçları taranıyor…")
    try:
        SG = run_surge(liquid(allsyms, 400, 1e6, 1.0, pmax), pmax, days_allowed,
                               lambda x: sbar.progress(min(x, 1.0), text=f"Ani sıçrama analizi… %{int(x * 100)}"))
        surge_alerts(SG, tok, chat, capital, risk, frac)
    except Exception:
        SG = None
    sbar.empty()
    mode_eff, manual_eff = "Elle liste", " ".join(syms)
    info_pre = f"Nasdaq’ta {len(allsyms)} hisse bulundu, en likit {len(syms)} tanesi analiz edildi"
A, NEW, INFO, MKT = scan(mode_eff, manual_eff, topn, minvol, cost, capital, risk, tok, chat, pmin, pmax, frac, maxatr, use_mkt, use_earn)
if info_pre:
    INFO = info_pre
if not A:
    st.error("Bu fiyat aralığında analiz edilecek hisse bulunamadı. Fiyat aralığını genişletin, hacim şartını düşürün ya da internet bağlantısını kontrol edin.")
    st.stop()
st.caption(f"🔎 {INFO} · {len(A)} hisse analiz edildi")

pc = lambda x, d=1: f"{x*100:+.{d}f}%"
f = lambda v: f"${v:,.2f}"
def layout(fig, h=380):
    fig.update_layout(template="plotly_dark", paper_bgcolor="#0a1330", plot_bgcolor="#0a1330", height=h,
                      margin=dict(l=8, r=8, t=30, b=8), legend=dict(orientation="h", y=1.08),
                      xaxis_rangeslider_visible=False)
    return fig

t1, t2, t3, t4, t5, t6 = st.tabs(["Sinyal", "Strateji Lab", "Stratejiler", "Backtest & Performans", "Günlük", "Radar"])

# ---------- 1) Sinyal ----------
with t1:
    L = sorted(A.items(), key=lambda x: (x[1]["plan"]["kind"] != "AL", -x[1]["score"]))
    (st.success if MKT[0] else st.warning)("📊 Piyasa: " + MKT[1])
    st.subheader("🎯 Bugünün önerisi")
    picks = []
    if SG and SG["ok"]:
        picks += [("surge", r) for r in SG["rows"][:3]]
    picks += [("main", (s, a_)) for s, a_ in L if a_["plan"]["kind"] == "AL"][:max(0, 3 - len(picks))]
    for kind_, obj in picks:
        with st.container(border=True):
            if kind_ == "surge":
                Sg = SG["stats"]
                lv = surge_levels(obj["price"], Sg["stop"])
                q_ = lots(capital, risk, dict(R=lv["R"], entry=lv["entry"]), frac)
                st.markdown(f"### 🚀 {obj['sym']} · {f(obj['price'])} · bugün {pc(obj['change'], 0)} · hacim {obj['volx']:.0f}x")
                k_ = st.columns(3)
                k_[0].metric("Alış (yarın açılışta)", f"≤ {f(lv['limit'])}")
                k_[1].metric(f"Stop (−%{Sg['stop'] * 100:.0f})", f(lv["stop"])); k_[2].metric(f"Hedef (+%{Sg['tgt'] * 100:.0f})", f(lv["t1"]))
                cik_ = "Yarın kapanışta mutlaka çıkın." if Sg["days"] == 1 else f"{Sg['days']} işlem günü içinde hedefe gitmezse çıkın."
                st.caption(f"Lot: {q_} · {cik_} Açılış bugünkü kapanışın %3’ünden fazla üstündeyse almayın. "
                           f"Geçmiş {Sg['n']} benzer olayda kazanç %{Sg['win'] * 100:.0f}, ortalama {pc(Sg['exp'])}.")
            else:
                s_, a2 = obj; p_ = a2["plan"]
                st.markdown(f"### 🟢 {s_} · {f(p_['price'])} · {p_['label']}")
                k_ = st.columns(3)
                k_[0].metric("Alış", f(p_["entry"])); k_[1].metric("Stop", f(p_["stop"])); k_[2].metric("Hedef", f(p_["t1"]))
                st.caption(f"Skor {a2['score']:.0f} · Lot: {lots(capital, risk, p_, frac)} · {p_['max_days']} işlem günü içinde hedefe gitmezse çıkın.")
    if not picks:
        st.info("Bugün alınacak güçlü aday yok. Beklemek de bir karardır; program yeni aday bulunca uyarır.")
    st.caption("Fiyatlar Yahoo Finance’ten gelir (yaklaşık 15 dk gecikmeli); aracı kurum fiyatından birkaç sent farklı olabilir. Alış ve stop rakamları yalnızca alım önerisi olan hisselerde gösterilir.")
    st.caption(f"🛑 Günlük zarar limiti: {f(capital * 0.05)} (sermayenin %5’i). Ulaşırsanız o gün işlemi bırakın; kaybı kovalamak sermayeyi en hızlı eriten şeydir.")
    if SG:
        S_ = SG["stats"]
        with st.expander("📊 Ani sıçrama yapan hisselerde geçmiş 1 yılın sonucu"):
            if SG["cfgs"]:
                st.markdown(f"**En iyi ayar:** {S_['days']} gün · stop %{S_['stop'] * 100:.0f} · hedef %{S_['tgt'] * 100:.0f} → "
                            f"**{S_['n']} olay**, kazanç **%{S_['win'] * 100:.0f}**, ortalama **{pc(S_['exp'])}**, son dönem (doğrulama) {pc(S_['test_exp'])}")
                st.write("✅ Hem seçim döneminde hem hiç görmediği doğrulama döneminde olumlu: program bu tür adayları öneriyor." if SG["ok"] else
                         "⛔ Hiçbir ayar hem seçim hem doğrulama döneminde olumlu çıkmadı: program ani sıçrama hisselerine girmenizi **önermiyor**.")
                T_ = pd.DataFrame([{"Vade (gün)": c["days"], "Stop": f"%{c['stop'] * 100:.0f}", "Hedef": f"%{c['tgt'] * 100:.0f}", "Olay": c["n"],
                                    "Kazanç": f"%{c['win'] * 100:.0f}", "Ortalama": pc(c["exp"]), "Doğrulama": pc(c["test_exp"]),
                                    "Geçerli": "✅" if c["ok"] else "—"} for c in SG["cfgs"]])
                st.dataframe(T_, hide_index=True, use_container_width=True)
            st.caption("Kural: %8-60 yükseliş, hacim ≥3x, günü zirveye yakın kapanış, 20 günlük zirve kırılımı. Ertesi gün açılıştan giriş (açılış %3'ten fazla yukarıdaysa girilmez), "
                       "hedef = 2×stop, %1 maliyet ve kayma dahil. Aynı gün stop ve hedef görülürse stop sayılır. "
                       "Uyarı: yalnızca bugün listede olan hisseler incelendi, borsadan çıkarılmış hisseler yok; gerçek sonuç bundan daha kötü olabilir.")
    if NEW:
        st.error("🔔 **YENİ SİNYAL:** " + ", ".join(n["sym"] for n in NEW))
        st.toast("Yeni sinyal: " + ", ".join(n["sym"] for n in NEW), icon="🔔")
    top = [s for s, a in L if a["plan"]["kind"] == "AL"]
    if top:
        st.success(f"⚠ {len(top)} hisse alım adayı: " + ", ".join(f"**{s}** ({A[s]['score']:.0f})" for s in top) +
                   ". Kırılım, alış ve stop seviyelerine uyun; stop olmadan işleme girmeyin.")
    else:
        st.warning("⚠ Şu an güçlü aday yok (skorlar 70’in altında). Beklemek de bir pozisyondur.")
    for s, a in L[:15]:
        p = a["plan"]
        icon = "🟢" if p["kind"] == "AL" else "🟡" if p["kind"] == "IZLE" else "🔴"
        with st.expander(f"{icon} {s} · {f(p['price'])} · skor {a['score']:.0f} · {p['label']}", expanded=p["kind"] == "AL"):
            st.caption(f"Güncel fiyat {f(p['price'])} · veri: {a.get('ts', '—')} taraması · son bar {a['df'].index[-1].date()}")
            if p["kind"] == "AL":
                al_ = {"pre": "Alış (kırılım olursa)", "extended": "Alış (geri dönüşte)", "retest": "Alış (şimdi)"}.get(p["stage"], "Alış")
                c = st.columns(4)
                c[0].metric("Kırılım seviyesi", f(p["hh"]))
                c[1].metric(al_, f(p["entry"]))
                c[2].metric("Stop", f(p["stop"]), pc(p["stop"] / p["entry"] - 1))
                c[3].metric("Hedef 2R / 3R", f(p["t1"]), f(p["t2"]), delta_color="off")
                q = lots(capital, risk, p, frac)
                st.caption(f"Hisse başı risk {f(p['R'])} (%{p['R']/p['entry']*100:.1f}) · Lot: **{q}** · "
                           f"Maks. kayıp ≈ ${q*p['R']:,.0f} · Pozisyon büyüklüğü ≈ ${q*p['entry']:,.0f}")
                if p["stage"] == "extended":
                    st.info(f"Kırılım gerçekleşti ama fiyat uzaklaştı. Kovalamayın: fiyatın {f(p['hh'])} seviyesine geri dönüp tutunmasını (geri test) bekleyin. Alış {f(p['entry'])} civarı.")
                elif p["stage"] == "retest":
                    st.success("Fiyat kırılım seviyesine geri döndü ve üstünde tutunuyor: geri test bölgesi.")
                st.caption(f"⏱ Zaman stopu: {p['max_days']} işlem günü içinde hedefe gitmezse çıkın. Stop’u aşağı çekmeyin.")
            else:
                st.info(f"Şu an alım önerilmiyor: {p['label']}. Alış ve stop rakamları yalnızca alım önerisi olan hisselerde gösterilir.")
                if p["stage"] == "fail":
                    st.error("Kırılım başarısız: fiyat seviyenin altına döndü (sahte kırılım).")
                c = st.columns(2)
                c[0].metric("Güncel fiyat", f(p["price"]))
                c[1].metric("Kırılım seviyesi", f(p["hh"]), pc(p["price"] / p["hh"] - 1), delta_color="off")
            for w in p["warn"]:
                st.warning(w, icon="⚠️")
            st.markdown("**Neden bu skor?**")
            for t, pts in a["why"]:
                st.write(f"{'🟢' if pts > 0 else '🔴' if pts < 0 else '⚪'} {t} ({pts:+d})")
    if len(L) > 15:
        st.subheader("Diğer hisseler")
        st.dataframe(pd.DataFrame([{"Sembol": s, "Skor": round(a["score"]), "Durum": a["plan"]["label"],
                                    "Fiyat": f(a["plan"]["price"]), "Kırılım seviyesi": f(a["plan"]["hh"]),
                                    "Alış": f(a["plan"]["entry"]) if a["plan"]["kind"] == "AL" else "—",
                                    "Stop": f(a["plan"]["stop"]) if a["plan"]["kind"] == "AL" else "—"} for s, a in L[15:]]),
                     hide_index=True, use_container_width=True)
    st.caption("Kırılım = önceki 20 günün zirvesi. Alış = kırılımın %0,2 üstü (teyit) ya da uzaklaşmışsa EMA9 geri çekilmesi. "
               "Stop = alış − 1,5×ATR(14). Bu bir yatırım tavsiyesi değildir; geçmiş performans geleceği garanti etmez.")

# ---------- 2) Strateji Lab ----------
with t2:
    c1, c2 = st.columns(2)
    sym = c1.selectbox("Hisse", list(A), key="lab_sym")
    sid = c2.selectbox("Strateji", list(NAMES), format_func=NAMES.get, key="lab_st")
    a = A[sym]; df = a["df"].tail(160)
    pos = FUNCS[sid](a["df"]).loc[df.index]
    ent, ext = df[(pos.diff() == 1)], df[(pos.diff() == -1)]
    fig = go.Figure(go.Candlestick(x=df.index, open=df.Open, high=df.High, low=df.Low, close=df.Close, name="Fiyat",
                                   increasing_line_color="#22c58b", decreasing_line_color="#ff5d73"))
    fig.add_scatter(x=df.index, y=df.e9, name="EMA9", line=dict(color="#f5b942", width=1.5))
    fig.add_scatter(x=df.index, y=df.vw, name="VWAP (20g)", line=dict(color="#4d8dff", width=1.5))
    tl = a.get("trend")
    if tl:
        n_ = len(a["df"]); x0 = max(tl["i1"], n_ - len(df)); x1 = n_ - 1
        fig.add_scatter(x=[a["df"].index[x0], a["df"].index[x1]], y=[tl["y1"] + tl["m"] * (x0 - tl["i1"]), tl["y1"] + tl["m"] * (x1 - tl["i1"])],
                        name="Düşen trend çizgisi", line=dict(color="#f5b942", dash="dash", width=1.5))
    fig.add_hline(y=a["plan"]["hh"], line_dash="dot", line_color="#4d8dff", annotation_text="Kırılım seviyesi")
    fig.add_scatter(x=ent.index, y=ent.Low * .985, mode="markers", name="Alım", marker=dict(symbol="triangle-up", size=11, color="#22c58b"))
    fig.add_scatter(x=ext.index, y=ext.High * 1.015, mode="markers", name="Satım", marker=dict(symbol="triangle-down", size=11, color="#ff5d73"))
    st.plotly_chart(layout(fig, 430), use_container_width=True)
    if st.toggle("Gün içi (5 dk) seans VWAP’ını göster"):
        try:
            d5 = load_intra(sym); day = d5.index.date[-1]; d5 = d5[d5.index.date == day]
            fi = go.Figure(go.Scatter(x=d5.index, y=d5.Close, name="Fiyat", line=dict(color="#e8edff")))
            fi.add_scatter(x=d5.index, y=d5.svwap, name="Seans VWAP", line=dict(color="#4d8dff", width=2))
            st.plotly_chart(layout(fi, 300), use_container_width=True)
        except Exception:
            st.info("Gün içi veri alınamadı (piyasa kapalı olabilir).")
    rows = [{"Strateji": NAMES[k], "Getiri": r["tot"], "Al-tut": r["bhr"], "Kazanç %": r["win"], "İşlem": r["n"],
             "Profit factor": r["pf"], "Maks. düşüş": r["mdd"], "Sharpe": r["sh"]} for k, r in a["bt"].items()]
    T = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    st.subheader("Strateji karşılaştırması")
    st.dataframe(T.style.format({"Getiri": "{:+.1%}", "Al-tut": "{:+.1%}", "Kazanç %": "{:.0%}", "Profit factor": "{:.2f}",
                                 "Maks. düşüş": "{:.1%}", "Sharpe": "{:.2f}"}), hide_index=True, use_container_width=True)
    st.caption("İlk satır en yüksek Sharpe’a sahip stratejidir.")

# ---------- 3) Stratejiler (tüm hisselerde) ----------
with t3:
    st.caption(f"{len(A)} hissede eşit ağırlıklı ortalama. Sıralama: Sharpe.")
    cards = []
    for k in NAMES:
        eq = pd.concat([x["bt"][k]["eq"] for x in A.values()], axis=1).ffill().mean(axis=1)
        rs = [x["bt"][k] for x in A.values()]
        nt = sum(r["n"] for r in rs)
        g = lambda d: eq.iloc[-1] / eq.iloc[-1 - d] - 1
        cards.append(dict(k=k, eq=eq, nt=nt, win=sum(r["win"] * r["n"] for r in rs) / nt if nt else 0,
                          pf=sum(r["pf"] for r in rs) / len(rs), mdd=sum(r["mdd"] for r in rs) / len(rs),
                          sh=sum(r["sh"] for r in rs) / len(rs), d7=g(7), d30=g(30), d90=g(90)))
    for i, c in enumerate(sorted(cards, key=lambda x: -x["sh"])):
        with st.container(border=True):
            st.subheader(f"{NAMES[c['k']]}  {'🏆' if i == 0 else ''}")
            fg = go.Figure(go.Scatter(x=c["eq"].index, y=c["eq"], fill="tozeroy", line=dict(color="#4d8dff"),
                                      fillcolor="rgba(77,141,255,.15)"))
            fg.update_yaxes(range=[c["eq"].min() * .98, c["eq"].max() * 1.02])
            st.plotly_chart(layout(fg, 160), use_container_width=True, key=f"m{c['k']}")
            m = st.columns(4)
            m[0].metric("Toplam işlem", c["nt"]); m[1].metric("Kazanç oranı", f"{c['win']:.0%}")
            m[2].metric("Profit factor", f"{c['pf']:.2f}"); m[3].metric("Ort. maks. düşüş", pc(c["mdd"]))
            m = st.columns(3)
            m[0].metric("7 gün", pc(c["d7"])); m[1].metric("30 gün", pc(c["d30"])); m[2].metric("90 gün", pc(c["d90"]))
    st.subheader("Son aktiviteler")
    act = [{"Sembol": s, "Karar": {"AL": "İşlem aç", "IZLE": "Beklet", "UZAK": "Reddedildi"}[a["plan"]["kind"]],
            "Skor": round(a["score"]), "Açıklama": ", ".join([t for t, p in a["why"] if p > 0][:2]) or a["why"][0][0]}
           for s, a in L[:40]]
    st.dataframe(pd.DataFrame(act), hide_index=True, use_container_width=True)

# ---------- 4) Backtest & Performans ----------
with t4:
    c1, c2 = st.columns(2)
    sym4 = c1.selectbox("Hisse", list(A), key="bt_sym")
    sid4 = c2.selectbox("Strateji", list(NAMES), format_func=NAMES.get, key="bt_st")
    r = A[sym4]["bt"][sid4]
    fe = go.Figure()
    fe.add_scatter(x=r["eq"].index, y=r["eq"], name="Strateji", line=dict(color="#e8edff", width=2))
    fe.add_scatter(x=r["bh"].index, y=r["bh"], name="Al-tut", line=dict(color="#4d8dff"))
    st.plotly_chart(layout(fe, 340), use_container_width=True)
    K = [("Toplam getiri", pc(r["tot"])), ("Al-tut", pc(r["bhr"])), ("Kazanma oranı", f"{r['win']:.0%}"), ("İşlem sayısı", r["n"]),
         ("Ort. kazanç", pc(r["aw"], 2)), ("Ort. kayıp", pc(r["al"], 2)), ("Profit factor", f"{r['pf']:.2f}"),
         ("Maks. düşüş", pc(r["mdd"])), ("Sharpe", f"{r['sh']:.2f}"), ("Risk/Getiri", f"{r['rr']:.2f}"),
         ("Piyasada kalma", f"{r['expo']:.0%}"), ("Beklenti/işlem", pc(r["win"] * r["aw"] + (1 - r["win"]) * r["al"], 2))]
    for i in range(0, 12, 4):
        cc = st.columns(4)
        for j, (k, v) in enumerate(K[i:i + 4]):
            cc[j].metric(k, v)
    o = A[sym4]["oos"]
    st.info(f"**Veri dışı doğrulama:** ilk %60’lık dönemde en iyi çıkan strateji **{NAMES[o['k']]}** (Sharpe {o['train_sh']:.2f}). "
            f"Hiç görmediği son %40’lık dönemde: getiri {pc(o['test_tot'])}, Sharpe {o['test_sh']:.2f}, al-tut {pc(o['test_bh'])}. "
            "İki sonuç arasındaki fark, backtestin ne kadar iyimser olduğunu gösterir.")
    fd = go.Figure(go.Scatter(x=r["dd"].index, y=r["dd"], fill="tozeroy", line=dict(color="#ff5d73"), fillcolor="rgba(255,93,115,.2)"))
    st.subheader("Düşüş (drawdown)")
    st.plotly_chart(layout(fd, 220), use_container_width=True)
    st.subheader("İşlemler")
    tr = r["trades"].iloc[::-1].copy()
    if len(tr):
        tr["Giriş"], tr["Çıkış"] = tr["Giriş"].dt.date, tr["Çıkış"].dt.date
    st.dataframe(tr.style.format({"Getiri": "{:+.2%}"}), hide_index=True, use_container_width=True)

# ---------- 5) Günlük (sinyal sonuçları) ----------
with t5:
    LG = update_log(A)
    done = [r for r in LG if r["durum"] in ("hedefe ulaştı", "stopa değdi", "süre doldu")]
    wins = [r for r in done if r["sonuc"] is not None and r["sonuc"] > 0]
    m = st.columns(4)
    m[0].metric("Toplam kayıt", len(LG)); m[1].metric("Sonuçlanan", len(done))
    m[2].metric("Kazanç oranı", f"{len(wins) / len(done):.0%}" if done else "—")
    m[3].metric("Ort. sonuç", pc(sum(r["sonuc"] for r in done) / len(done), 2) if done else "—")
    if len(done) < 30:
        st.caption("Güvenilir bir başarı oranı için en az 30-50 sonuçlanmış sinyal gerekir. O zamana kadar bu rakamlara tek başına güvenmeyin.")
    if LG:
        T = pd.DataFrame(LG[::-1])
        T["sonuc"] = T["sonuc"].map(lambda x: "" if x is None or x != x else f"{x:+.1%}")
        cols = ["sym", "tarih", "kaynak", "durum", "entry", "stop", "t1", "giris_tarihi", "cikis_tarihi", "sonuc"]
        T = T[cols].rename(columns={"sym": "Sembol", "tarih": "Sinyal tarihi", "kaynak": "Kaynak", "durum": "Durum", "entry": "Alış",
                                    "stop": "Stop", "t1": "Hedef", "giris_tarihi": "Giriş", "cikis_tarihi": "Çıkış", "sonuc": "Sonuç"})
        st.dataframe(T, hide_index=True, use_container_width=True)
    else:
        st.info("Henüz kayıt yok. Program yeni bir alım sinyali ürettiğinde burada otomatik görünür. Kendi aldığınız hisseyi aşağıdan elle ekleyebilirsiniz.")
    st.caption("Durumlar: bekliyor = alış seviyesine henüz değmedi, açık = girildi, girilmedi = 5 günde alış seviyesine değmedi. "
               "Aynı gün hem stop hem hedef görülürse stop sayılır (temkinli). Sonuçlar işlem maliyetini içerir.")
    with st.expander("➕ Aldığım hisseyi elle kaydet"):
        with st.form("manuel"):
            ms = st.text_input("Sembol (örn. MAT)").upper().strip()
            k1, k2, k3 = st.columns(3)
            me = k1.number_input("Alış fiyatı", 0.0, 100000.0, 0.0, 0.01)
            mst = k2.number_input("Stop", 0.0, 100000.0, 0.0, 0.01)
            mt = k3.number_input("Hedef", 0.0, 100000.0, 0.0, 0.01)
            md = st.date_input("Alış tarihi")
            if st.form_submit_button("Kaydet"):
                if ms and 0 < mst < me < mt:
                    log_add(dict(sym=ms, tarih=str(md), kaynak="elle", etiket="elle", skor=None, entry=me, stop=mst, t1=mt,
                                         max_days=10, durum="açık", giris_tarihi=str(md), cikis_tarihi=None, sonuc=None))
                    st.success(ms + " kaydedildi.")
                else:
                    st.error("Sembol girin ve stop < alış < hedef olacak şekilde doldurun.")
    st.download_button("⬇ Günlüğü indir (JSON)", json.dumps(LG, ensure_ascii=False, indent=1), "sinyal_gunlugu.json", "application/json")
    up = st.file_uploader("Günlüğü geri yükle (JSON)", type="json")
    if up is not None and st.button("Yükle ve birleştir"):
        try:
            merge_log(json.load(up))
            st.rerun()
        except Exception:
            st.error("Dosya okunamadı.")
    st.caption("⚠ Streamlit Cloud uygulama yeniden başladığında günlük silinebilir. Düzenli olarak indirin; Telegram mesajları da ayrı bir kayıt tutar.")

# ---------- 6) Radar ----------
with t6:
    st.subheader("🔎 Hisse incele")
    q = st.text_input("Sembol yazın (örn. RETO, CLRO)", "").upper().strip()
    if q:
        X = inspect(q)
        if not X:
            st.error(f"{q} için veri bulunamadı. Sembolü kontrol edin.")
        else:
            m = st.columns(4)
            m[0].metric("Fiyat", f(X["price"])); m[1].metric("Bugün", pc(X["ch1"]))
            m[2].metric("5 gün", pc(X["ch5"])); m[3].metric("20 gün", pc(X["ch20"]))
            m = st.columns(4)
            m[0].metric("Hacim / ortalama", f"{X['vx']:.1f}x"); m[1].metric("1 yıl aralığı", f"{f(X['lo'])} – {f(X['hi'])}")
            m[2].metric("60 gün zirvesinden", pc(X["from_peak"])); m[3].metric("Günlük işlem", f"${X['dollar'] / 1e6:.1f} M")
            d_ = X["raw"].tail(120)
            fg = go.Figure(go.Candlestick(x=d_.index, open=d_.Open, high=d_.High, low=d_.Low, close=d_.Close,
                                          increasing_line_color="#22c58b", decreasing_line_color="#ff5d73"))
            st.plotly_chart(layout(fg, 340), use_container_width=True, key="insp")
            for w in X["flags"]:
                st.warning(w, icon="⚠️")
            if X.get("plan"):
                p = X["plan"]
                st.markdown(f"**Programın görüşü:** skor {X['score']:.0f} · {p['label']}")
                k = st.columns(3)
                k[0].metric("Kırılım seviyesi", f(p["hh"])); k[1].metric("Alış", f(p["entry"])); k[2].metric("Stop", f(p["stop"]))
                st.caption("Program bu skoru normal hisseler için ayarladı; aşırı oynak mikro hisselerde güvenilir değildir.")
            else:
                st.caption("Yeterli geçmiş veri olmadığı için program bu hisse için plan üretemedi.")
    st.divider()
    st.subheader("📡 Ani hareket radarı")
    st.caption("Bugün fiyatı ve hacmi birlikte sıçrayan hisseleri listeler. Bu bir alım listesi değildir: sıçrama genelde çoktan olmuştur.")
    r1, r2 = st.columns(2)
    mm = r1.slider("Min. günlük yükseliş (%)", 10, 200, 30)
    vxm = r2.slider("Min. hacim katı", 2, 50, 5)
    if st.button("📡 Radarı tara", use_container_width=True):
        bar = st.progress(0.0, text="Nasdaq hisseleri taranıyor… (birkaç dakika sürebilir)")
        try:
            rows = run_radar(mm / 100, vxm, lambda x: bar.progress(min(x, 1.0), text=f"Taranıyor… %{int(x * 100)}"))
            st.session_state["radar"] = (rows, time.strftime("%H:%M"))
        except Exception as e:
            st.error("Radar çalışmadı: " + str(e)[:120])
        bar.empty()
    RR = st.session_state.get("radar")
    if RR:
        rows, tm = RR
        st.caption(f"Son tarama: {tm} · {len(rows)} hisse")
        if rows:
            T = pd.DataFrame(rows).rename(columns={"Gün5": "5 gün", "Hacimx": "Hacim katı", "GünlükDolar": "Günlük $"})
            T["Fiyat"] = T["Fiyat"].map(lambda x: f"${x:,.2f}"); T["Bugün"] = T["Bugün"].map(lambda x: f"{x:+.0%}")
            T["5 gün"] = T["5 gün"].map(lambda x: f"{x:+.0%}"); T["Hacim katı"] = T["Hacim katı"].map(lambda x: f"{x:.0f}x")
            T["Günlük $"] = T["Günlük $"].map(lambda x: f"{x / 1e6:.1f} M")
            st.dataframe(T[["Sembol", "Fiyat", "Bugün", "5 gün", "Hacim katı", "Günlük $", "Uyarı"]], hide_index=True, use_container_width=True)
            st.caption("Bir sembolü yukarıdaki kutuya yazıp incelemenizi öneririm. Programın önerdiği hisseler değildir.")
        else:
            st.info("Bu eşiklerle hisse bulunamadı. Eşikleri düşürün.")

if auto and st_autorefresh:
    st_autorefresh(interval=300_000, key="auto")
