import pandas as pd
import matplotlib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import yfinance as yf
import pytz
import datetime as dt
import math
from sklearn.impute import KNNImputer
from scipy.stats import gmean
from dateutil.relativedelta import relativedelta
import time
import dash
from dash import dcc 
from dash import html
from dash.dependencies import Input, Output
import threading

matplotlib.use('agg')

portfolio = []
portfoliod = {}
unrealized_profits = 0
realized_profits = 0
total_profits = unrealized_profits + realized_profits
np.random.seed(42)

plt.style.use("fivethirtyeight")

timezone = pytz.timezone('Europe/Istanbul')
companies_dict = {
    "Adel Kalemcilik": "ADEL.IS",
    "Afyon Çimento": "AFYON.IS",
    "Agahlar Emaye": "AGHOL.IS",
    "Akbank": "AKBNK.IS",
    "Akçansa": "AKCNS.IS",
    "Aken Enerji": "AKENR.IS",
    "Akfen GYO": "AKFGY.IS",
    "Aktaş Gaz": "AKGRT.IS",
    "Akşa Enerji": "AKSA.IS",
    "Alarko Holding": "ALARK.IS",
    "Albaraka Türk": "ALBRK.IS",
    "Alkim Kağıt": "ALCAR.IS",
    "Alarko Gayrimenkul": "ALGYO.IS",
    "Alkim Alkali": "ALKA.IS",
    "Alkim Kimya": "ALKIM.IS",
    "Anel Elektrik": "ANELE.IS",
    "Arçelik": "ARCLK.IS",
    "Aselsan": "ASELS.IS",
    "Asuzac": "ASUZU.IS",
    "Ata GMYO": "ATAGY.IS",
    "Ayçiçek Yağı": "AYCES.IS",
    "Ayen Enerji": "AYEN.IS",
    "Bağfas": "BAGFS.IS",
    "Bak Ambalaj": "BAKAB.IS",
    "Banvit": "BANVT.IS",
    "Beyaz Filo": "BEYAZ.IS",
    "Bim Birleşik Mağazalar": "BIMAS.IS",
    "Bizim Mağazaları": "BIZIM.IS",
    "Beşiktaş": "BJKAS.IS",
    "Bantaş": "BNTAS.IS",
    "Bossa": "BOSSA.IS",
    "Brisa": "BRISA.IS",
    "Berkosan": "BRKO.IS",
    "Çemtaş": "CEMTS.IS",
    "Çelebi": "CLEBI.IS",
    "Çimsa": "CMBTN.IS",
    "Cardinal": "CRDFA.IS",
    "Desa Deri": "DESA.IS",
    "Denge Yatırım": "DENGE.IS",
    "Derimod": "DERIM.IS",
    "Doğuş Gayrimenkul": "DGGYO.IS",
    "Ditaş": "DITAS.IS",
    "Doğuş MS": "DMSAS.IS",
    "Doğan Holding": "DOAS.IS",
    "Dobur Gayrimenkul": "DOBUR.IS",
    "DO & CO": "DOCO.IS",
    "Döktaş": "DOKTA.IS",
    "Durdoğan": "DURDO.IS",
    "Ege Çelik": "ECILC.IS",
    "Ege Endüstri": "EGEEN.IS",
    "Ege Profil": "EGPRO.IS",
    "Emlak Konut": "EKGYO.IS",
    "Emlak GYO": "EKIZ.IS",
    "Emek Elektrik": "EMKEL.IS",
    "Enka İnşaat": "ENJSA.IS",
    "Ereğli Demir Çelik": "EREGL.IS",
    "Fenerbahçe": "FENER.IS",
    "Formet Çelik Kapı": "FMIZP.IS",
    "Ford Otosan": "FROTO.IS",
    "Garanti BBVA": "GARAN.IS",
    "Gedik Yatırım": "GEDIK.IS",
    "Global Yatırım": "GLYHO.IS",
    "Güler Yatırım": "GLRYH.IS",
    "Göltaş Çimento": "GOLTS.IS",
    "Goodyear": "GOODY.IS",
    "GSD Holding": "GSDDE.IS",
    "Galatasaray": "GSRAY.IS",
    "Gübre Fabrikaları": "GUBRF.IS",
    "Halkbank": "HALKB.IS",
    "Halic GMYO": "HLGYO.IS",
    "Hurtaş": "HURGZ.IS",
    "ICBC Turkey": "ICBCT.IS",
    "Idea Yatırım": "IDGYO.IS",
    "İdealist Danışmanlık": "IEYHO.IS",
    "İhlas Ev Aletleri": "IHEVA.IS",
    "İhlas Holding": "IHLAS.IS",
    "Indeks Bilgisayar": "INDES.IS",
    "Info Yatırım": "INFO.IS",
    "Intema": "INTEM.IS",
    "İpek Doğal Enerji": "IPEKE.IS",
    "İşbir Holding": "ISBIR.IS",
    "İşbank": "ISCTR.IS",
    "İsdemir": "ISDMR.IS",
    "İş Mensucat": "ISMEN.IS",
    "İzmir Demir Çelik": "IZMDC.IS",
    "Jant Sanayi": "JANTS.IS",
    "Karel Elektronik": "KAREL.IS",
    "Karadeniz Holding": "KARSN.IS",
    "Katmerciler": "KATMR.IS",
    "Koç Holding": "KCHOL.IS",
    "Kiler GMYO": "KLGYO.IS",
    "Klimasan Klima": "KLMSN.IS",
    "Konya Çimento": "KONYA.IS",
    "Kordsa": "KORDS.IS",
    "Kozaa": "KOZAA.IS",
    "Koza Altın": "KOZAL.IS",
    "Kardemir": "KRDMD.IS",
    "Kron Telekom": "KRONT.IS",
    "Kütahya Porselen": "KUTPO.IS",
    "Lider Faktoring": "LIDFA.IS",
    "Link Bilgisayar": "LINK.IS",
    "Logo Yazılım": "LOGO.IS",
    "Mavi Giyim": "MAVI.IS",
    "Mega Polietilen": "MEGAP.IS",
    "Merit Turizm": "MERIT.IS",
    "Merko GYO": "MERKO.IS",
    "Metro Holding": "METRO.IS",
    "Metro Turizm": "METUR.IS",
    "Migros": "MGROS.IS",
    "Mitaş": "MIPAZ.IS",
    "Mapaş": "MPARK.IS",
    "Marmara GMYO": "MRGYO.IS",
    "Marshall Boya": "MRSHL.IS",
    "Net Holding": "NTHOL.IS",
    "Nurol GYO": "NUGYO.IS",
    "Odaş Elektrik": "ODAS.IS",
    "Oylum": "OYLUM.IS",
    "Özak GYO": "OZGYO.IS",
    "Özkaya GMYO": "OZKGY.IS",
    "Özderici GMYO": "OZRDN.IS",
    "Pegasus GMYO": "PAGYO.IS",
    "Petkim": "PETKM.IS",
    "Pınar Su": "PINSU.IS",
    "Park Elek. Mad.": "PRKME.IS",
    "Piraziz": "PRZMA.IS",
    "Pusula": "PSDTC.IS",
    "Ray Sigorta": "RAYSG.IS",
    "Rodriguez": "RODRG.IS",
    "RTA Laboratuvarları": "RTALB.IS",
    "San-El Mühendislik": "SANEL.IS",
    "Sarı Kırmızı": "SARKY.IS",
    "Sasa Polyester": "SASA.IS",
    "Sayaç Elektrik": "SAYAS.IS",
    "Selek Elektrik": "SELEC.IS",
    "Seyitler Kimya": "SEYKM.IS",
    "Şişe Cam": "SISE.IS",
    "Şekerbank": "SKBNK.IS",
    "Sinpaş GMYO": "SNGYO.IS",
    "Server GMYO": "SRVGY.IS",
    "Tat Gıda": "TATGD.IS",
    "TAV Havalimanları": "TAVHL.IS",
    "Turkcell": "TCELL.IS",
    "Torunlar GMYO": "TDGYO.IS",
    "Tekstilbank": "TEKTU.IS",
    "Turk Silahlı Kuvvetleri": "TGSAS.IS",
    "Turkish Airlines": "THYAO.IS",
    "Tofaş Türk Otomobil Fabrikası": "TKFEN.IS",
    "Tatmsn Tekstil": "TMSN.IS",
    "Toprak Holding": "TOASO.IS",
    "Türk Traktör": "TRCAS.IS",
    "Türkiye Girişim": "TRGYO.IS",
    "Türkiye Sınai Kalkınma": "TSKB.IS",
    "Trabzonspor": "TSPOR.IS",
    "Türk Traktör": "TTRAK.IS",
    "Türk Telekom": "TTKOM.IS",
    "Türk Ticaret": "TTKOM.IS",
    "Türkler Çelik": "TUCLK.IS",
    "Tüpraş": "TUPRS.IS",
    "Ülker Bisküvi": "ULKER.IS",
    "Ulusal Yatırım": "ULUSE.IS",
    "Uşak Seramik": "USAK.IS",
    "VakıfBank": "VAKBN.IS",
    "Vestel": "VESTL.IS",
    "Yataş": "YATAS.IS",
    "Yıldız GMYO": "YGGYO.IS",
    "Yapı GMYO": "YGYO.IS",
    "Yapı Kredi": "YKBNK.IS",
}

#start = dt.datetime.now().replace(hour=10, minute=0, second=0) - dt.timedelta(days=19)
start = dt.datetime(year=2023, month=6, day=15, hour=9, minute=45, second=0)
end = start.replace(hour=10, minute=30, second=0)
goal = start.replace(hour=18, minute=30, second=0)

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Portfolio Dashboard"),
    html.Div([
        html.Div([
            html.H2("Current Portfolio"),
            html.Ul(id="portfolio-list")
        ], className="portfolio-section"),
        html.Div([
            html.H2("Stocks to Buy"),
            html.Ul(id="stocks-bought-list")
        ], className="stocks-section"),
        html.Div([
            html.H2("Stocks to Sell"),
            html.Ul(id="stocks-sold-list")
        ], className="stocks-section")
    ], className="main-section"),
    html.Div([
        html.Div([
            dcc.Graph(id="profits-bar-chart")
        ], className="profits-chart"),
        html.Div([
            dcc.Graph(id="profits-line-chart")
        ], className="profits-chart")
    ], className="profits-section"),
    dcc.Interval(
        id='interval-component',
        interval=15 * 60 * 1000,  
        n_intervals=0
    )
])

@app.callback(
    Output("portfolio-list", "children"),
    Output("stocks-bought-list", "children"),
    Output("stocks-sold-list", "children"),
    Output("profits-bar-chart", "figure"),
    Output("profits-line-chart", "figure"),
    Input("interval-component", "n_intervals")
)
def update_portfolio(n):
    global portfolio, portfoliod, unrealized_profits, realized_profits, total_profits

    portfolio_list = [html.Li(stock) for stock in portfolio]
    stocks_bought_list = [html.Li(stock) for stock in stocks_to_buy if stock]
    stocks_sold_list = [html.Li(stock) for stock in stocks_to_sell if stock]

    profits_data = pd.DataFrame({
        'Profits': [realized_profits, unrealized_profits, total_profits],
        'Category': ['Realized Profits', 'Unrealized Profits', 'Total Profits']
    })

    bar_chart_figure = {
        'data': [
            {'x': profits_data['Category'], 'y': profits_data['Profits'], 'type': 'bar'}
        ],
        'layout': {
            'title': 'Profits',
            'yaxis': {'title': 'Amount'}
        }
    }

    line_chart_figure = {
        'data': [
            {'x': profits_data['Category'], 'y': profits_data['Profits'], 'type': 'line'}
        ], 
        'layout': {
            'title': 'Profits',
            'yaxis': {'title': 'Amount'}
        }
    }

    return portfolio_list, stocks_bought_list, stocks_sold_list, bar_chart_figure, line_chart_figure

if __name__ == "__main__":
    thread = threading.Thread(target=app.run_server, args=("localhost", 8050), daemon=True)
    thread.start()

while end != goal:

    #end = dt.datetime.now().replace(hour=18, minute=30, second=0) - dt.timedelta(days=1)
    #end = dt.datetime.now()
    #start = end.replace(hour=10, minute=0, second=0)
    period = "1d"
    interval = "15m"

    #endn = dt.datetime.now().replace(hour=19, minute=0, second=0) - dt.timedelta(days=19)
    #endn = dt.datetime.now()
    endn = dt.datetime(year=2023, month=6, day=15, hour=18, minute=30, second=0)
    startn = endn - relativedelta(months=1)
    startn = startn.replace(hour=9, minute=45)

    periodn = "1m"
    intervaln = "1h"


    dfgunluk = yf.download(list(companies_dict.values()), start=start, end=end, period=period, interval=interval)['Close']
    dfgunluk

    num_rows = dfgunluk.shape[0]
    num_rows

    daily_returns = dfgunluk.pct_change()

    daily_returns.drop(daily_returns.index[0], axis=0, inplace=True)

    minutereturns = daily_returns.mean() * (num_rows)
    return_var = daily_returns.var() * (num_rows)

    dfgunluk2 = pd.DataFrame(dfgunluk.columns, columns=['Tickers'])
    dfgunluk2['Variance'] = return_var.values
    dfgunluk2['Returns'] = minutereturns.values

    dfgunluk2

    X = dfgunluk2[['Returns', 'Variance']].values

    inertia_list = []

    dfgunluk3 = pd.DataFrame(X, columns = ['Returns','Variance'])

    imputer = KNNImputer(n_neighbors=2, weights="uniform", copy=False)
    imputer.fit_transform(X)

    #np.random.seed(42)

    kmeans = KMeans(n_clusters=4).fit(X)

    cluster_centers = kmeans.cluster_centers_
    center_indices = np.arange(len(cluster_centers))

    sorted_centers_indices = sorted(center_indices, key=lambda x: cluster_centers[x, 0])

    label_mapping = {sorted_centers_indices[i]: i for i in range(len(sorted_centers_indices))}

    labels = np.array([label_mapping[label] for label in kmeans.labels_])

    dfgunluk2['Cluster_Labels'] = labels
    new_dfgunluk = dfgunluk2.iloc[:, [0, 3]]
    new_dfgunluk

    result_dictgunluk = new_dfgunluk.set_index('Tickers')['Cluster_Labels'].to_dict()

    print(result_dictgunluk)

    cl0 = dfgunluk2[dfgunluk2["Cluster_Labels"] == 0]
    print(cl0["Tickers"].tolist())

    cl1 = dfgunluk2[dfgunluk2["Cluster_Labels"] == 1]
    print(cl1["Tickers"].tolist())

    cl2 = dfgunluk2[dfgunluk2["Cluster_Labels"] == 2]
    print(cl2["Tickers"].tolist())

    cl3 = dfgunluk2[dfgunluk2["Cluster_Labels"] == 3]
    print(cl3["Tickers"].tolist())

    plt.scatter(X[:,0], X[:,1], c = labels, cmap = 'rainbow')
    plt.title('Clustered Stocks')
    plt.xlabel('Returns')
    plt.ylabel('Variances')

    dfaylik = yf.download(list(companies_dict.values()), start=startn, end=endn, period=periodn, interval=intervaln)['Close']
    dfaylik

    num_rowsa = dfaylik.shape[0]
    num_rowsa

    daily_returnsa = dfaylik.pct_change()

    daily_returnsa.drop(daily_returnsa.index[0], axis=0, inplace=True)

    minutereturnsa = daily_returnsa.mean() * (num_rowsa)
    return_vara = daily_returnsa.var() * (num_rowsa)

    dfaylik2 = pd.DataFrame(dfaylik.columns, columns=['Tickers'])
    dfaylik2['Variance'] = return_vara.values
    dfaylik2['Returns'] = minutereturnsa.values

    dfaylik2

    X = dfaylik2[['Returns', 'Variance']].values

    inertia_list = []

    dfaylik3 = pd.DataFrame(X, columns = ['Returns','Variance'])

    imputer = KNNImputer(n_neighbors=2, weights="uniform", copy=False)
    imputer.fit_transform(X)

    kmeansa = KMeans(n_clusters=4).fit(X)

    cluster_centersa = kmeansa.cluster_centers_
    center_indicesa = np.arange(len(cluster_centersa))

    sorted_centers_indicesa = sorted(center_indicesa, key=lambda x: cluster_centersa[x, 0])

    label_mappinga = {sorted_centers_indicesa[i]: i for i in range(len(sorted_centers_indicesa))}

    labelsa = np.array([label_mappinga[kmeansa.labels_[i]] for i in range(len(kmeansa.labels_))])

    dfaylik2['Cluster_Labels'] = labelsa
    new_dfaylik = dfaylik2.iloc[:, [0, 3]]
    new_dfaylik

    c0 = dfaylik2[dfaylik2["Cluster_Labels"] == 0]
    print(c0["Tickers"].tolist())

    c1 = dfaylik2[dfaylik2["Cluster_Labels"] == 1]
    print(c1["Tickers"].tolist())

    c2 = dfaylik2[dfaylik2["Cluster_Labels"] == 2]
    print(c2["Tickers"].tolist())

    c3 = dfaylik2[dfaylik2["Cluster_Labels"] == 3]
    print(c3["Tickers"].tolist())

    result_dictaylik = new_dfaylik.set_index('Tickers')['Cluster_Labels'].to_dict()

    print(result_dictaylik)

    plt.scatter(X[:,0], X[:,1], c = labelsa, cmap = 'rainbow')
    plt.title('Clustered Stocks')
    plt.xlabel('Returns')
    plt.ylabel('Variances')

    cluster_transitions = {}

    for stock, cluster1 in result_dictaylik.items():

        if stock in result_dictgunluk:
            cluster2 = result_dictgunluk[stock]
            if cluster1 != cluster2:
                cluster_transitions[stock] = (cluster1, cluster2)

    print("Stocks that changed clusters:")
    for stock, (from_cluster, to_cluster) in cluster_transitions.items():
        print(f"{stock}: {from_cluster} -> {to_cluster}")

    stocks_moved_up = []
    stocks_to_buy = []

    for stock, (from_cluster, to_cluster) in cluster_transitions.items():
        if from_cluster < to_cluster:
            stocks_moved_up.append((stock, from_cluster, to_cluster))
            stocks_to_buy.append((stock))

    print("Stocks that moved up a cluster:")
    for stock, from_cluster, to_cluster in stocks_moved_up:
        print(f"{stock}: {from_cluster} -> {to_cluster}")


    stocks_moved_down = []
    stocks_to_sell = []

    for stock, (from_cluster, to_cluster) in cluster_transitions.items():
        if from_cluster > to_cluster:
            stocks_moved_down.append((stock, from_cluster, to_cluster))
            stocks_to_sell.append((stock))

    print("Stocks that moved down a cluster:")
    for stock, from_cluster, to_cluster in stocks_moved_down:
        print(f"{stock}: {from_cluster} -> {to_cluster}")

    update_interval = 900
    total_profit = 0
    total_portfolio_value = 0
    #portfolio = []

    if len(portfolio) == 0:
        for symbol in stocks_to_buy:
            stock_data = yf.Ticker(symbol)
            price = stock_data.history(start=start, end=end, interval = "15m")['Close'].iloc[-1]
            print(f"Bought {symbol} at {price}")
            #total_portfolio_value += price
            #total_profit -= price
            portfolio.append(symbol)
            portfoliod[symbol] = price
    else:
        for symbol in stocks_to_buy:
            if symbol not in portfolio: 
                stock_data = yf.Ticker(symbol)
                price = stock_data.history(start=start, end=end, interval = "15m")['Close'].iloc[-1]
                print(f"Bought {symbol} at {price}")
                #total_portfolio_value += price
                #total_profit -= price
                portfolio.append(symbol)
                portfoliod[symbol] = price
            elif symbol in portfolio:
                stock_data = yf.Ticker(symbol)
                price = stock_data.history(start=start, end=end, interval = '15m')['Close'].iloc[-1]
                b = price - portfoliod[symbol]
                unrealized_profits += b
        
        for symbol in stocks_to_sell:
            if symbol in portfolio:
                stock_data = yf.Ticker(symbol)
                price = stock_data.history(start=start, end=end, interval = "15m")['Close'].iloc[-1] 
                print(f"Sold {symbol} at {price}")
                #total_portfolio_value -= price
                #total_profit += price
                a = price - portfoliod[symbol]
                realized_profits += a
                portfolio.remove(symbol)
                del portfoliod[symbol]
    
                
        total_profits = unrealized_profits + realized_profits
    
    #print("Current portfolio:", portfolio)
    #print("Stocks to sell:", stocks_to_sell)
    print("Porfolio Dictionary:", portfoliod)
    print("Unrealized Profits:", unrealized_profits)
    print("Realized Profits:", realized_profits)
    print("Total Profits:", total_profits)
    end += dt.timedelta(minutes=15)
    time.sleep(3)