from stupids.StupidHead import *


def doubleMa(*args, **kwargs):
    TradedHQDf = args[0]
    fast_ma = talib.SMA(TradedHQDf.close, timeperiod=kwargs['fast'])
    fast_ma0 = fast_ma[-1]
    fast_ma1 = fast_ma[-2]
    slow_ma = talib.SMA(TradedHQDf.close, timeperiod=kwargs['slow'])
    slow_ma0 = slow_ma[-1]
    slow_ma1 = slow_ma[-2]
    cross_over = fast_ma0 > slow_ma0 and fast_ma1 < slow_ma1
    cross_below = fast_ma0 < slow_ma0 and fast_ma1 > slow_ma1
    if cross_over:
        setpos(1, *args)
    elif cross_below:
        setpos(-1, *args)


if __name__ == '__main__':
    HQDf = pd.read_csv('data\T888_1d.csv', index_col='date')
    HQDf.index = pd.to_datetime(HQDf.index)
    ctaParas = {'fast': 5, 'slow': 10}
    ResultTSDf, StatDf = CTA(HQDf, 30, doubleMa, **ctaParas)
    plotResult(ResultTSDf)

    

