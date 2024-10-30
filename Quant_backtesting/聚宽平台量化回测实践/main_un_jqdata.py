# 导入函数库
from jqdata import *

# ; 选择标的为：002594.XSHE 比亚迪
# ; 选择基准为：000300.XSHG 沪深300
# ; 策略为：当5日线金叉10日线，全仓买入；当5日线死叉10日线全仓卖出。
# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深上证作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
      # 开盘时运行
    run_daily(market_open, time='open', reference_security='000300.XSHG')
      # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')

## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))

    # 给微信发送消息（添加模拟交易，并绑定微信生效）
    # send_message('美好的一天~')

    # 要操作的股票：比亚迪（g.为全局变量）
    g.security = '002594.XSHE'

## 开盘时运行函数
def market_open(context):
    log.info('函数运行时间(market_open):'+str(context.current_dt.time()))
    security = g.security
    # 获取股票的收盘价
    close_data5 = get_bars(security, count=5, unit='1d', fields=['close'])
    close_data10 = get_bars(security, count=10, unit='1d', fields=['close'])
    # close_data20 = get_bars(security, count=20, unit='1d', fields=['close'])
    # 取得过去五天，十天的平均价格
    MA5 = close_data5['close'].mean()
    MA10 = close_data10['close'].mean()
    # 取得上一时间点价格
    #current_price = close_data['close'][-1]
    # 取得当前的现金
    cash = context.portfolio.available_cash

    # 五日均线上穿十日均线
    if (MA5 > MA10) and (cash > 0):
        # 记录这次买入
        log.info("5日线金叉10日线，买入 %s" % (security))
        # 用所有 cash 买入股票
        order_value(security, cash)
    # 五日均线跌破十日均线
    elif (MA5 < MA10) and context.portfolio.positions[security].closeable_amount > 0:
        # 记录这次卖出
        log.info("5日线死叉10日线, 卖出 %s" % (security))
        # 卖出所有股票,使这只股票的最终持有量为0
        for security in context.portfolio.positions.keys():
            order_target(security, 0)

## 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')
