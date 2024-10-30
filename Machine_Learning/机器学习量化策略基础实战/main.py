from biglearning.api import M
from biglearning.api import tools as T
from bigdatasource.api import DataSource
from biglearning.module2.common.data import Outputs
from zipline.finance.commission import PerOrder


# 对训练数据和测试数据进行标准化处理
def m6_run_bigquant_run(input_1, input_2, input_3):
    train_df = input_1.read()
    features = input_2.read()
    feature_min = train_df[features].quantile(0.005)
    feature_max = train_df[features].quantile(0.995)
    train_df[features] = train_df[features].clip(feature_min,feature_max,axis=1) 
    data_1 = DataSource.write_df(train_df)
    test_df = input_3.read()
    test_df[features] = test_df[features].clip(feature_min,feature_max,axis=1)
    data_2 = DataSource.write_df(test_df)
    return Outputs(data_1=data_1, data_2=data_2, data_3=None)

# 后处理函数
def m6_post_run_bigquant_run(outputs):
    return outputs

# 处理每个交易日的数据
def m4_handle_data_bigquant_run(context, data):
    context.extension['index'] += 1
    if  context.extension['index'] % context.rebalance_days != 0:
        return 
    
    date = data.current_dt.strftime('%Y-%m-%d')
    
    cur_data = context.indicator_data[context.indicator_data['date'] == date]

    cur_data = cur_data[cur_data['pred_label'] == 1.0]
    
    stock_to_buy =  list(cur_data.sort_values('instrument',ascending=False).instrument)[:context.stock_num]
    if date == '2017-02-06':
        print(date, len(stock_to_buy), stock_to_buy)

    # 获取当前持仓股票
    stock_hold_now = [equity.symbol for equity in context.portfolio.positions]
    
    # 需要保留的股票
    no_need_to_sell = [i for i in stock_hold_now if i in stock_to_buy]

    # 需要卖出的股票
    stock_to_sell = [i for i in stock_hold_now if i not in no_need_to_sell]
  

    for stock in stock_to_sell:
        if data.can_trade(context.symbol(stock)):
            context.order_target_percent(context.symbol(stock), 0)
    
    if len(stock_to_buy) == 0:
        return

    weight =  1 / len(stock_to_buy)
    
    for stock in stock_to_buy:
        if data.can_trade(context.symbol(stock)):
            context.order_target_percent(context.symbol(stock), weight)
 
# 准备工作
def m4_prepare_bigquant_run(context):
    pass

# 初始化策略
def m4_initialize_bigquant_run(context):
    context.indicator_data = context.options['data'].read_df()
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    context.rebalance_days = 5
    context.stock_num = 50
    if 'index' not in context.extension:
        context.extension['index'] = 0


# 开盘前处理函数
def m4_before_trading_start_bigquant_run(context, data):
    pass


# 获取2020年至2021年股票数据
m1 = M.instruments.v2(
    start_date='2020-01-01',
    end_date='2021-01-01',
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

# 使用高级自动标注器获取标签
m2 = M.advanced_auto_labeler.v2(
    instruments=m1.data,
    label_expr="""shift(close, -5) / shift(open, -1)-1
rank(label)
where(label>=0.95,1,0)""",
    start_date='',
    end_date='',
    benchmark='000300.SHA',
    drop_na_label=False,
    cast_label_int=False
)

# 输入特征
m3 = M.input_features.v1(
    features="""(close_0-mean(close_0,12))/mean(close_0,12)*100
rank(std(amount_0,15))
rank_avg_amount_0/rank_avg_amount_8
ts_argmin(low_0,20)
rank_return_30
(low_1-close_0)/close_0
ta_bbands_lowerband_14_0
mean(mf_net_pct_s_0,4)
amount_0/avg_amount_3
return_0/return_5
return_1/return_5
rank_avg_amount_7/rank_avg_amount_10
ta_sma_10_0/close_0
sqrt(high_0*low_0)-amount_0/volume_0*adjust_factor_0
avg_turn_15/(turn_0+1e-5)
return_10
mf_net_pct_s_0
(close_0-open_0)/close_1
 """
)

# 抽取基础特征
m15 = M.general_feature_extractor.v7(
    instruments=m1.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=0
)

# 提取派生特征
m16 = M.derived_feature_extractor.v3(
    input_data=m15.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False
)

# 合并标签和特征
m7 = M.join.v3(
    data1=m2.data,
    data2=m16.data,
    on='date,instrument',
    how='inner',
    sort=False
)

# 删除缺失值
m13 = M.dropnan.v1(
    input_data=m7.data
)

# 获取2021年至2022年股票数据
m9 = M.instruments.v2(
    start_date=T.live_run_param('trading_date', '2021-01-01'),
    end_date=T.live_run_param('trading_date', '2022-01-01'),
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

# 抽取基础特征
m17 = M.general_feature_extractor.v7(
    instruments=m9.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=0
)

# 提取派生特征
m18 = M.derived_feature_extractor.v3(
    input_data=m17.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False
)

# 删除缺失值
m14 = M.dropnan.v1(
    input_data=m18.data
)

# 标准化训练数据和测试数据
m6 = M.cached.v3(
    input_1=m13.data,
    input_2=m3.data,
    input_3=m14.data,
    run=m6_run_bigquant_run,
    post_run=m6_post_run_bigquant_run,
    input_ports='',
    params='{}',
    output_ports=''
)

# 对数据进行RobustScaler标准化处理
m8 = M.RobustScaler.v13(
    train_ds=m6.data_1,
    features=m3.data,
    test_ds=m6.data_2,
    scale_type='standard',
    quantile_range_min=0.01,
    quantile_range_max=0.99,
    global_scale=True
)

# 使用SVC进行训练和预测
m10 = M.svc.v1(
    training_ds=m8.train_data,
    features=m3.data,
    predict_ds=m8.test_data,
    C=1,
    kernel='rbf',
    degree=3,
    gamma=-1,
    coef0=0,
    tol=0.1,
    max_iter=100,
    key_cols='date,instrument',
    other_train_parameters={}
)

# 创建交易策略实例
m4 = M.trade.v4(
    instruments=m9.data,
    options_data=m10.predictions,
    start_date='',
    end_date='',
    handle_data=m4_handle_data_bigquant_run,
    prepare=m4_prepare_bigquant_run,
    initialize=m4_initialize_bigquant_run,
    before_trading_start=m4_before_trading_start_bigquant_run,
    volume_limit=0,
    order_price_field_buy='open',
    order_price_field_sell='open',
    capital_base=10000000,
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='后复权',
    product_type='股票',
    plot_charts=True,
    backtest_only=False,
    benchmark=''
)
