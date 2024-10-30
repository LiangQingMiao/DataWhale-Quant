# # sapce 是参数空间，定义贝叶斯搜索的空间
# # func 技术指标名称
# # fast slow 为技术指标的参数范围
# space = {
#         "HQDf": HQDf,
#         "loadBars": 40,
#         "func": doubleMa,
#         "fast": hp.quniform("fast", 3, 30, 1),
#         "slow": hp.quniform("slow", 5, 40, 1),
#     }

# # 调用贝叶斯搜索，第一个参数为参数空间，第二个为优化目标（求解优化目标极值）
# trials, best = hypeFun(space, 'sharpe_ratio')

# BestResultTSDf, BestStatDf = CTA(HQDf, 30, doubleMa, **best)
# plotResult(BestResultTSDf)
