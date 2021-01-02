from datetime import datetime
import numpy as np
import tushare as ts
import sys

try:
    from sklearn import svm
except:
    print('请安装scikit-learn库和带mkl的numpy')
    sys.exit(-1)


def init(context):
    sid = '600848'
    recent_data = ts.get_hist_data(sid, '2020-06-01', '2020-07-30')
    recent_data = recent_data.iloc[::-1]
    days_value = recent_data.index
    days_close = recent_data['close'].values
    days = []
    # 获取行情日期列表
    print('准备数据训练SVM')
    for i in range(len(days_value)):
        days.append(str(days_value[i])[0:10])
    x_all = []
    y_all = []
    for index in range(15, (len(days) - 5)):
        # 计算三星期共15个交易日相关数据
        start_day = days[index - 15]
        end_day = days[index]
        data = ts.get_hist_data(sid, start=start_day, end=end_day)
        close = data['close'].values
        max_x = data['high'].values
        min_n = data['low'].values
        amount = data['volume'].values
        volume = []
        for i in range(len(close)):
            volume_temp = amount[i] / close[i]
            volume.append(volume_temp)
        close_mean = close[-1] / np.mean(close)  # 收盘价/均值
        volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
        max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
        min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
        vol = volume[-1]  # 现量
        return_now = close[-1] / close[0]  # 区间收益率
        std = np.std(np.array(close), axis=0)  # 区间标准差
        # 将计算出的指标添加到训练集X
        # features用于存放因子
        features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
        x_all.append(features)

    for i in range(len(days_close) - 20):
        if days_close[i + 20] > days_close[i + 15]:
            label = 1
        else:
            label = 0
        y_all.append(label)
    x_train = x_all[: -1]
    y_train = y_all[: -1]
    # 训练SVM
    clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                          tol=0.001, cache_size=200, verbose=False, max_iter=-1,
                          decision_function_shape='ovr', random_state=None)
    clf.fit(x_train, y_train)
    print('训练完成!')

init("")
