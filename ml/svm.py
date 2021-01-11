from datetime import datetime
import numpy as np
import tushare as ts
import sys

try:
    from sklearn import svm
except:
    print('请安装scikit-learn库和带mkl的numpy')
    sys.exit(-1)

gclf = ""

def init(context):
    sid = context['sid']
    recent_data = ts.get_hist_data(sid, '2020-06-01', '2020-08-30')
    recent_data = recent_data.iloc[::-1]
    # 获取行情日期列表
    print('准备数据训练SVM')
    x_all = []
    y_all = []

    index = 0
    for date, data in recent_data.iterrows():
        close = data['close']
        max_x = data['high']
        min_n = data['low']
        volume = data['volume']
        close_mean = close / data['ma20']  # 收盘价/均值
        volume_mean = volume / data['v_ma20']  # 现量/均量
        max_mean = close/max_x
        min_mean = close/min_n
        turn_over = data['turnover']
        features = [close_mean, volume_mean, max_mean, min_mean, turn_over]
        x_all.append(features)

        #if np.mean(recent_data.iloc[index:index+3]['close']) > close:
        if recent_data.iloc[index+1]['close'] > close:
            label = 1
        else:
            label = 0
        y_all.append(label)

        index = index + 1
    # for index in range(15, (len(days) - 5)):
    #     # 计算三星期共15个交易日相关数据
    #     start_day = days[index - 15]
    #     end_day = days[index]
    #     data = ts.get_hist_data(sid, start=start_day, end=end_day)
    #     close = data['close'].values
    #     ma5 = data['ma5'].values
    #     max_x = data['high'].values
    #     min_n = data['low'].values
    #     amount = data['volume'].values
    #     volume = []
    #     for i in range(len(close)):
    #         volume_temp = amount[i] / close[i]
    #         volume.append(volume_temp)
    #     close_mean = close[-1] / np.mean(close)  # 收盘价/均值
    #     volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
    #     max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
    #     min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
    #     vol = volume[-1]  # 现量
    #     return_now = close[-1] / close[0]  # 区间收益率
    #     std = np.std(np.array(close), axis=0)  # 区间标准差
    #     # 将计算出的指标添加到训练集X
    #     # features用于存放因子
    #     features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
    #     x_all.append(features)
    #
    # for i in range(len(days_close) - 20):
    #     if days_close[i + 16] > days_close[i + 15]:
    #         label = 1
    #     else:
    #         label = 0
    #     y_all.append(label)
    x_train = x_all[: -1]
    y_train = y_all[: -1]
    # 训练SVM
    clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                          tol=0.001, cache_size=200, verbose=False, max_iter=-1,
                          decision_function_shape='ovr', random_state=None)
    clf.fit(x_train, y_train)
    context["clf"] = clf
    global gclf
    gclf = clf
    print('训练完成!')


def on_bar(context):
    sid = context['sid']
    recent_data = ts.get_hist_data(sid, '2020-06-01', '2020-08-30')
    recent_data = recent_data.iloc[::-1]
    x_all = []

    for date, data in recent_data.iterrows():
        close = data['close']
        max_x = data['high']
        min_n = data['low']
        volume = data['volume']
        close_mean = close / data['ma20']  # 收盘价/均值
        volume_mean = volume / data['v_ma20']  # 现量/均量
        max_mean = close/max_x
        min_mean = close/min_n
        turn_over = data['turnover']
        features = [close_mean, volume_mean, max_mean, min_mean, turn_over]
        x_all.append(features)

        features = np.array(features).reshape(1, -1)
        global gclf
        prediction = gclf.predict(features)
        print(date, prediction[0])

    # recent_data = recent_data.iloc[::-1]
    # days_value = recent_data.index
    # days_close = recent_data['close'].values
    # days = []
    # # 获取行情日期列表
    # for i in range(len(days_value)):
    #     days.append(str(days_value[i])[0:10])
    # x_all = []
    # y_all = []
    # for index in range(15, (len(days) - 5)):
    #     # 计算三星期共15个交易日相关数据
    #     start_day = days[index - 15]
    #     end_day = days[index]
    #     data = ts.get_hist_data(sid, start=start_day, end=end_day)
    #     close = data['close'].values
    #     max_x = data['high'].values
    #     min_n = data['low'].values
    #     amount = data['volume'].values
    #     volume = []
    #     for i in range(len(close)):
    #         volume_temp = amount[i] / close[i]
    #         volume.append(volume_temp)
    #     close_mean = close[-1] / np.mean(close)  # 收盘价/均值
    #     volume_mean = volume[-1] / np.mean(volume)  # 现量/均量
    #     max_mean = max_x[-1] / np.mean(max_x)  # 最高价/均价
    #     min_mean = min_n[-1] / np.mean(min_n)  # 最低价/均价
    #     vol = volume[-1]  # 现量
    #     return_now = close[-1] / close[0]  # 区间收益率
    #     std = np.std(np.array(close), axis=0)  # 区间标准差
    #     # 将计算出的指标添加到训练集X
    #     # features用于存放因子
    #     features = [close_mean, volume_mean, max_mean, min_mean, vol, return_now, std]
    #     x_all.append(features)
    #     features = np.array(features).reshape(1, -1)
    #     global gclf
    #     prediction = gclf.predict(features)
    #     print(end_day, prediction[0])
    #     y_all.append(prediction[0])

def main():
    context = {"sid": '600848'}
    init(context)
    on_bar(context)

if __name__ == '__main__':
    main()
