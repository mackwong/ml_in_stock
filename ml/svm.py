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
        if recent_data.shape[0] == index+1:
            break
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

    x_train = x_all[: -1]
    y_train = y_all[: -1]
    # 训练SVM
    clf = svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False,
                          tol=0.001, cache_size=200, verbose=False, max_iter=-1,
                          decision_function_shape='ovr', random_state=None)
    clf.fit(x_train, y_train)
    context["clf"] = clf
    print('训练完成!')


def on_bar(context):
    sid = context['sid']
    recent_data = ts.get_hist_data(sid, '2020-06-01', '2020-08-30')
    recent_data = recent_data.iloc[::-1]
    x_all = []

    index = 0
    for date, data in recent_data.iterrows():
        if recent_data.shape[0] == index+1:
            break
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
        prediction =  context["clf"].predict(features)
        print(date, prediction[0])
        index = index + 1

def main():
    context = {"sid": '600848'}
    init(context)
    on_bar(context)

if __name__ == '__main__':
    main()
