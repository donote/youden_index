"""
demo for youden index
"""

from youden import youden_index
import numpy as np


def demo():
    # 生成随机标签和预测概率
    np.random.seed(42)
    y_true = np.random.randint(0, 2, size=50)
    y_score = np.random.rand(50)

    df, mj_val, mf1_val, auc = youden_index(y_true, y_score, pos_label=1, step=5)
    print('-' * 100)
    print(df)
    print('-' * 100)
    print('max_youden_value: {:.3f}'.format(mj_val))
    print('max_F1_value: {:.3f}'.format(mf1_val))
    print('AUC: {:.3f}'.format(auc))

    mj_idx = df['YoudenIdx'].idxmax()
    thr_val = df['Thr'][mj_idx]
    print('The max youden index correspond to threshold is {:.3f}'.format(thr_val))

    mf1_idx = df['F1'].idxmax()
    thr_val = df['Thr'][mf1_idx]
    print('The max F1 correspond to threshold is {:.3f}'.format(thr_val))


if __name__ == '__main__':
    demo()
