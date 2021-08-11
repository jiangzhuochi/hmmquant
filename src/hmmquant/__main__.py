import matplotlib.pyplot as plt
import pandas as pd
from hmmlearn import hmm

from hmmquant.utils import *

# data_df = futures_main_sina("T0")
# data_df.to_pickle("data/t0.pkl")
data_df = pd.read_pickle("data/t0.pkl")

data_df["date"] = pd.to_datetime(data_df["date"])
close = data_df[["date", "c"]].set_index("date")["c"]

logrr = get_logrr(close)


q1, q2, q3, m = logrr.quantile([0.25, 0.5, 0.75, 1]).values
q3 = pd.Series([i for i in logrr.values if i > 0]).quantile(0.8)
q1 = pd.Series([i for i in logrr.values if i <= 0]).quantile(0.2)

train = logrr[:800]
test = logrr[800:]


def make_label(x):
    tag = ""
    if x < q1:
        tag = 0
    elif x < q2:
        tag = 1
    elif x < q3:
        tag = 2
    else:
        tag = 3
    return tag


a = list(map(make_label, logrr.values))
x = list(map(make_label, train.values))
y = list(map(make_label, test.values))


states = ["牛市", "震荡", "熊市"]
n_states = len(states)

observations = ["强烈上涨", "缓慢上涨", "缓慢下跌", "快速下跌"]
n_observations = len(observations)

model = hmm.MultinomialHMM(
    n_components=n_states, n_iter=200000, tol=0.00001, init_params="e"
)
model.fit(np.array(x).reshape(-1, 1))


ob = np.array(a).reshape(-1, 1)  # 全样本
box = model.predict(ob)

ob2 = np.array(y).reshape(-1, 1)  # 测试集
box2 = model.predict(ob2)


position = 0
rr = 0
rrlist = []
for sig, (t, r) in zip(pd.Series(box).values, logrr.items()):
    if sig == 0:
        rrlist.append(position * r)
        rr += position * r
        position += 1

    elif sig == 1:
        rrlist.append(position * r)

        rr += position * r

    else:
        rrlist.append(position * r)

        rr += position * r
        position = 0

ret = pd.Series(rrlist, index=logrr.index)
# ret.cumsum().plot()

# plt.show()

print(evaluation(ret, 0.03))
