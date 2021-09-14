from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./logs/rr.log", index_col=0, names=["strategy", "long"])
df.index = pd.to_datetime(df.index.values)
cumrr = df.loc[datetime.now().strftime(r"%Y-%m-%d")].cumsum()
print(cumrr)
cumrr.plot()
plt.show()



index_date = df.index
fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(16, 9))
index_x = list(range(len(index_date)))
pretty_x = index_x[:: len(index_x) // 20]
pretty_date = index_date[:: len(index_date) // 20]
ax.set_xticks(pretty_x)
ax.set_xticklabels(pretty_date, rotation=30)
lines = ax.plot(index_x, df.cumsum().values, label=df.columns)
ax.legend(lines, df.columns)
plt.show()
