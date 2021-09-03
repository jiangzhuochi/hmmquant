from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./logs/rr.log", index_col=0, names=["strategy", "long"])
df.index = pd.to_datetime(df.index.values)
cumrr = df.cumsum().loc[datetime.now().strftime(r"%Y-%m-%d")]
print(cumrr)
cumrr.plot()
plt.show()
