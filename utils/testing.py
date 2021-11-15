import pandas as pd
df = pd.DataFrame(pd.date_range(start="2019-09-01", end="2019-09-30", freq='D', name='ds'))
df["y"] = range(1,31)
df["add1"] = range(101,131)
df["add2"] = range(201,231)

df_train = df.loc[df["ds"]<"2019-09-21"]
df_test  = df.loc[df["ds"]>="2019-09-21"]

from fbprophet import Prophet
m = Prophet()
m.add_regressor('add1')
m.add_regressor('add2')
m.fit(df_train)
forecast = m.predict(df_test.drop(columns="y"))
print(forecast)
print(df_test.drop(columns="y"))