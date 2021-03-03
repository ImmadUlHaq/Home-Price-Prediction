
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')

#matplotlib inline
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

new_df = df.drop('price',axis='columns')


price = df.price

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)

reg.predict([[3300]])

#reg.coef_

#reg.intercept_

#3300*135.78767123 + 180616.43835616432

reg.predict([[5000]])

area_df = pd.read_csv("areas.csv")
area_df.head(3)

p = reg.predict(area_df)
#p

area_df['prices']=p
#area_df


area_df.to_csv("prediction.csv")