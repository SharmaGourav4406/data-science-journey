import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.metrics import r2_score
df = pd.read_csv(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\real_projects\Housing.csv')
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum()) 
#now I am going to check if there are any outliers in the dataset or not
df['price'].plot(kind='box')
plt.show()
# Now I will remove the outliers by using Upper fence and lower fence and IQR
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 1.5*IQR
lower_fence = Q1-1.5*IQR
df = df[(df['price']>=lower_fence)&(df['price']<=upper_fence)]
plt.scatter(df['area'],df['price'])
plt.xlabel('area')
plt.ylabel('Price')
plt.show()
#now check the relationship between the price and the area of the neighbourhood houses
df['area_per_bedroom'] = df['area']/(df['bedrooms']+1)
features = ['area','bedrooms','bathrooms','stories','parking','area_per_bedroom']
x = df[features]
y=df['price']
#going to split the given data into training and testing data
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.20,random_state=42)
#now going to train the models based upon the splitted data
model1 = LinearRegression()
model1.fit(x_train,y_train)
prediction_1 = model1.predict(x_test)
model2 = DecisionTreeRegressor()
model2.fit(x_train,y_train)
prediction_2 = model2.predict(x_test)
model3 = RandomForestRegressor()
model3.fit(x_train,y_train)
prediction_3=model3.predict(x_test)
#going to check the mae and mse and rmse of all three models
print('the mse of linear regression is :',np.mean((y_test-prediction_1)**2))
print('the mae of linear regression is :',np.mean(abs(y_test-prediction_1)))
print('rmse of linear regression is :',np.sqrt(np.mean((y_test-prediction_1)**2)))
print('the mse of decision tree regression is :',np.mean((y_test-prediction_2)**2))
print('the mae of decision tree regression is :',np.mean(abs(y_test-prediction_2)))
print('rmse of decision tree regression is :',np.sqrt(np.mean((y_test-prediction_2)**2)))
print('the mse of random forest regression is :',np.mean((y_test-prediction_3)**2))
print('the mae of random forest regression is :',np.mean(abs(y_test-prediction_3)))
print('rmse of random forest regression is :',np.sqrt(np.mean((y_test-prediction_3)**2)))
#we can also use mean_absolute_percentage_error to check the accuracy of the model or mean_squared_error
#now I am going to save the best performing model which gave the least rmse and mae and mse
#in this case, the best performing model is linear regression as it has the least rmse and mae and mse
pickle.dump(model1,open('house_price_prediction_model.pkl','wb'))
#here I have saved the model in a pickle file and now I can use this file to make predictions on new data without retraining the model again.
#wb stands for write binary mode, which is used to write the model in a binary format.
print('R2 Score LR:', r2_score(y_test, prediction_1))
print('R2 Score DT:', r2_score(y_test, prediction_2))
print('R2 Score RF:', r2_score(y_test, prediction_3))
new_house = [[5000,3,2,2,1,5000/(3+1)]]
prediction = model1.predict(new_house)
print(f'Predicted Price ${prediction[0]:,.2f}')
