import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
df=pd.read_csv('data/loan_data.csv')
#print(df.head)
x=df.iloc[:,0:4] #feature
y=df.iloc[:,-1] #target variable
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=DecisionTreeClassifier(random_state=42,max_depth=4)
model.fit(X_train,Y_train)
accuracy=model.score(X_test,Y_test)
print(f"Model Accuracy: {accuracy}")
#save the model to file
joblib.dump(model,'model/loan_model.pkl')
print("model saved ")