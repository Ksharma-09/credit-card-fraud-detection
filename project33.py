import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
df=pd.read_csv("creditcard.csv")
print(df["Class"]==1)
fraud=df[df["Class"]==1]
normal=df[df["Class"]==0].sample(len(fraud),random_state=42)
df=pd.concat([fraud,normal]).sample(frac=1)
print("Data loaded")
x=df.drop("Class",axis=1)
y=df["Class"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
print("Data Split Done")
model=RandomForestClassifier()
model.fit(x_train,y_train)
print("model Training Complete")
accuracy=model.score(x_test,y_test)
print("Accuracy:",accuracy)
joblib.dump(model,"model.pkl")
print("Model saved successfully")
