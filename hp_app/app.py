from flask import Flask, render_template,request
import  numpy  as np
from joblib import load
import os
import xgboost as xgb



def load_hp_model():
  filepath='house_pricing_model.xgb'
  saved_model=xgb.Booster()
  saved_model.load_model(filepath)
  return saved_model

def predict(Beds=0,Baths=0,SquareFeet=0):
  userinp=np.array([[Beds,Baths,SquareFeet]])
  model=load_hp_model()
  out=model.predict(xgb.DMatrix(userinp))
  return out[0]
  


app = Flask(__name__)
@app.route('/',methods=['GET','POST'])
def index():
  if request.method=='POST':
      form=request.form
      Beds=int(form.get('Beds'))
      Baths=int(form.get('Baths'))
      SquareFeet=float(form.get('SquareFeet'))
      
    
      userinp=np.array([[Beds,Baths,SquareFeet]])
      result=predict(Beds,Baths,SquareFeet)
      return render_template('index.html',Beds=Beds,Baths=Baths,SquareFeet=SquareFeet,result=result)

  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
 