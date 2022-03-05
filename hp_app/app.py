from flask import Flask, render_template,request
import  numpy  as np
from joblib import load
import os



def load_hp_model():
  filepath='Machine-Learning/hp_app/house_price_model.xgb'
  return load(filepath)

def predict(Beds=0,Baths=0,SquareFeet=0):
  userinp=np.array([[Beds,Baths,SquareFeet]])
  model_dict=load_hp_model()

  x=model_dict.get('scaler').transform(userinp)
  p=model_dict.get('classifier').predict(x)
  return p[0]


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
 