
from flask import Flask, render_template,request
import  numpy  as np
from joblib import load
import os


app = Flask(__name__)

def load_fs_model():
  print(os.listdir())
  filepath='Machine-Learning/myapp_1/fs_ap.pkl'
  return load(filepath)

def predict(age=0,BMI=0,Insulin=0,Glucose=0):
  userinp=np.array([[age,BMI,Insulin,Glucose]])
  model_dict=load_fs_model()

  x=model_dict.get('scaler').transform(userinp)
  p=model_dict.get('classifier').predict(x)
  if p[0]==0:
    return "does not have diabetes"
  else:
    return 'have diabetes'  
  

@app.route('/',methods=['GET','POST'])
def index():
  if request.method=='POST':
      form=request.form
      age=int(form.get('age'))
      BMI=float(form.get('BMI'))
      Insulin=int(form.get('Insulin'))
      Glucose=int(form.get('Glucose'))
    
      userinp=np.array([[age,BMI,Insulin,Glucose]])
      result=predict(age,BMI,Insulin,Glucose)
      return render_template('index.html',age=age,BMI=BMI,Insulin=Insulin,Glucose=Glucose,result=result)

  return render_template('index.html')

if __name__ == '__main__':
  app.run(debug=True)
 