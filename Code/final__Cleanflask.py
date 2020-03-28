import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import model_from_json

model_path = 'C:\\Users\\ugur_\\Desktop\\MSCBD\\Thesis\\'

json_file =open(model_path+"model.json","r")
#json format
model = json_file.read()
json_file.close()
#converted model
model = tf.keras.models.model_from_json(model)
model.load_weights(model_path+"weight.h5")
model._make_predict_function()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


model1 = tf.keras.models.load_model(model_path+'model.h5')
X_test = np.load(model_path+'xtest.npy')
y_test = np.load(model_path+'ytest.npy')

# evaluate the model
score,acc = model.evaluate(X_test,y_test,batch_size = 100,verbose=2)
print("Test accuracy: ",acc)

#prediction
predict_y = model.predict(X_test)
predict_y= np.around(predict_y,decimals = 5)


#table variable
c_names = ['inning', 'p_score', 'b_score', 'outs', 'pitch_num', 'on_1b', 'on_2b',
       'on_3b', 'spin_rate', 'spin_dir', 'start_speed', 'end_speed',
       'total_CH', 'total_CU', 'total_EP', 'total_FC', 'total_FF', 'total_FS',
       'total_KC', 'total_KN', 'total_PO', 'total_SC', 'total_SI', 'total_SL',
       'Game_pt_pCount', 'Game_pt_t_s', 'df_CH', 'df_CU', 'df_EP', 'df_FC',
       'df_FF', 'df_FS', 'df_KC', 'df_KN', 'df_PO', 'df_SC', 'df_SI', 'df_SL']
table_X = pd.DataFrame(X_test, columns=c_names)
##web
from flask import Flask,redirect,url_for, render_template,request
app = Flask (__name__)


@app.route("/")
def home():
    return render_template("index.html")
@app.route("/Rule")
def rule():
    return render_template("rules.html")
    
@app.route("/Predict",methods = ["POST","GET"])
def predict():
    p_types =["CH","CU","EP","FC","FF","FS","KC","KN","PO","SC","SI","SL"]
    if request.method == "POST":
        if request.form["submit"] == "1":
            if request.form['index'] != "":
                index =int(request.form["index"])
            else:
                index = 0
                
            return render_template("result.html",result=predict_y[index],origin = y_test[index])
        
        else:
            
            Inning = request.form["Inning"]           
            p_score = request.form["p_score"]            
            b_score = request.form["b_score"]
            outs = request.form["outs"]           
            p_num = request.form["p_num"]            
            base_list = [0,0,0]            
            base = request.form.getlist("base")           
            base = list(map(int,base))
            for i in base:
                base_list[i-1] =1             
            
            start_speed = request.form["start_speed"]            
            end_speed = request.form["end_speed"]            
            spin_rate = request.form["spin_rate"]
            spin_dir = request.form["spin_dir"]            
                   
            Game_pt_pCount = request.form["Game_pt_pCount"]
            Game_pt_t_s= request.form["Game_pt_t_s"]
            pitch_type = request.form["pt"]
            
            #total_types
            t_types = []
            for i in p_types:
                t_types.append(request.form[i])
            
            lst= [Inning,p_score,b_score,outs,p_num]
            lst.extend(base_list)
            lst.extend([spin_rate,spin_dir,start_speed,
            end_speed])
            lst.extend(t_types)
            lst.extend([Game_pt_pCount,Game_pt_t_s])
            p_types = [1 if i == pitch_type  else 0 for i in p_types]
            lst.extend(p_types)
            
            lst = ['0' if i == '' else i for i in lst]
            
            lst = list(map(int, lst))
            
            predict_given = model.predict([lst])
            predict_given= np.around(predict_given,decimals = 5)
            
            return render_template("result.html",result = predict_given[0])
       
    else:        
        return render_template("predict.html",pt= p_types,tables= table_X.iloc[:10,].to_html(border = None,classes='table table-striped',header="true"))


if __name__ == "__main__":
    app.run()
    