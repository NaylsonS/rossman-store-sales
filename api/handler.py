import pickle
import pandas as pd
from flask             import Flask, request, Response
from rossmann.Rossmann import Rossmann

#loading model
model = pickle.load(open('D:/Documentos/ESTUDOS/ComunidadeDS/repos/rossman-store-sales/model/model_rossman.pkl', 'rb'))

#initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])
def rossmann_predict():
    test_json = request.get_json()
    
    if test_json: #se tiver dados
        if isinstance(test_json, dict): # único exemplo
            test_raw = pd.DataFrame( test_json, index=[0])
        else: # multiplo exemplo
            test_raw = pd.DataFrame( test_json, columns=test_json[0].keys())
        
        # Intanciar a classe rossmann
        pipeline = Rossmann()
        
        #Limpeza dos dados
        df1 = pipeline.data_cleaning( test_raw )
        
        #Feature engineering
        df2 = pipeline.feature_engineering( df1 )
        
        #Preparação dos dados
        df3 = pipeline.data_preparation( df2 )
        
        #Predição
        df_response = pipeline.get_prediction( model, test_raw, df3 )
        
        return df_response

    
    else:
        return Response('{}', status=200, mimetype='application/json')
    
if __name__ == '__main__':
    app.run('0.0.0.0') 