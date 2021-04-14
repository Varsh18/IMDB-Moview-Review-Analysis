import json
import numpy as np
import os
from sklearn.externals import joblib
from azureml.core import Model
import joblib, pickle
import scipy

def init():
    global model
    #model_path = Model.get_model_path('outputs/bhypermodel.joblib')
    #model_path = Model.get_model_path('bhypermodel.joblib')
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'bestmodel.pkl')
    model = joblib.load(model_path)

def run(data):
    try:
        data = np.array(json.loads(data))
        sparse_data = scipy.sparse.csr_matrix(data)
        result = model.predict(sparse_data)
        # You can return any data type, as long as it is JSON serializable.
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error
