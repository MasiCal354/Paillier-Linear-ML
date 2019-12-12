import numpy as np
from sklearn.utils.validation import check_is_fitted
from phe import paillier
from tqdm import tqdm
from sklearn.metrics import r2_score, accuracy_score

class PaillierLM:
    def __init__(self, model, key_size=2048):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        self.model = model

    def _encrypt(self, value):
        return self.public_key.encrypt(value)    
    
    def encrypt_data(self, data):
        v_encrypt = np.vectorize(self._encrypt)
        enc_data = v_encrypt(np.array(data))
        return np.array(enc_data)
    
    def _decrypt(self, enc_value):
        return self.private_key.decrypt(enc_value)
    
    def decrypt_data(self, enc_data):
        v_decrypt = np.vectorize(self._decrypt)
        data = v_decrypt(np.array(enc_data))
        return np.array(data)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self._encrypt_intercept()
        return self
    
    def _encrypt_intercept(self):
        check_is_fitted(self.model)
        self.enc_intercept_ = self._encrypt(self.model.intercept_[0])
        return self
    
    def predict(self, enc_X):
        check_is_fitted(self.model)
        
        enc_prediction = np.dot(np.array(enc_X), np.array(self.model.coef_)) + self.enc_intercept_
        return enc_prediction

class PaillierRegressor(PaillierLM):    
    def score(self, enc_X, y):
        enc_y_pred = self.predict(enc_X)
        y_pred = self.decrypt_data(enc_y_pred)
        return r2_score(y, y_pred)
    
class PaillierClassifier(PaillierLM):
    def score(self, enc_X, y):
        enc_y_pred = self.predict(enc_X)
        y_pred = self.decrypt_data(enc_y_pred)
        
        if len(y_pred.shape) == 1:
            indices = (y_pred > 0).astype(np.int)
        else:
            indices = y_pred.argmax(axis=1)
        
        y_class = self.model.classes_[indices]
        return acuracy_score(y, y_class)