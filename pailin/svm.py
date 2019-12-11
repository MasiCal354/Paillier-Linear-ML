import numpy as np
from sklearn.svm import LinearSVR
from sklearn.utils.validation import check_is_fitted
from phe import paillier
from tqdm import tqdm
from sklearn.metrics import r2_score

class PaillierLinearSVR:
    def __init__(self, key_size=2048, model=LinearSVR()):
        self.public_key, self.private_key = paillier.generate_paillier_keypair(n_length=key_size)
        self.zero_ = self._encrypt(0)
        self.one_ = self._encrypt(1)
        self.model = model
    
    def _identity_function(self, value):
        return value
    
    def fit(self, X, y):
        self.model.fit(X, y)
        self._encrypt_intercept()
        return self
    
    def _encrypt(self, value):
        return self.public_key.encrypt(value)
    
    def encrypt_data(self, X, verbose=0):
        if verbose == 0:
            verboser = self._identity_function
        else:
            verboser = tqdm
        enc_X = [[self._encrypt(j) for j in i] for i in verboser(X)]
        return np.array(enc_X)
    
    def _encrypt_intercept(self):
        check_is_fitted(self.model)
        self.enc_intercept_ = self._encrypt(self.model.intercept_[0])
        return self
    
    def _decrypt(self, enc_value):
        return self.private_key.decrypt(enc_value)
    
    def _decrypt_prediction(self, enc_prediction):
        prediction = [self._decrypt(enc_pred) for enc_pred in enc_prediction]
        return np.array(prediction)
        
    def predict(self, enc_X, verbose=0):
        check_is_fitted(self.model)
        if verbose == 0:
            verboser = self._identity_function
        else:
            verboser = tqdm
        enc_prediction = list()
        for i in verboser(range(len(enc_X))):
            enc_pred = self.zero_
            for j in range(len(enc_X[i])):
                enc_pred = enc_pred + enc_X[i][j] * self.model.coef_[j]
            enc_pred = enc_pred + self.enc_intercept_
            enc_prediction.append(enc_pred)
        return np.array(enc_prediction)
    
    def score(self, enc_X, y):
        enc_y_pred = self.predict(enc_X)
        y_pred = self._decrypt_prediction(enc_y_pred)
        return r2_score(y, y_pred)