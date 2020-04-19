from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator
import xgboost as xgb

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg =  xgb.XGBRegressor(silent=True)

    def fit(self, X, y):
        self.reg.fit(X, y)
    
    def get_booster(self):
        x = self.reg.get_booster().get_fscore()
        return x

    def plot_importance(self):
        return xgb.plot_importance(self.reg)


    def predict(self, X):
        return self.reg.predict(X)
