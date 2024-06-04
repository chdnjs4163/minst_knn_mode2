import joblib
class MinstModel():
    def __init__(self):
        self.knn_minst_model = joblib.load('MinstKnn.pkl')
    def output(self,pic):
        pic_flat = pic.reshape(1, -1).astype('float32') / 255.0
        return self.knn_minst_model.predict(pic_flat)
