import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

class DiseasePredictor:
    def __init__(self, data_path="training_data.csv"):
        self.data_path = data_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.symptom_columns = []

    def load_and_train(self):
        df = pd.read_csv(self.data_path)
        df = df.drop(columns=["Unnamed: 133"], errors="ignore")
        X = df.drop(columns=["prognosis"])
        y = self.label_encoder.fit_transform(df["prognosis"])
        self.symptom_columns = list(X.columns)
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.model.fit(X, y)

    def predict(self, user_symptoms: dict):
        input_vector = [int(user_symptoms.get(symptom, 0)) for symptom in self.symptom_columns]
        prediction = self.model.predict([input_vector])[0]
        return self.label_encoder.inverse_transform([prediction])[0]

