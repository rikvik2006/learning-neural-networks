import pandas as pd
from typing import Literal, List
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TitanicSurvivalClassifier:
    def __init__(self, data_path):
        self.dataset = pd.read_csv(data_path) 
        self.data = self.dataset.copy()
        # Inizializiamo il LabelEncoder ovvero un algorimo che ci permette di trasformare le stringhe in numeri
        self.label_encoder = LabelEncoder()
        # Inizializiamo l'Imputer in modo che inserisca i valori più fequenti delle celle delle colonne dove mancano i dati
        self.imputer = SimpleImputer(strategy='most_frequent')
        # Inizializiamo lo StandardScaler che ci permette di standardizzare i dati
        self.scaler = StandardScaler() 
        
        self.model = None
        self.X = None
        self.X_test = None
        self.y = None

    def __fit_transform_label_encoder(self, fit_columns: List[pd.Series], encode_columns: List[pd.Series]) -> List[pd.Series]:
        encoded_columns = []
        for i, fit_column in enumerate(fit_columns):
            self.label_encoder.fit(fit_column)
            encoded_column = self.label_encoder.transform(encode_columns[i])
            print(f"🔢 Encoded {fit_column.name}: {encoded_column[:5]} mapped to {encode_columns[i].values[:5]}")
            
            encoded_columns.append(encoded_column)

        return encoded_columns

    def preprocess_data(self):
        # Rimuoviamo la colonna Cabin dato che è per il 78% null
        self.data.drop(columns=['Cabin'], inplace=True)

        # Riempiamo i valori mancanti nella colonna 'Age', e 'Fare'
        # Stiamo utilizzando un imputer di tipo SimpleImputer, questa classe 
        self.data['Age'] = self.imputer.fit_transform(self.data[['Age']])
        self.data['Fare'] = self.imputer.fit_transform(self.data[['Fare']])

        # Codifica le variabili categoriche
        # self.data['Sex'] = self.label_encoder.fit_transform(self.data['Sex'])
        # self.data['Embarked'] = self.label_encoder.fit_transform(self.data['Embarked'])
        encoded_sex, encoded_embarked = self.__fit_transform_label_encoder(fit_columns=[self.data['Sex'], self.data['Embarked']], encode_columns=[self.data['Sex'], self.data['Embarked']])
        self.data['Sex'] = encoded_sex
        self.data['Embarked'] = encoded_embarked

        # Seleziona le feature e il target
        X_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        self.X = self.data[X_columns]

        # Standardiziamo i dati
        # self.X = self.scaler.fit_transform(self.X)
        # self.X = pd.DataFrame(self.X, columns=X_columns)
        self.y = self.data['Survived']
        print("\n🧠  ----------Preprocessing----------")        
        print(self.X)

    def train_model(self):
        # Dividi il dataset in training e test set
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_test = X_test
        
        print("\n🔢 ----------X and Y train datasets----------")
        print(X_train)
        print(y_train)
        print("\n🔢 ----------X and Y test datasets----------")
        print(X_test)
        print(y_test.head(100))

        # Inizializza il modello MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)

        # Addestra il modello
        self.model.fit(X_train, y_train)

        # Valuta l'accuratezza del modello
        print("\n🎯 ----------Accuracy----------")
        train_accuracy = accuracy_score(y_train, self.model.predict(X_train))
        test_accuracy = accuracy_score(y_test, self.model.predict(X_test))
        model_score = self.model.score(X_train, y_train)
        print(f"Training Accuracy: {train_accuracy}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Model Score: {model_score}")


    def predict_custom_data(self, pclass: Literal[1, 2, 3], sex: Literal["male", "female"], age: int, sibsp: int, parch: int, fare: float, embarked: Literal["C", "Q", "S"]):
        """Effettua una previsione per nuovi dati di passeggeri
        Parameters
        ----------
        passenger_data : pd.DataFrame
            I dati del passeggero per il quale si desidera effettuare una previsione
            Formato: ['Pclass (Classe stanza: 1, 2, 3)', 'Sex: (Male, Female)', 'Age: int', 'SibSp: (numero fratelli: int)', 'Parch: (Rispetto: int)', 'Fare: (Tariffa: float) ', 'Embarked: (Porto di imbarco: C, Q, S)']
        """
        # Effettua una previsione per nuovi dati di passeggeri
        print("\n🔢 ----------Custom Prediction Data----------")
        X_columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        passenger_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]], columns=X_columns)

        # Dato che abbiamo richiesto i  dati in formato stringa, dobbiamo codificarli in numeri e anche standardizzarli
        # self.label_encoder.fit(self.data['Sex'])
        # passenger_data['Sex'] = self.label_encoder.transform(passenger_data['Sex'])
        # self.label_encoder.fit(self.data['Embarked'])
        # passenger_data['Embarked'] = self.label_encoder.transform(passenger_data['Embarked'])

        encoded_sex, encoded_embarked = self.__fit_transform_label_encoder(fit_columns=[self.dataset['Sex'], self.dataset['Embarked']], encode_columns=[passenger_data['Sex'], passenger_data['Embarked']])
        passenger_data['Sex'] = encoded_sex
        passenger_data['Embarked'] = encoded_embarked
        print("Passenger Data: ", passenger_data)

        # Standardiziamo i dati
        # passenger_data = self.scaler.fit_transform(passenger_data)
        # passenger_data = pd.DataFrame(passenger_data, columns=X_columns)


        return self.predict(passenger_data)
    
    def predict_test_data(self):
        predictions = list(self.model.predict(self.X_test))
        n_people = len(predictions)
        n_dead = len(list(filter(lambda x: x == 0, predictions)))
        n_survived = len(list(filter(lambda x: x == 1, predictions)))
        predictions = " ".join(list(map(lambda x: "🔥" if x == 1 else "💀", predictions)))
        predictions = "\n".join(predictions[i:i+20] for i in range(0, len(predictions), 20))
        print(predictions)
        print(f"People: {n_people}, Dead: {n_dead}, Survived: {n_survived}")
        
    
    def predict(self, passenger_data):
        return self.model.predict(passenger_data)

if __name__ == "__main__":
    # Percorso del file CSV
    data_path = "./datasets/titanic.csv"

    # Crea un'istanza del classificatore TitanicSurvivalClassifier
    classifier = TitanicSurvivalClassifier(data_path)

    # Preelabora i dati
    classifier.preprocess_data()

    # Addestra il modello
    classifier.train_model()

    # Effettua una previsione per i dati di test
    print("\n🪄 ----------Test Data Prediction----------")
    classifier.predict_test_data()

    # Esempio di previsione per un nuovo passeggero
    # prediction = classifier.predict_custom_data(pclass=3, sex="female", age=47, sibsp=1, parch=0, fare=7, embarked="S")
    prediction = classifier.predict_custom_data(pclass=3, sex="male", age=25, sibsp=0, parch=0, fare=7.2292, embarked="S")

    print("\n🪄 ----------Prediction----------")
    if prediction  == 1:
        print("🔥 Survived!")
    else: 
        print("💀 Dead")
