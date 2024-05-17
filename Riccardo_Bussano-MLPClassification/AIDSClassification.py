import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, data_path, train: bool = True):
        self.trainable = train
        self.data = pd.read_csv(data_path)
        self.X = self.data.drop(columns=["infected"])  # Features
        self.y = self.data["infected"]  # Target variable

        X_columns = self.X.columns

        print("\n🧠  ----------Preprocessing----------")
        print(f"📊 Shape of the dataset: {self.data.shape}")
        print(f"📊 Shape of the features: {self.X.shape}")
        print(f"📊 Shape of the target: {self.y.shape}")

        # Dividendo il dataset in Training e Test set
        self.X_train, self.X_test, self.y_train, self.y_test = (
            train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        )

        # Standardizzazione delle features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.X_train = pd.DataFrame(self.X_train, columns=X_columns)
        self.X_test = pd.DataFrame(self.X_test, columns=X_columns)

        print("\n🧠  ----------Preprocessing----------")
        print(f"📊 Shape of the X_train: {self.X_train.shape}")
        print(f"📊 Shape of the X_test: {self.X_test.shape}")
        print(f"📊 Shape of the y_train: {self.y_train.shape}")
        print(f"📊 Shape of the y_test: {self.y_test.shape}")

        print("\n📊 Features after scaling:")
        print("🔢 X Train")
        print(self.X_train.head())
        print("🔢 X Test")
        print(self.X_test.head())

        # Inizializzazione della classe the MLPClassifier
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation="tanh",
            solver="adam",
            max_iter=1000,
            random_state=42,
            learning_rate_init=0.001,
        )

    def train(self):
        # Addestramento del modello
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate(self):
        # Predizioni del modello con il test set e il training set
        y_test_pred = self.classifier.predict(self.X_test)
        y_train_pred = self.classifier.predict(self.X_train)

        # Valutazione dell'accuratezza del modello
        # y_reali    # y_predette
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        # self.classifier.score(self.X_train, self.y_train)           # Questo è equivalente a train_accuracy

        print("\n🎯 ----------Accuracy----------")
        print("🎯 Test Accuracy:", test_accuracy)
        print("🎯 Train Accuracy:", train_accuracy)
        # print("🎯 Model Score:", self.classifier.score(self.X_train, self.y_train))

    def evaluate_emoji(self):
        y_pred = self.classifier.predict(self.X_test)
        y_pred = " ".join(
            list(map(lambda x: "💀" if x == 1 else "🔥", y_pred))
        )
        y_pred = "\n".join(
            y_pred[i : i + 20] for i in range(0, len(y_pred), 20)
        )
        print(y_pred)

    def predict(self, data):
        # Predicting new data
        scaled_data = self.scaler.transform(data)
        prediction = self.classifier.predict(scaled_data)
        return prediction

    def save_model(
        self, path="./models_weights/AIDS_Classification_model.pkl"
    ):
        # Save the model
        joblib.dump(self.classifier, path)
        print("Model saved successfully!")

    def load_model(
        self, path="./models_weights/AIDS_Classification_model.pkl"
    ):
        self.classifier: MLPClassifier = joblib.load(path)


if __name__ == "__main__":
    # Path to your CSV file
    data_path = "./datasets/AIDS_Classification_50000.csv"

    # Create an instance of the NeuralNetwork class
    nn = NeuralNetwork(data_path, train=False)

    if nn.trainable:
        # Train the neural network
        nn.train()
        # Salvo il modello
        nn.save_model()

    # Modello caricato
    nn.load_model()

    # Evaluate the model
    nn.evaluate()

    # Utilizziamo nuovi dati per fare una predzione
    # new_data = pd.DataFrame({
    #     'time': [1073],
    #     'trt': [1],
    #     'age': [37],
    #     'wtkg': [79.46339],
    #     'hemo': [0],
    #     'homo': [1],
    #     'drugs': [0],
    #     'karnof': [100],
    #     'oprior': [0],
    #     'z30': [1],
    #     'preanti': [18],
    #     'race': [1],
    #     'gender': [1],
    #     'str2': [1],
    #     'strat': [2],
    #     'symptom': [0],
    #     'treat': [0],
    #     'offtrt': [1],
    #     'cd40': [322],
    #     'cd420': [469],
    #     'cd80': [882],
    #     'cd820': [754]
    # })

    # Dati di una persona infetta persi dal dataset, per provare se funziona con dei dati inseriti manualmente
    new_data = pd.DataFrame(
        {
            "time": [1201],
            "trt": [3],
            "age": [42],
            "wtkg": [89.15934],
            "hemo": [0],
            "homo": [1],
            "drugs": [0],
            "karnof": [100],
            "oprior": [1],
            "z30": [1],
            "preanti": [513],
            "race": [0],
            "gender": [1],
            "str2": [1],
            "strat": [3],
            "symptom": [0],
            "treat": [0],
            "offtrt": [0],
            "cd40": [500],
            "cd420": [324],
            "cd80": [775],
            "cd820": [1019],
        }
    )

    prediction = nn.predict(new_data)
    if prediction == 1:
        print(f"💀 Predizione classe: {prediction}")
    else:
        print(f"🔥 Predizione classe: {prediction}")
