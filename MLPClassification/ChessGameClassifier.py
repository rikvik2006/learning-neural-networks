import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

class ChessGameClassifier:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None

    def load_data(self):
        self.data = pd.read_csv(self.dataset_path)

    def preprocess_data(self):
        # Rimuovi le righe con vittoria "draw"
        self.data = self.data[self.data['winner'] != 'draw']
        # Dividi il dataset in features e target
        self.X = self.data.drop(['id', 'rated', 'created_at', 'last_move_at', 'victory_status', 'winner'], axis=1)
        self.y = self.data['winner']
        # Codifica le label "black" e "white" come 0 e 1
        self.y = self.y.map({'black': 0, 'white': 1})
        
    def split_data(self):
        # Dividi il dataset in set di addestramento e set di test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def scale_features(self):
        # Standardizza le features
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def build_model(self):
        # Crea un'istanza di MLPClassifier
        self.model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

    def train_model(self):
        # Addestra il modello
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        # Valuta le prestazioni del modello
        accuracy = self.model.score(self.X_test, self.y_test)
        print("Accuracy:", accuracy)

if __name__ == "__main__":
    dataset_path = "./datasets/games.csv"
    classifier = ChessGameClassifier(dataset_path)
    classifier.load_data()
    classifier.preprocess_data()
    print(classifier.data)
    # classifier.split_data()
    # classifier.scale_features()
    # classifier.build_model()
    # classifier.train_model()
    # classifier.evaluate_model()
