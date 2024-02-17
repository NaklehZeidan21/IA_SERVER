from flask import Flask, request, jsonify
from xml.sax.handler import feature_namespace_prefixes
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class C45Classifier:
    def __init__(self):
        self.tree = None
        self.diagnostic_path = None

    def fit(self, X, y, feature_names):
        self.tree = DecisionTreeClassifier(criterion='entropy')
        self.tree.fit(X, y)

    def predict(self, instance, verbose=True):
        instance_array = np.array([list(instance.values())], dtype=float)
        predicted_class = int(self.tree.predict(instance_array)[0])

        if verbose:
            print("Camino elegido:")
            path = [(feature_namespace_prefixes[i], instance_array[0][i]) for i in range(len(instance_array[0]))]
            path.append(("Clase", predicted_class))
            for node, value in path:
                print(f"{node}: {value}")

        self.diagnostic_path = path

        return predicted_class

@app.route('/diagnose', methods=['POST'])
def diagnose():
    c45_classifier = app.config['c45_classifier']
    data = request.json
    try:
        predicted_class = c45_classifier.predict(data)
        resultado_diagnostico = "Potable" if predicted_class == 1 else "No Potable"
        return jsonify({'resultado': resultado_diagnostico}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    archivo_datos = "water_potability_vistaMinable.csv"
    data = pd.read_csv(archivo_datos)
    atributos = data.columns[:-1]
    etiquetas = data["Potabilidad"]
    X_train, _, y_train, _ = train_test_split(data[atributos], etiquetas, test_size=0.2, random_state=42)

    c45_classifier = C45Classifier()
    c45_classifier.fit(X_train, y_train, atributos)

    app.config['c45_classifier'] = c45_classifier
    app.run(debug=True)
