
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import pandas as pd


datos = pd.read_csv("Example-bank-data.csv")

print(datos)
datos['y'] = datos['y'].replace({'no': 0, 'yes': 1})
y = datos['y'].astype(int)

X = datos['duration']


print(datos)

print(datos.isnull().sum())

print("------------")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)
y_train= y_train.values.reshape(-1,1)
y_test = y_test.values.reshape(-1,1)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

E1 = mean_absolute_error(y_test, y_pred)
sc1 = lr.score(X_test, y_test)
a1 = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


cuadro = pd.DataFrame(index=["modelo lr"])
cuadro["Error cuadrático medio"] = [E1]
cuadro["score"] = [sc1]
cuadro["f1"] = [f1]
cuadro["acuracy"] = [a1]
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option("expand_frame_repr", False)
print(cuadro)

cuadro.to_csv('resultados.csv', index_label='Modelo')



import numpy as np


class Perceptron:

    def __init__(self, inputs, outputs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.weights = None
        self.init_weights(0.647385, 0.37817776, 0.3316005)

    def init_weights(self, w1, w2, w3):
        self.weights = np.array([w1, w2, w3])

    def Fit(self):
        epochs, num_inputs = 0, 0
        while num_inputs < 4:
            print('-------------------- epochs {} -------------------- '.format(epochs))
            for input, output in zip(self.inputs, self.outputs):
                print('Input: {}'.format(input))
                y_generate = input @ self.weights
                print("(", self.weights[0], "*", input[0], ")+(", self.weights[1], "*", input[1], ")+(", self.weights[2], "*", input[2], ")")
                print("suma es:", y_generate)
                y_generate = 1 if y_generate >= 0 else -1
                print("y=", y_generate)
                print('error= ( yD ), - ( y )')
                print('error= ( {} ), - ( {} )'.format(output, y_generate))
                error = output-y_generate
                print('error=', error)
                if error == 0:
                    num_inputs += 1
                else:
                    for i in range(len(input)):
                        print(self.weights[i], "+(", 0.4, "*", "(", error, ")*", input[i], ")")
                        self.weights[i] += 0.4 * error * input[i]
                        print("w", i, "=", self.weights[i])
                        print()
                    self.init_weights(self.weights[0], self.weights[1], self.weights[2])
                    break
                print()
                print()
            epochs += 1







if __name__ == '__main__':
    inputs = [
        [1, 1, 1],
        [1, 1, 1],
        [1, -1, 1],
        [1, -1, -1]
    ]
    outputs = [1, 1, 1, -1]
    perceptron = Perceptron(inputs, outputs)
    perceptron.Fit()
    print(perceptron.weights)


from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

ppn = Perceptron(max_iter=10, eta0=0.1, random_state=0)
x = np.array([[1, 1, 1,
               1, 0, 0,
               1, 1, 1],

             [1, 1, 1,
              0, 1, 0,
              0, 1, 1],

             [1, 0, 1,
              1, 1, 1,
              1, 1, 1]])

y = np.array([0, 1, 2])

x_test = np.array([[1, 0, 1,
                    1, 1, 1,
                    1, 1, 1]])
ppn.fit(x, y)
y_pred = ppn.predict(x_test)
accuracy = accuracy_score(y, ppn.predict(x))
print('Accuracy:', accuracy)
print('Predictions:', y_pred)






import numpy as np

class Perceptron:
    def __init__(self, inputs, outputs, learning_rate=0.1, epochs=10):
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = np.random.rand(len(inputs[0])) # Inicializar pesos aleatorios

    def activation_function(self, x):
        return np.where(x >= 0, 1, -1) # Función de activación de unidad de paso

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights)
        prediction = self.activation_function(weighted_sum)
        return prediction

    def fit(self):
        for epoch in range(self.epochs):
            for inputs, output in zip(self.inputs, self.outputs):
                prediction = self.predict(inputs)
                error = output - prediction
                self.weights += self.learning_rate * error * inputs

       y = np.array([0, 1, 2])
       x_test = np.array([[1, 0, 1,
                    1, 1, 1,
                    1, 1, 1]])









      from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'

im = Image.open("1.png")
texto = pytesseract.image_to_string(im)
print(texto)











#backpropagation

import numpy as np

# Funciones de activación
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Clase de la red neuronal
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Inicialización de pesos
        self.weights1 = np.random.rand(self.input_size, self.hidden_size)
        self.weights2 = np.random.rand(self.hidden_size, self.output_size)

    def forward(self, X):
        # Propagación hacia adelante
        self.hidden_layer = sigmoid(np.dot(X, self.weights1))
        self.output_layer = sigmoid(np.dot(self.hidden_layer, self.weights2))
        return self.output_layer

    def backward(self, X, y, output):
        # Cálculo de errores y gradientes
        self.output_error = y - output
        self.output_delta = self.output_error * sigmoid_derivative(output)
        self.hidden_error = np.dot(self.output_delta, self.weights2.T)
        self.hidden_delta = self.hidden_error * sigmoid_derivative(self.hidden_layer)

        # Actualización de pesos
        self.weights1 += np.dot(X.T, self.hidden_delta)
        self.weights2 += np.dot(self.hidden_layer.T, self.output_delta)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)

# Ejemplo de uso
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(3, 4, 1)
nn.train(X, y, 10000)





import easyocr

reader = easyocr.Reader(['en'])
result = reader.readtext('1.png')
texto = ""
for detection in result:
    texto += detection[1] + " "
print(texto)
"""



import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
from PIL import Image,ImageEnhance
im = Image.open('5.jpg')
new_size = tuple(2*x for x in im.size)
im = im.resize(new_size, Image.ANTIALIAS)
im.save('1_resized.png')


im = Image.open("1_resized.png")
# aumentar el contraste en un 50% y brillo en 50%
contraste = ImageEnhance.Contrast(im)
im = contraste.enhance(1.5)
brillo = ImageEnhance.Brightness(im)
im = brillo.enhance(1.5)
im.save("imagen_con_contraste.jpg")
texto = pytesseract.image_to_string('imagen_con_contraste.jpg')
print(texto)

