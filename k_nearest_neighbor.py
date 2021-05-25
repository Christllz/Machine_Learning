#importar librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix




if __name__ == "__main__" :
    datos = pd.read_csv("C:/Users/Pc/Downloads/Social_Network_Ads.csv")
    
    # print(datos)
    # print(datos.shape)
    # print(datos.describe())
    
    x = datos[[ "Age", "EstimatedSalary"]].values
    y = datos["Purchased"]
    # print(datos.groupby("Purchased"). size()) 

    # print(x.shape); print(y.shape)

    #datos x_train x_test y_train y_test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state= 4)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)

    clasf = KNeighborsClassifier(n_neighbors= 3, metric= "minkowski", p= 3)
    clasf.fit(x_train, y_train)
    y_pred = clasf.predict(x_test)

    matrix_c = confusion_matrix(y_test, y_pred)
    print(matrix_c)

    x_set, y_set = x_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, clasf.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('blue', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title('K-NN (Datos de Prueba)')
    plt.xlabel('Edad')
    plt.ylabel('Salario Estimado')
    plt.legend()
    plt.show()





print("Fin")