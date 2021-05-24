import numpy
import matplotlib.pyplot as plt


def estimate_b0_b1 (x, y):
    # realizar un conteo de nuestros datos
    n = np.size(x)
    # obtener los promedios de X y de Y
    m_x, m_y = np.mean(x), np.mean(y)
    # calcular sumatoria de XX y XY
    sumatoria_xy = np.sum((x-m_x)*(x-m_y))
    sumatoria_xx = np.sum(x*(x-m_x))

    # coeficientes de regresion
    b_1 = sumatoria_xy / sumatoria_xx
    b_0 = m_y - b_1*m_x

    return (b_0, b_1)
  
  
  # Funcion de graficado
def plot_regression(x, y, b):
    plt.scatter(x, y, color = "g", marker = "o", s =30)  

    y_pred = b[0] + b[1]*x
    plt.plot(x, y_pred, color = "b") 

    # etiqueta
    plt.xlabel("x-Independiente")
    plt.ylabel("y-Dependiente")
    
    
    # codigo main
def main():
    # dataset
    x = np.array([1,2,3,4,5])
    y = np.array([2,3,6,5,6])
    #  obtenemos b1 y b2
    b = estimate_b0_b1(x, y)
    print("Los valores b0 = {}, b1= {}".format(b[0], b[1]))
    # graficamos nuestra linea de regresion
    plot_regression(x, y, b)

if __name__ == "__main__":
    main()

  
