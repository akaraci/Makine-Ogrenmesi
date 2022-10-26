from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

veriseti = pd.read_csv('data/linear_model.csv')
print(type(veriseti))
print(veriseti.head()) #veri seti başlığını ve ilk 5 satırı göster

print("\n---null veri seti sayısı----")
print(veriseti.isnull().sum()) 

#---Boş verileri ortalama ile doldur
veriseti.y.fillna(value=veriseti.y.mean(), inplace=True)

print("\n----veri setidescribe---")
print(veriseti.describe())

scaler = MinMaxScaler()
scaler.fit(veriseti)
veriseti = pd.DataFrame(scaler.transform(veriseti)) #minmax normalizasyon uygulanıyor

veriseti.columns=["x","y"] #dönüşüm sırasında header 0 ve 1' dönüştüğü için header yeniden düzenleniyor
print(veriseti.describe())

x=veriseti.x
y=veriseti.y
#iki boyutlu numpy array'e dönüştürülüyor ilk boyuttaki eleman sayısına 100'de yazılabilir. -1 yazılırsa kendi hesaplar
x=x.values.reshape(-1,1) 
#iki boyuta dönüştürülüyor
y=y.values.reshape(-1,1) 

#x ve y arasındaki ilişkiye plotter olarak bakalım
plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('Y')
plt.title('x ve y arasındaki ilişki')
plt.show()

lineer_regresyon = LinearRegression()
lineer_regresyon.fit(x,y)

#---model bilgileri yazdırılıyor
print("\nModel Information")
print(lineer_regresyon.intercept_) #parameter m
print(lineer_regresyon.coef_)      #parameter b
print("Elde edilen regresyon modeli: Y={}+{}X".format(lineer_regresyon.intercept_,lineer_regresyon.coef_[0]))


# model test
print("\nTest Model")
y_predicted = lineer_regresyon.predict(x)
print("R2=",r2_score(y,y_predicted))
print("MSE=",mean_squared_error(y, y_predicted))


plt.scatter(x, y,color='red') #ham verinin dağılımı kırmızı noktalar.
plt.plot(x, y_predicted, color='blue',label='regresyon grafiği') # modelin tahmin ettiği mavi çizgi
plt.xlabel('x')
plt.ylabel('Y')
plt.title('X y regresyon analiz')
plt.show()
