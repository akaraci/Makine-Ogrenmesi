import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
# Eğitim ve Test Seti olarak Ayırmak
from sklearn.model_selection import train_test_split

def kategori_sayisi(y):
    cat_list=[]
    for value in y:
        if value not in cat_list:
            cat_list.append(value)
    return len(cat_list)

def hasta_sayisi_bilgi(y):
    y_non_category = [ np.argmax(t) for t in y ]
    hs,hos=0,0
    for value in y_non_category:
        if value==0:
            hos+=1
        else:
            hs+=1
    
    print ("Hasta sayisi:",hs," Hasta olmayan sayisi:",hos)
    print ("Toplam kayit sayisi:",hs+hos)
    return hos,hs

    
    
def create_X_and_y(dataset_name): 
    file_path = "./data/"+dataset_name
    print (file_path," isleniyor...")
    
    dataset=pd.read_excel(file_path, comment='#')
                       
    veriseti=dataset.values
    np.random.shuffle(veriseti)
    print (veriseti.shape)
    
            
    labels=dataset.columns
    
    X=veriseti[:,0:len(labels)-1]
    y=veriseti[:,len(labels)-1]
    
    """#Gerekli ise çıkış integer değere dönüştürülebilir
    for i,value in enumerate(y):
        y[i]=int(value)"""
       
    class_number=kategori_sayisi(y)
    print ("Categori sayisi:",class_number)

    # Hedef Değişkeni Dönüştürmek
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    y_for_dl = np_utils.to_categorical(y)
    print (y_for_dl)
     
        
    joblib.dump(X,"./data/Parkinson_X.pkl")
    joblib.dump(y_for_dl,"./data/Parkinson_y_for_dl.pkl")
    joblib.dump(y,"./data/Parkinson_y_for_rf.pkl")
    
    y=joblib.load("./data/Parkinson_y_for_dl.pkl")
    print (y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
      
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)    
    """
      
    joblib.dump(X_train,"./data/Parkinson_X_train.pkl")
    joblib.dump(X_test,"./data/Parkinson_X_test.pkl")
    joblib.dump(y_train,"./data/Parkinson_y_train.pkl")
    joblib.dump(y_test,"./data/Parkinson_y_test.pkl")
    
def oku(etiket):
    if etiket=="Train":        
        X=joblib.load("./data/Parkinson_X_train.pkl")
        y=joblib.load("./data/Parkinson_y_train.pkl")
    if etiket=="Val":        
        X=joblib.load("./data/Parkinson_X_val.pkl")
        y=joblib.load("./data/Parkinson_y_val.pkl")    
    if etiket=="Test":        
        X=joblib.load("./data/Parkinson_X_test.pkl")
        y=joblib.load("./data/Parkinson_y_test.pkl")
    
    print (etiket,":",X.shape,y.shape)
    hasta_sayisi_bilgi(y)
    
    
 
dataset_name ="Parkinson.xlsx"
create_X_and_y(dataset_name)    

oku("Train")
oku("Test")
#oku("Val")






