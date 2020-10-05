# dbMgr.py


######################### Import Library #########################

# Connection Data Base Library
import pymysql

# Data Analyze Library
import pandas as pd 
import numpy as np

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.externals import joblib

##################################################################

''' Data Base 
        * 2019-08-07 
        * Data Collect & Insert Data Base 
        * Data Base Infomation : 
                host : localhost
                port : 3306
                user : root
                password : 1253
                database : dp_1
                charset : utf 8

        * Data Base Table create SQL : 
                CREATE TABLE table1 (
                        name      VARCHAR(10)  NOT NULL ,
                        gender    VARCHAR(3)   NULL     ,
                        age       INTEGER(3)   NULL     ,
                        age_group VARCHAR(5)   NULL     ,
                        weight    FLOAT(10,10) NULL     ,
                        oxy       FLOAT(10,10) NULL     ,
                        runtime   FLOAT(10,10) NULL     ,
                        runpulse  FLOAT(10,10) NULL     ,
                        rstpulse  FLOAT(10,10) NULL     ,
                        maxpulse  FLOAT(10,10) NULL     
                );

                ALTER TABLE table1
                        ADD CONSTRAINT PK_TABLE PRIMARY KEY (name);
'''

# Connection Data Base 
def getConnection():
    conn = pymysql.connect(host='localhost',port=3306 , user='root', \
                        password='1253', db='dp_1', charset='utf8', autocommit=True, \
                                cursorclass=pymysql.cursors.DictCursor)
    return conn


# Insert Customer Data
def insert_customer(data):
    affected = None
    try: 
        conn = getConnection()
        cursor = conn.cursor()
        sql = "INSERT INTO table1(name,gender,age,age_group,weight,oxy,runtime,runpulse,rstpulse,maxpulse) VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"
        affected = cursor.execute(sql,data)
        cursor.close()
    except Exception as e:
            print("========DATABASE INSERT ERROR========")
    finally:
            conn.close()
    return affected 


''' Data Analysis  
    * 2019-08-06 
    * Define Function & Analyze Data 
    * Methods : Lasso Rregression 
    * model_data Infomation
        - Rows : 31
        - Columns : 'NAME', 'GENDER', 'AGE', 'AGEGROUP', 'WEIGHT', 'OXY', 'RUNTIME','RUNPULSE', 'RSTPULSE', 'MAXPULSE'
        - Y Target : OXY 
        - X Values : Other Numeric Columns 

'''

# Data Set Learning & Modeling
def load_model_Lasso():
    result = None
    try:
        # Load Train Data 
        print("=========Modeling========")
        df1 = pd.read_csv('model_data.csv',encoding='cp949')
        print("학습 데이터 row / column 수 : ",df1.shape)

        # Data Preprocessing
        df1['WEIGHT'] = df1['WEIGHT'].fillna(df1['WEIGHT'].mean())
        df1 = df1.dropna()

        # Select Target
        X = df1[['AGE', 'WEIGHT', 'RUNTIME', 'RUNPULSE', 'RSTPULSE', 'MAXPULSE']]
        Y = df1[['OXY']]
        print("설명 변수 데이터 row/column 수 : ",X.shape)
        print("목표 변수 데이터 row/column 수 : ",Y.shape)

        print("=========Split Data========")
        # Split Data Set
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1234)
        print("Train X 데이터 row/column 수 :  ",X_train.shape)
        print("Test X 데이터 row/column 수 :  ",X_test.shape)
        print("Train Y 데이터 row/column 수 :  ",Y_train.shape)
        print("Test Y 데이터 row/column 수 :  ",Y_test.shape)

        # Fitting Model
        model = Lasso()
        result = model.fit(X_train,Y_train)

    except Exception as e :
        print("========MODELING ERROR========")    
    finally:
        pass
    return result


# Model Performance 
def model_Score_Lasso():
    result = None
    try :
        # Load Train Data 
        print("=========MODEL SCORE========")
        df1 = pd.read_csv('model_data.csv',encoding='cp949')
        
        # Data Preprocessing
        df1['WEIGHT'] = df1['WEIGHT'].fillna(df1['WEIGHT'].mean())
        df1 = df1.dropna()

        # Select Target
        X = df1[['AGE', 'WEIGHT', 'RUNTIME', 'RUNPULSE', 'RSTPULSE', 'MAXPULSE']]
        Y = df1[['OXY']]

        # Split Data Set
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=1234)

        # Fitting Model
        model = Lasso()
        result = model.fit(X_train,Y_train)

        # Model Test
        Y_pred = model.predict(X_train)
        Y_pred_test = model.predict(X_test)

        # TRAIN SCORE
        #Score
        score_train = model.score(X_train, Y_train).round(2)
        score_test = model.score(X_test, Y_test).round(2)
        #MAE
        mae_score_train = metrics.mean_absolute_error(Y_train, Y_pred)
        mae_score_test = metrics.mean_absolute_error(Y_test, Y_test)
        #MSE
        mse_score_train = metrics.mean_squared_error(Y_train, Y_pred)
        mse_score_test = metrics.mean_squared_error(Y_test, Y_test)
        #RMSE
        rmse_score_train = np.sqrt(metrics.mean_squared_error(Y_train, Y_pred))
        rmse_score_test = np.sqrt(metrics.mean_squared_error(Y_test, Y_test))

        result = [score_train,mae_score_train,mse_score_train,rmse_score_train,score_test,mae_score_test,mse_score_test,rmse_score_test]

    except Exception as e :
            print("========MODEL SCORE ERROR========") 
    finally : 
        pass
    return result 

#Finished Code 
if __name__ == '__main__':
    pass