
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

def heart():

    #Importing the heart dataset
    Dataset = pd.read_csv('heart.csv')

    #Taking all the parameters excpet target as X and the Target as Y
    X = Dataset.drop(["target"],axis=1)
    Y = Dataset["target"]

    #Splitting into Train and Test portions with 70%training
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.30)

    print("-------------------------------------")
    print(f"Length of the Dataset = {len(Dataset)}")
    print("-------------------------------------")
    print(f"Length of the X_Train,Y_Train = {len(X_train),len(Y_train)}")
    print(f"Length of the X_Test,Y_Test = {len(X_test),len(Y_test)}")

    Train_loaded = DataLoader(X_train,batch_size=16)
    Test_loaded = DataLoader(X_test,batch_size=16)

    print("Size : Train",len(Train_loaded))
    print("Size : Test",len(Test_loaded))


def main():
    heart()

main()
