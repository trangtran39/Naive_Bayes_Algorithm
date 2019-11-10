import numpy as np
import csv

def priorProb(training,dataset):
    return len(dataset)/len(training)

data = np.genfromtxt('lymphography.csv', delimiter=',')
#Laplace formula:
#P(Ai|C)=(Nic +1)/(Nc + c)

laplaceC = {1:4,2:2,3:2,4:2,5:2,6:2,7:2,8:2,9:4,10:4,11:3,12:4,13:4,14:8,15:3,16:2,17:2,18:8}

#split data into training set and validation set
validation = data[1]
training = data[31]
for i in range(2,31):
    validation = np.vstack((validation, data[i]))
for i in range(32,len(data)):
    training = np.vstack((training,data[i]))  

#seperate training set by class
thisDict = {}
for i in range(len(training)):
    classVal = training[i,0]
    if (classVal not in thisDict):
        thisDict[classVal]=list()
    thisDict[classVal].append(training[i])
      
#Prior probability:
prior={}
for i in range (2,5):
    prior[i]=priorProb(training, thisDict[i])

#Total instances of each class
classLen={}
for i in range (2,5):
    classLen[i]=len(thisDict[i])

##Re-organize thisDict to this form:
##thisDict={Class1: {Attr1: {Value1: Count1, Value2: Count2}, Attr2: {Value1: Count1, Value2: Count2}},
##          Class2: {Attr1: {Value1: Count1, Value2: Count2}, Attr2: {Value1: Count1, Value2: Count2}}}
    
for a in range(2,5):
    dictAttribute={}
    test = list(zip(*thisDict[a])) #from [48*19] to [19*48]
    for i in range(1, len(test)): # skip the fist row which is label row
        dictColumn={}
        for j in range(len(test[0])):
            value = test[i][j]
            if (value not in dictColumn):
                dictColumn[value]=1
            else:
                dictColumn[value]+= 1
            dictAttribute[i]=dictColumn
    thisDict[a]=dictAttribute

#Create predict array to store predict values when testing on validation set
#For each row ofvalidation dataset, we would calculate the probability for each class (2,3,4) -> Compare the probability
#-> Choose the maximal probability -> Classify the row to the class that yields the maximum probability
#->Add the class label into predict array.
#Repeat the process to every row. 
predict = []
lengthOfValidation = len(validation)
for row in range(0, lengthOfValidation):
    maxProb=0
    bestLabel=0
    for i in range(2,5): # for each class
        probability=1
        for j in range(1,len(validation[0])): # for each attribute
            if(validation[row][j] in thisDict[i][j]):
                #probability *= (thisDict[i][j][validation[row][j]]+1)/(classLen[i]+len(thisDict[i][j]))
                probability *= (thisDict[i][j][validation[row][j]]+1)/(classLen[i]+laplaceC[j])
            else:
                #probability *= 1/(classLen[i]+len(thisDict[i][j]))
                probability *= 1/(classLen[i]+laplaceC[j])
        probability *= prior[i]
        if (probability > maxProb):
            maxProb = probability
            bestLabel = i
    predict.append(bestLabel)

#Get the actual label column
actual = [row[0] for row in validation]

#Join actual and predict array into pair array
pair=np.vstack((actual,predict)).T

#Get the count of correct classification: When actual value == predict value
count=0
for row in pair:
    if (row[0]==row[1]):
        count += 1    
correctClassification = count
accuracyRate = correctClassification/len(pair)

#Calculate accuracy for each class.
#accuracy form: accuracy = {class1:[val1, val2], class2:[val1,val2]}
#where val1 == number of correct classification; val2 = total number of value in corresponding class
accuracy={}
for row in range(0,len(pair)):
    val = pair[row][0]
    if (val not in accuracy):
        accuracy[val]=[0,1]
        if (val == pair[row][1]):
            accuracy[val][0] += 1
    else:
        accuracy[val][1] += 1
        if (val == pair[row][1]):
            accuracy[val][0] += 1

#Write report to a file
f = open("output.txt","w")
report = "\t\tVALIDATION SET\n"
report += ("Choice of language: Python3\n"
           "\nFormula used to calculate Probability: "
           "\n\tLaplace P(Ai|C)=(Nic +1)/(Nc + c)\n"
           "\nWe have: P(C|X)=p(X|C)*P(C)/P(X) where C is class, X is test record"
           "\nP(X1,X2,X3...|C)=P(X1|C)*P(X2|C)*P(X3|C)*..."
            "\nSince all P(Cn|X) share same denominator which is P(X) "
           "\n==> We just need to calculate P(X|C)*P(C), then pick out the class that has highest probability"
           "\nOnce we generated array of predict values for validation dataset, we could compare those value to the actual value"
           "\nBy doing so, we could see the accuracy of our classification")        
report += ("\n\nOverall Performance:\n"
           "\tNumber of correct classification is %d\n"
        "\tNumber of incorrect classification is %d\n"
        "\tAccuracy is %f\n"%(correctClassification, len(pair)-correctClassification,accuracyRate))
for i in range(2,5):
    rate=accuracy[i][0]/accuracy[i][1]
    report += "\nAccuracy for class %d is %f\n"%(i,rate)

print("Please view the output.txt")
f.write(report)
f.close()
