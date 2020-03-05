import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier


class IA:
    def __init__(self, train, test):
        self.train = pd.read_csv(train, header=None)
        self.test = pd.read_csv(test, header=None)

    def getTrain(self):
        return self.train.head()

    def getTest(self):
        return self.test.head()

    def getDimensionTraining(self):
        return self.train.shape()

    def getDimensionTest(self):
        return self.test.shape

    def graph(self):
        plt.subplot(2, 2, 1)
        plt.plot(self.train.iloc[0, :187])

        plt.subplot(2, 2, 2)
        plt.plot(self.train.iloc[1, :187])

        plt.subplot(2, 2, 3)
        plt.plot(self.train.iloc[2, :187])

        plt.subplot(2, 2, 4)
        plt.plot(self.train.iloc[3, :187])

        print(self.train[187][0], self.train[187][1], self.train[187][2], self.train[187][3])

    def getNumberClasses(self):
        return self.train[187].value_counts()

    def graphClasses(self):
        f, axs = plt.subplots(5, 1, figsize=(5, 10))

        plt.subplot(5, 1, 1)
        plt.ylabel("Normal")
        plt.ylim(0, 1)
        plt.plot(self.train.loc[self.train[187] == 0.0].loc[0])

        plt.subplot(5, 1, 2)
        plt.ylabel("Supraventricular Premature")
        plt.ylim(0, 1)
        plt.plot(self.train.loc[self.train[187] == 1.0].loc[72471])

        plt.subplot(5, 1, 3)
        plt.ylabel("Premature VC")
        plt.ylim(0, 1)
        plt.plot(self.train.loc[self.train[187] == 2.0].loc[74694])

        plt.subplot(5, 1, 4)
        plt.ylabel("Fusion")
        plt.ylim(0, 1)
        plt.plot(self.train.loc[self.train[187] == 3.0].loc[80482])

        plt.subplot(5, 1, 5)
        plt.ylabel("Unclassifiable Beat")
        plt.ylim(0, 1)
        plt.plot(self.train.loc[self.train[187] == 4.0].loc[81123])

    def balanceClasses(self):
        train_target = self.train[187]
        label = 187
        df = self.train.groupby(label, group_keys=False)
        self.train = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()))).reset_index(drop=True)

    def labelFeature(self):
        labelData = self.train[187]
        labelTest = self.test[187]
        del self.train[187]
        del self.test[187]
        self.features = self.train.values
        self.featuresTest = self.test.values
        self.labels = labelData.values
        self.labelsTest = labelTest.values

    def trainBoostingTree(self):
        clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), random_state=0)
        clf.fit(self.features, self.labels)
        return clf

    def traingBoostingRF(self):
        clf = AdaBoostClassifier(RandomForestClassifier(), random_state=0)
        clf.fit(self.features, self.labels)
        return clf

    def dataReceived(self, data):
        self.data = []
        self.labelT = []
        self.data.append(data[0:187])
        self.labelT.append(data[187])

    def predictModel(self, clf):
        predictions = clf.predict(self.data)
        probPredict = max(clf.predict_proba(self.data)[0])
        score = clf.score(self.data, self.labelT)
        print(confusion_matrix(self.labelT, predictions))
        print(classification_report(self.labelT, predictions))
        print(accuracy_score(self.labelT, predictions))
        return predictions[0], probPredict


class Disease:
    __nDisease = {0.0: 'Normal', 1.0: 'Supraventricular Premature', 2.0: 'Premature VC', 3.0: 'Fusion',
                  4.0: 'Unclassifiable Beat'}

    def __init__(self, name, probability):
        self.name = self.__nDisease[name]
        self.probability = probability

    def __str__(self):
        return self.name + " " + str(self.probability)

    def getName(self):
        return self.name

    def getProbability(self):
        return self.probability

    def setName(self, name):
        self.name = name

    def setProbability(self, probability):
        self.probability = probability
