import json
from joblib import dump, load
import time

import joblib
import matplotlib.pyplot as plt
import paho
import paho.mqtt.client as mqtt
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
import sys
import threading


class IA:
    def __init__(self, train, test):
        self.train = pd.read_csv(train, header=None)
        self.test = pd.read_csv(test, header=None)

    def getTrain(self):
        return self.train.head()

    def getTest(self):
        return self.test.head()

    def getDimensionTraining(self):
        return self.train.shape

    def getDimensionTest(self):
        return self.test.shape

    def graph(self, data):
        plt.plot(data[:len(data)])
        plt.show()

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
        self.loadModel()
        if(self.clf==None):
            print("Cruel")
            self.clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=20), random_state=0)
            self.clf.fit(self.features, self.labels)
            self.saveModel()
    def saveModel(self):
        dump(self.clf, 'modelo_entrenado.joblib')
    def loadModel(self):
        try:
            self.clf= load('modelo_entrenado.joblib')
        except:
            self.clf=None
    def dataReceived(self, data):
        x=self.normalize(data)
        self.data = []
        self.labelT = []
        self.data.append(x[0:187])
        print(self.data)
        self.labelT.append(x[187])
        print(self.labelT)

    def normalize(self,data):
        print("Entró")
        while (len(data)<188):
            print("x")
            data.append(0.0);
        print("Salió")
        maximo = abs(max(data))
        minimo = abs(min(data))
        if (maximo < minimo):
            maximo = minimo
        for i in range(len(data)):
            data[i] = (data[i] / maximo)
        print("Terminó")
        return data;

    def predictModel(self):
        predictions = self.clf.predict(self.data)
        probPredict = max(self.clf.predict_proba(self.data)[0])
    #   score = self.clf.score(self.data, self.labelT)
        #print(confusion_matrix(self.labelT, predictions))
        #print(classification_report(self.labelT, predictions))
        # print(accuracy_score(self.labelT, predictions))
        return predictions[0], format(probPredict)


class Disease:
    __nDisease = {0.0: 'Normal', 1.0: 'Supraventricular Premature \n(atrial fibrillation)', 2.0: 'Premature VC (cardiomyopathy)', 3.0: 'Fusion',
                  4.0: 'Unclassifiable Beat'}

    def __init__(self, name="", probability=0,id=0):
        self.name = self.__nDisease[name]
        self.probability = probability
        self.id=id

    def _str_(self):
        return self.name + " " + str(self.probability)

    def getName(self):
        return self.name

    def getProbability(self):
        return self.probability

    def setName(self, name):
        self.name = self.__nDisease[name]

    def setProbability(self, probability):
        self.probability = probability

    def getId(self):
        return self.id

    def setId(self,id):
        self.id=id


listAux = []
TF = False
json_str = ""
iaGlobal = None


class myThread(threading.Thread):
    def __init__(self, threadID, name, clientePublicar, topicPublicar, numeroElegirMetodo, mensajeEnviar, ia, data):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.client = clientePublicar
        self.topic = topicPublicar
        self.numero = numeroElegirMetodo
        self.mensaje = mensajeEnviar
        self.ia = ia
        self.data = data

    def run(self):
        print("Starting " + self.name)
        metodo_recibir(self.client, self.topic, self.numero, self.mensaje, self.ia, self.data)
        print("Exiting " + self.name)


def metodo_recibir(clientePublicar, topicPublicar, numeroElegirMetodo, mensaje, ia, data):
    if numeroElegirMetodo == 1:
        funcion_auxiliar_recibir(clientePublicar, topicPublicar)
    if numeroElegirMetodo == 2:
        funcion_auxiliar_enviar(clientePublicar, topicPublicar, mensaje)
    if numeroElegirMetodo == 3:
        funcion_auxiliar_ia(ia, data)


def funcion_auxiliar_ia(ia, data):
    ia.dataReceived(data)
    pred, prob = ia.predictModel()
    print(pred, prob)
    d = Disease(pred, prob)
    global json_str
    json_str = json.dumps(d.__dict__)
    global TF
    TF = False


def funcion_auxiliar_enviar(client, topic, mensaje):
    client.on_publish = on_publish
    client.on_connect = on_connect
    client.connect(host='broker.hivemq.com', port=1883)

    client.publish(topic, mensaje)
    client.loop_start()
    time.sleep(1)


def funcion_auxiliar_recibir(client, topic):
    client.on_message = on_message
    client.on_connect = on_connect
    client.connect(host='broker.hivemq.com', port=1883)
    client.subscribe(topic)
    client.loop_forever()
    time.sleep(3)

def on_connect(client, userdata, flags, rc):
    print('connected (%s)' % client._client_id)


def on_publish(client, userdata, mid):
    print("Message Published...")
    client.loop_stop(force=False)


def on_message(client, userdata, message):
    id=0;
    if message.payload == "exit":
        sys.exit(0)
    print('------------------------------')  # Imprime una linea para diferenciar un mensaje de otro
    print(message.topic)  # Imprime el 'topic' tema del mensaje
    messageRecibido = json.loads(message.payload)  # Decodifica el archivo JSON
    print(messageRecibido)
    for element in messageRecibido['ecg']:
        listAux.append(float(element))
    for element in messageRecibido['id']:
        id=element;
    iaGlobal.dataReceived(listAux)
    listAux.clear()
    pred, prob = iaGlobal.predictModel()
    print(pred, prob)
    d = Disease(pred, prob,id)
    global json_str
    json_str = json.dumps(d.__dict__)

    global TF
    TF = True
    cliente3 = paho.mqtt.client.Client()
    ejecutarHilos(5, "Hilo enviar", cliente3, cliente3._client_id, "test/IA", 2, json_str, None, None)


def ejecutarHilos(threadID, threadName, cliente, clienteId, topicPublicar, numeroElegirMetodo, mensajeEnviar, ia, data):
    # client = paho.mqtt.client.Client(client_id=clienteId, clean_session=False)
    threadMQTT = myThread(threadID, threadName, cliente, topicPublicar, numeroElegirMetodo, mensajeEnviar, ia, data)
    threadMQTT.start()


def main():
    print("")
    ia = IA('mitbih_train.csv', 'mitbih_test.csv')
    global iaGlobal
    iaGlobal = ia
    print(ia.getTrain())
    print(ia.getTest())
    print(ia.getDimensionTraining())
    print(ia.getDimensionTest())
    print(ia.getNumberClasses())
    ia.graphClasses()
    ia.balanceClasses()
    print(ia.getDimensionTraining())
    print(ia.getNumberClasses())
    ia.labelFeature()
    ia.trainBoostingTree()
    client = paho.mqtt.client.Client()
    ejecutarHilos(1, "primerHilo", client, client._client_id, "ECG", 1, '', ia, listAux)


if __name__ == "__main__":
    main()
