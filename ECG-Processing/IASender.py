import paho.mqtt.client as mqtt  # import the client1
import json
import time


def main():
    broker_address = "broker.hivemq.com"  # use external broker
    # broker_address="localhost"
    client = mqtt.Client()  # create new instance
    client.connect(broker_address)  # connect to broker

    while client.loop() == 0:
        disease = "colera"
        probability = 0.56

        data = {"disease": {"name": disease, "probability": probability}}

        json_str = json.dumps(data, indent=4)

        client.publish("desease", json_str)
        time.sleep(1)


main()
