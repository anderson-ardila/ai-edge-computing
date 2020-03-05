import json
import time

import paho.mqtt.client as mqtt  # import the client1


def main():
    broker_address = "broker.hivemq.com"  # use external broker
    # broker_address="localhost"
    client = mqtt.Client()  # create new instance
    client.connect(broker_address)  # connect to broker
    while client.loop() == 0:
        value1 = ""
        value2 = ""
        value3 = ""
        value4 = ""

        data = {"ecg": {"value1": value1, "value2": value2, "value3": value3, "value4": value4}}

        json_str = json.dumps(data, indent=4)

        client.publish("ecg", json_str)
        time.sleep(1)


main()
