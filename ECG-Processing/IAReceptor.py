import paho.mqtt.client as mqtt
import json


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("ecg")


def on_message(client, userdata, msg):
    if msg.payload.decode() != None:

        data = msg.payload.decode()
        print(data)

        with open("ECG_values.json", "w") as document:
            json.dump(data, document, indent=4)
    else:
        print("None")


def main():
    client = mqtt.Client()
    client.connect("broker.hivemq.com")
    # client.connect("localhost")
    client.on_connect = on_connect
    client.on_message = on_message

    client.loop_forever()


main()
