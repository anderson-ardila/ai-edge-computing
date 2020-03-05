import json

import paho.mqtt.client as mqtt


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("desease")


def on_message(client, userdata, msg):
    if msg.payload.decode() is not None:

        data = msg.payload.decode()
        print(data)

        with open("IA_values.json", "w") as document:
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
