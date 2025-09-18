import os
import requests

def handler(event, context):
    print(event)
    print(context)
    events = event["data"]["events"]
    for event in events:
        if event["eventName"] == "tos:ObjectCreated:Put":
            url = event["tos"]["object"]["key"]
            url_data = url.split("/")
            char_id = url_data[1]
            user_id = url_data[2]
            image_url = url
            requests.post(f"{os.getenv('POSTGRES_HOST')}/api/v1/storage/trigger", json={"char_id": char_id, "user_id": user_id, "image_url": image_url})
