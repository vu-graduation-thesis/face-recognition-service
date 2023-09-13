import redis
from config import config
import json
import training

redisInstance = None


def get_redis_connection():
    global redisInstance
    if redisInstance is None:
        redisInstance = redis.StrictRedis(host=config['redis']['host'], port=config['redis']
                                          ['port'], password=config['redis']['password'])
    return redisInstance


class Message:
    def __init__(self, type, value) -> None:
        self.type = type
        self.value = value
        pass


def message_listener():
    print("Starting message listener...")
    r = get_redis_connection()

    pong = r.ping()
    if pong:
        print("Connected to Redis")
    else:
        print("Could not connect to Redis")

    pubsub = r.pubsub()
    pubsub.subscribe(config['training_data_queue'])
    print("Listening for messages...")

    for message in pubsub.listen():
        if message['type'] == 'message':
            print("Message received", message['data'].decode())
            data = json.loads(message['data'].decode())
            if data['type'] == 'TRAINING_DATA':
                student_id = data['value']
                training.start_training_model(student_id)
