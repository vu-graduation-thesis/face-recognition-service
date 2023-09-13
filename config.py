from decouple import config as env_config

config = {
    "port": env_config('PORT'),
    "redis": {
        "host": env_config('REDIS_HOST'),
        "port": env_config('REDIS_PORT'),
        "password": env_config('REDIS_PASSWORD'),
    },
    "aws": {
        "access_key": env_config('AWS_ACCESS_KEY'),
        "secret_key": env_config('AWS_SECRET_KEY'),
    },
    "raw_folder_data": "raw-data",
    "bucket_name": "face-recognition-service",
    "training_folder_data": "training-data",
    "training_data_queue": env_config("TRAINING_DATA_QUEUE"),
    "trained_model": "trained_model.yml",
}
