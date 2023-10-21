import boto3
import os
from config import config

PREFIX_DOWNLOAD_FOLDER = config['download_folder']


def download_folder_from_s3(bucket_name, folder_name=""):
    s3 = boto3.client('s3', aws_access_key_id=config['aws']['access_key'],
                      aws_secret_access_key=config['aws']['secret_key'])

    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=folder_name)
    os.makedirs(os.path.join(PREFIX_DOWNLOAD_FOLDER,
                folder_name), exist_ok=True)
    result = []

    for obj in response['Contents']:
        object_key = obj['Key']
        if not object_key.endswith('/'):

            local_file_path = os.path.join(
                PREFIX_DOWNLOAD_FOLDER,
                folder_name, os.path.basename(object_key))

            try:
                s3.download_file(bucket_name, object_key, local_file_path)
                result.append(local_file_path)
                print(f"Downloaded file: {local_file_path}")
            except Exception as e:
                print(f"Error downloading file: {local_file_path}, {str(e)}")

    return result


def download_file_from_s3(bucket_name=config['bucket_name'], file_path=""):
    s3 = boto3.resource('s3', aws_access_key_id=config['aws']['access_key'],
                        aws_secret_access_key=config['aws']['secret_key'])

    local_file_path = os.path.join(
        PREFIX_DOWNLOAD_FOLDER, bucket_name, file_path)

    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
    print("Downloading file: ", local_file_path)

    try:
        s3.Bucket(bucket_name).download_file(file_path, local_file_path)
        print("Downloaded file: ", local_file_path)
    except Exception as e:
        print("Error downloading image from S3: ", e)
        return None
    return local_file_path
