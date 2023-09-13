import boto3
import os
from config import config


def getResourceFromS3(student_id=""):
    if student_id != "":
        print(f"\nStart get resource of student_id {student_id} from S3")
    else:
        print(f"\nStart get all resource from S3")
    s3 = boto3.resource('s3', aws_access_key_id=config['aws']['access_key'],
                        aws_secret_access_key=config['aws']['secret_key'])

    objects = s3.Bucket(config['bucket_name']
                        ).objects.filter(Prefix=student_id)

    raw_folder_data = config['raw_folder_data']

    os.makedirs(raw_folder_data, exist_ok=True)

    for obj in objects:
        if not obj.key.endswith('/') and os.path.dirname(obj.key).isdigit():
            folder = os.path.dirname(obj.key)
            local_image_folder = os.path.join(
                raw_folder_data, folder)
            os.makedirs(local_image_folder, exist_ok=True)

            local_image_path = os.path.join(
                local_image_folder, os.path.basename(obj.key))

            image_data = obj.get().get('Body').read()

            with open(local_image_path, 'wb') as file:
                file.write(image_data)

    if student_id != "":
        print(f"Completed get resource of student_id {student_id} from S3")
    else:
        print(f"Completed get all resource from S3")
