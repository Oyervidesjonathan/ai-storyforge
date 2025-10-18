import os, uuid, boto3
from botocore.client import Config

S3 = boto3.client(
    "s3",
    endpoint_url=os.getenv("S3_ENDPOINT", "http://localhost:9000"),
    aws_access_key_id=os.getenv("S3_KEY","minio"),
    aws_secret_access_key=os.getenv("S3_SECRET","minio123"),
    config=Config(signature_version="s3v4"),
    region_name="us-east-1",
)
BUCKET = os.getenv("S3_BUCKET","kids-pub")

def put_image(data: bytes, ext="png") -> str:
    key = f"images/{uuid.uuid4()}.{ext}"
    S3.put_object(Bucket=BUCKET, Key=key, Body=data, ContentType=f"image/{ext}")
    return f"{os.getenv('S3_PUBLIC','http://localhost:9000')}/{BUCKET}/{key}"
