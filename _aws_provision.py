"""Aprovisiona el bucket S3 para la Gestión Documental del expediente.
Sin secretos embebidos: lee credenciales de variables de entorno
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION)."""
import sys
import boto3
from botocore.exceptions import ClientError

REGION = "us-east-1"
s3 = boto3.client("s3", region_name=REGION)
sts = boto3.client("sts", region_name=REGION)

ident = sts.get_caller_identity()
acct = ident["Account"]
print("Cuenta AWS:", acct)
print("Identidad :", ident["Arn"])

bucket = f"parcial1-expediente-docs-{acct}"
print("Bucket    :", bucket)

# 1) Crear bucket (idempotente). us-east-1 => sin LocationConstraint.
try:
    s3.create_bucket(Bucket=bucket)
    print("[ok] Bucket creado.")
except ClientError as e:
    code = e.response["Error"]["Code"]
    if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
        print(f"[=] Bucket ya existe ({code}).")
    else:
        print("[ERROR] create_bucket:", code, e.response["Error"].get("Message"))
        sys.exit(1)

# 2) Bloquear todo acceso público (el backend hace de proxy; bucket privado).
s3.put_public_access_block(
    Bucket=bucket,
    PublicAccessBlockConfiguration={
        "BlockPublicAcls": True,
        "IgnorePublicAcls": True,
        "BlockPublicPolicy": True,
        "RestrictPublicBuckets": True,
    },
)
print("[ok] Acceso público bloqueado.")

# 3) Cifrado por defecto en reposo (SSE-S3 / AES256).
s3.put_bucket_encryption(
    Bucket=bucket,
    ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]
    },
)
print("[ok] Cifrado en reposo (AES256) activado.")

# 4) Prueba real de permisos: put + get + delete.
key = "_healthcheck/ping.txt"
s3.put_object(Bucket=bucket, Key=key, Body=b"ok")
data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
s3.delete_object(Bucket=bucket, Key=key)
print("[ok] Prueba put/get/delete:", data.decode())

print("BUCKET_READY", bucket)
