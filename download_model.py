import urllib.request
import ssl

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
output = "pose_landmarker_heavy.task"

# Create SSL context to avoid cert errors
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

print(f"Downloading {url}...")
try:
    with urllib.request.urlopen(url, context=ctx) as response, open(output, 'wb') as out_file:
        data = response.read()
        out_file.write(data)
    print("Download complete.")
except Exception as e:
    print(f"Error: {e}")
