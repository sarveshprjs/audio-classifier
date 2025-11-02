import requests

url = "http://127.0.0.1:5000/predict"
files = {'audio': open(r"D:\downloads-d\car-horn-beepsmp3-14659.mp3", 'rb')}

response = requests.post(url, files=files)

print("Status:", response.status_code)
print("Response:", response.text)
