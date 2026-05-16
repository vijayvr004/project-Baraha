import urllib.request
import urllib.error
import json

groq_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

try:
    url = "https://api.groq.com/openai/v1/models"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {groq_key}',
        'User-Agent': 'curl/8.4.0',
        'Accept': '*/*'
    })
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        for d in data.get('data', []):
            print(d.get('id'))
except Exception as e:
    print("Error:", e)
