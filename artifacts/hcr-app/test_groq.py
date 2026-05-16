import urllib.request
import json

groq_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

print("=== GROQ MODELS ===")
try:
    url = "https://api.groq.com/openai/v1/models"
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {groq_key}'})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        for d in data.get('data', []):
            if 'vision' in d.get('id', '').lower():
                print(d.get('id'))
except Exception as e:
    print("Groq Error:", e)
