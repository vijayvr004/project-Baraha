import urllib.request
import urllib.error

groq_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

try:
    url = "https://api.groq.com/openai/v1/models"
    req = urllib.request.Request(url, headers={
        'Authorization': f'Bearer {groq_key}',
        'User-Agent': 'curl/8.4.0',
        'Accept': '*/*'
    })
    with urllib.request.urlopen(req) as resp:
        print("Success:", resp.status)
except urllib.error.HTTPError as e:
    print("HTTPError:", e.code)
    print(e.read().decode('utf-8'))
except Exception as e:
    print("Error:", e)
