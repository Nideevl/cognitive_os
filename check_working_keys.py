import os
from dotenv import load_dotenv
load_dotenv(override=True)   # remove if cli.py doesn't use dotenv
from agent.key_manager import GroqKeyRotator

r = GroqKeyRotator()
name_of = {}
for n in ["GROQ_API_KEY"] + [f"GROQ_API_KEY_{i}" for i in range(1, 51)]:
    v = os.getenv(n)
    if v:
        name_of[v.strip()] = n

for i, (k, c) in enumerate(zip(r.keys, r.clients)):
    try:
        c.models.list()
        status = "OK"
    except Exception as e:
        status = "FAIL " + str(e)[:50]
    print(f"[{i}] {name_of.get(k, 'pool/arg')}  len={len(k)}  ...{k[-6:]}  {status}")