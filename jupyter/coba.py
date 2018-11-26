import json

kategori = []
pertanyaan = []
jawaban = []

with open('data/data_source/IslamicQA.json', encoding = 'utf-8-sig') as f:
    iqa = json.load(f)
print(iqa)