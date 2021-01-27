from transformers import pipeline
f = open("res/1-1000.txt", "r")
words=f.read().splitlines()

nlp = pipeline("fill-mask")
preds = nlp(f"I am exhausted, it's a {nlp.tokenizer.mask_token} task.")
for p in preds:
    if str(nlp.tokenizer.decode([p['token']])).strip() in words:
        print('Found Replacement')
        print("original : I am exhausted, it's a **** task.")
        print("simplified : I am exhasuted, it's a "+str(nlp.tokenizer.decode([p['token']])).strip()+" task.")
        break


