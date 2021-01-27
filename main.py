from transformers import pipeline
import textstat

nlp = pipeline("fill-mask")
preds = nlp(f"I am excited, it's a {nlp.tokenizer.mask_token} topic.")

new_sentences=[]
ease_levels=[]

for p in preds:
    s = "I am excited, it's a " + str(nlp.tokenizer.decode([p['token']])).strip() + " topic."
    new_sentences.append(s)
    ease_levels.append(float(textstat.flesch_reading_ease(s)))

print("original sentence : I am excited, it's a **** topic.")
print("simplified sentence : "+new_sentences[ease_levels.index(min(ease_levels))])


