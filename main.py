from re import split

from transformers import pipeline
import textstat
from nltk.corpus import words

nlp = pipeline("fill-mask")

sentence = input('Enter a sentence with difficult words :')
wordList = split('[^a-zA-Z]', sentence)
for w in wordList:
     if w in words.words() and int(textstat.difficult_words(w)) == 1:  # If the word is difficult
         preds = nlp(sentence.replace(w,'<mask>'))
         new_sentences = []
         ease_levels = []
         for p in preds:
             s = sentence.replace(w,str(nlp.tokenizer.decode([p['token']])).strip())
             new_sentences.append(s)
             ease_levels.append(float(textstat.flesch_reading_ease(s)))
         sentence = new_sentences[ease_levels.index(min(ease_levels))]

print(sentence)




