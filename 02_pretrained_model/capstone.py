#%%
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
# %%

feedback = [
    "I recently bought the EcoSmart Kettle, and while I love its design, the heating element broke after just two weeks. Customer service was friendly, but I had to wait over a week for a response. It's frustrating, especially given the high price I paid.",
    "Die Lieferung war super schnell, und die Verpackung war großartig! Die Galaxy Wireless Headphones kamen in perfektem Zustand an. Ich benutze sie jetzt seit einer Woche, und die Klangqualität ist erstaunlich. Vielen Dank für ein tolles Einkaufserlebnis!",
    "Je ne suis pas satisfait de la dernière mise à jour de l'application EasyHome. L'interface est devenue encombrée et le chargement des pages prend plus de temps. J'utilise cette application quotidiennement et cela affecte ma productivité. J'espère que ces problèmes seront bientôt résolus."
]

#%%

candidates = ['defect', 'delivery', 'interface']
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
semtiment_cognitor = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")

classification_res  = classifier(feedback, candidate_labels=candidates)
semtiment_res = semtiment_cognitor(feedback)
result = {
    "categories" : [ res["labels"][0] for res in classification_res ],
    "semtiment" : [res["label"] for res in semtiment_res]
}
result

# %%

