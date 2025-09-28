from transformers import pipeline
from langchain_community.document_loaders import ArxivLoader

summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

query = "prompt engineering"
loader = ArxivLoader(query=query, load_max_docs=1)
docs = loader.load()
article_text = docs[0].page_content

# article_text = """
# Generative AI's rapid rise is a testament to its transformative potential.
# It has unlocked possibilities that were unimaginable a decade ago and
# continues to redefine the boundaries of what machines and humans
# can achieve together. As we look ahead, the question is no longer
# whether generative AI will shape our future, but how we'll shape the
# future of generative AI.
# Generative AI's rapid development is not just a technological
# masterpiece. This transformative force operates across creative,
# technical, and commercial landscapes, and its influence transcends
# the borders of research labs, reaching into studios, factories, and
# boardrooms. Generative AI reshapes the way we create, build, and
# conduct business.
# """

result = summarizer(article_text[:2000], min_length=20, max_length=80, do_sample=False)
print(result[0]['summary_text'])

