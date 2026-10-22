from rag.retriever import Retriever
r = Retriever()
for c in r.query("vảy nến có ngứa không", disease_key="psoriasis"):
    print(round(c["score"],3), c["section_title"])
