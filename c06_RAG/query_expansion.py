# %% packages: use groq as chat model
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv(find_dotenv())  # read local .env file

# %% Query expansion function
def expand_query_with_groq(original_query: str, number : int, model_name = "llama-3.3-70b-versatile") -> str:
    messsage = [
        ( "system", """"You are part of an information retrieval system. You are given a user query and you need to expand the query to improve the search results. Return ONLY a list of expanded queries. 
            Be concise and focus on synonyms and related concepts.
            Format your response as a Python list of strings.
            The response must:
            1. Start immediately with [
            2. Contain quoted strings
            3. End with ]
            Example correct format:    
            ["alternative query 1", "alternative query 2", "alternative query 3"]
        """),
        ( "user", """Original Query: "{original_query}", Please provide {number} expanded queries.""" )
    ]
    prompt = ChatPromptTemplate.from_messages(messsage)
    model = ChatGroq(model=model_name)
    chain = prompt | model
    response = chain.invoke(dict(original_query=original_query, number=number))
    print(type(response.content))
    return eval(response.content)

# %% Example usage
original_query = "Machine Learning"
expanded_queries = expand_query_with_groq(original_query, number=3)
expanded_queries, type(expanded_queries)

# %%