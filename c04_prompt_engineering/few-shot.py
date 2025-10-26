#%%
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv(find_dotenv())

#%%
messages = [
    ("system", "You are a customer service specialist known for empathy, professionalism, and problem-solving. Your responses are warm yet professional, solution-focused, and always end with a concrete next step or resolution. You handle both routine inquiries and escalated issues with the same level of care."),
    ("user", """
    <examples>
    <example>
        Example 1:
        Customer: I received the wrong size shirt in my order #12345.
        Response: I'm so sorry about the sizing mix-up with your shirt order. That must be disappointing! I can help make this right immediately. You have two options:

        I can send you a return label and ship the correct size right away
        I can process a full refund if you prefer

        Which option works better for you? Once you let me know, I'll take care of it right away.
    </example>
    
    <example>
        Example 2:
        Customer: Your website won't let me update my payment method.
        Response: I understand how frustrating technical issues can be, especially when trying to update something as important as payment information. Let me help you with this step-by-step:
        First, could you try clearing your browser cache and cookies?
        If that doesn't work, I can help you update it directly from my end.
        Could you share your account email address so I can assist you further?
        New Request: {customer_request}
    </example>
    </examples>
     """
     ),
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)

#%%
model_name = "llama-3.3-70b-versatile"
model = ChatGroq(model_name=model_name)
chain = prompt_template | model
response = chain.invoke({"customer_request": "I want to change the shipping address for my order #67890."})
print(response.model_dump()["content"])
print("---------------------------")
  
#%%
response = chain.invoke({"customer_request": "I haven't received my refund yet after returning the item 2 weeks ago."})
print(response.model_dump()["content"])
#%%