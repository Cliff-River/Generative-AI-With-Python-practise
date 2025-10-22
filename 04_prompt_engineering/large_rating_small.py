#%% packages
import pprint
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field

load_dotenv(find_dotenv(usecwd=True))

#%% Initialize models
chat_model = ChatOpenAI(model_name="gpt-4o-mini")
rating_model = ChatGroq(model_name="openai/gpt-oss-120b")

#%%
class FeedbackResponse(BaseModel):
    rating : int = Field(..., description="Scoring in percentage")
    feedback : str = Field(..., description="Detailed feedback")

#%% Just test the model
feedback_to_test = {
    "rating": "85",
    "feedback": "The output provides a good overview of the American Civil War, covering major events and their significance. However, it could benefit from more depth in analyzing the long-term impacts and consequences of the war on American society and politics."
}
FeedbackResponse(**feedback_to_test).model_dump_json()

def chat_with_gpt(previous_output: str = "", feedback: str = ""):
    prompt = "The American Civil War was a civil war in the United States between the north and south." if not previous_output else ""
    # print(f"Prompt: {prompt}\nFeedback: {feedback}\nPrevious Output: {previous_output}\n")
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a skilled writer specializing in historical essays. Using the feedback provided, improve the previous output. Avoid special characters like apostrophes (') and double quotes (\")."),
        ("user", '''
            <prompt>{prompt}</prompt>
            <feedback>{feedback}</feedback>
            <previous_output>{previous_output}</previous_output>
         ''')
    ])
    chain = prompt_template | chat_model
    response = chain.invoke({
        "prompt": prompt,
        "feedback": feedback,
        "previous_output": previous_output
    })
    return response.content

#%%
def rating(target_score = 90, max_run = 5):
    previous_output = ""
    feedback = ""
    for i in range(max_run):
        previous_output = chat_with_gpt(previous_output, feedback)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
                You are an expert evaluator of historical essays. Evaluate the input in terms of how well it addresses the original task of explaining the key events and significance of the American Civil War. Consider factors such as: Breadth and depth of context provided; Coverage of major events; Analysis of short-term and long-term impacts/consequences. If you identify any gaps or areas that need further elaboration: 
                Return output as JSON with fields: <fields> 'rating': 'scoring in percentage without a %% sign', 'feedback': 'detailed feedback'</fields>.
            """),
            ("user", "Evaluate the following essay on the American Civil War: <essay>{essay}</essay>. Provide a rating in percentage without a %% sign and detailed feedback.")
        ])
        chain = prompt_template | rating_model | JsonOutputParser(pydantic_object=FeedbackResponse)
        response = chain.invoke({"essay": previous_output})
        rating_num = int(response['rating'])
        response["output"] = previous_output
        pprint.pprint(response)
        if rating_num >= target_score:
            return previous_output
        feedback = response['feedback']
    return previous_output

#%%
response = rating(90, 5)
#%%
response
#%%