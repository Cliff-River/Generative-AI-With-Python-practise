import random
from autogen import ConversableAgent
from nltk.corpus import words
from dotenv import load_dotenv, find_dotenv
import os
import nltk

load_dotenv(find_dotenv())

# %% LLM configuration
llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": os.environ.get("OPENROUTER_API_KEY"),
            "temperature": 0.2,
        }
    ]
}

# %% Pick a secret word
nltk.download('words')
word_list = [word for word in words.words() if len(word) <= 5]
secret_word = random.choice(word_list)
number_of_charaters = len(secret_word)
secret_word


def is_game_over(msg: dict) -> bool:
    content = str(msg.get("content", "")).lower()
    print("msg:", msg)
    return "you win" in content or "you lose" in content or secret_word.lower() in content

# %% host agent
hangman_host = ConversableAgent(
    "hangman_host",
    llm_config=llm_config,
    system_message=f"""
    You decide to use the secret word: {secret_word}
    It has {number_of_charaters} letters.
    The player selects letters to narrow down the word. 
    You start out with as many blanks as there are letters in the word.
    Return the word with the blanks filled in with the correct letters, at the correct position.
    Double check that the letters are at the correct position.
    If the player guesses a letter that is not in the word, you increment the number of fails by 1.
    If the number of fails reaches 7, the player loses.
    Return the word with the blanks filled in with the correct letters.
    Return the number of fails as x / 7.
    Say 'You lose!' if the number of fails reaches 7, and reveal the secret word.
    Say 'You win!' if you have found the secret word.
    """,
    human_input_mode="NEVER",
    is_termination_msg=is_game_over,
)

# %% human player
hangman_player = ConversableAgent(
    "hangman_player",
    llm_config=llm_config,
    system_message="""
    "You are guessing the secret word. 
    You select letters to narrow down the word. Only provide the letters as 'Guess: ...'.
    """,
    human_input_mode="ALWAYS",
    is_termination_msg=is_game_over,
)

# %%
result = hangman_host.initiate_chat(
    hangman_player,
    message="I have a secret word. Start guessing.")

# %%