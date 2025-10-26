#%%
import re

def custom_spliter(text):
    """
    Split text based on Roman numeral chapter headers.
    
    Pattern matches newlines followed by Roman numerals (I, II, III, IV, V, etc.)
    followed by a period and a space, and then a capital letter.
    
    Args:
        text (str): The text to split
        
    Returns:
        list: List of text chunks split by Roman numeral chapter headers
    """
    pattern = r'\n(?=[IVX]+\.\s[A-Z])' 
    return re.split(pattern, text)

def catch_title(text):
    pattern = r'\b[IVXLCDM]+\.\s+([A-Z\s\-]+?)\r\n'
    match = re.match(pattern, text)
    if match: 
        return match.group(1)
    else:
        return None
    
#%%