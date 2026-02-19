import json
from dotenv import load_dotenv
from openai import OpenAI

# This is just a simple hello, world-type program. It has all sorts of best practice stripped out of it, so we can focus just on clarity.

# builds a prompt to send to ChatGPT
def get_prompt(name):
    return f"""You are resolving a business/organization name to an official LinkedIn COMPANY profile.
Task: Given a person's name, say hello to them. i.e. if we're given Miles, print Hello, Miles
Name to say hello to: {name}
"""


# calls ChatGPT to answer our prompt. 
def get_answer(prompt, model):
    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt,
    )

    # Extract the assistant's text output
    return response.output[0].content[0].text






# our main function. specify details here
def main():
    load_dotenv(override=True) # this loads our API key 

    name = "Margaret"
    model = "gpt-4.1" # see for other models: https://developers.openai.com/api/docs/models
    
    prompt = get_prompt(name)
    print(f"Prompt: {prompt}")

    response = get_answer(prompt, model)
    print(f"Response: {response}")
    

# initial execution point
if __name__ == "__main__":
    main()
