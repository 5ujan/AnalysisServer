from langchain_experimental.agents import create_csv_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
import google.generativeai as genai
# from langchain_core.schema import ChatMessage, HumanMessage, SystemMessage
def main():
    load_dotenv()

    # Load the Google API key from the environment variable
    if os.getenv("GOOGLE_API_KEY") is None or os.getenv("GOOGLE_API_KEY") == "":
        print("GOOGLE_API_KEY is not set")
        exit(1)
    else:
        print("GOOGLE_API_KEY is set")

    # Configure the API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')

    # Ask the user to input the path to the CSV file
    csv_file_path = input("Enter the path to the CSV file: ")

    if not os.path.exists(csv_file_path):
        print("CSV file does not exist")
        exit(1)

    # Create the CSV agent with Gemini API
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, max_output_tokens=2048)
    agent = create_csv_agent(llm, csv_file_path, verbose=True, allow_dangerous_code=True)

    while True:
        user_question = input("Ask a question about your CSV (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            break

        if "trends" in user_question:
            # Create a list of ChatMessage objects
            
            response = model.generate_content(
            [user_question + "Based on the apriori association rules provided,find interesting trends,patterns and insights which can help the related organization, in layman language. Use percentage metrices if you can. make it simple."]
        ).text
        else:
            response = agent.run(user_question)
            
        print(response)

      


if __name__ == "__main__":
    main()