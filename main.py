import json
import os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

load_dotenv()


class QA_Agent:
    def __init__(self):
        client = os.getenv("GROQ_KEY")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=client,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        print("Model Loaded")

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_system_prompt()),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.q_chain = qa_prompt | self.llm

        self.chat_history = {}
        self.chat_model = RunnableWithMessageHistory(
            self.q_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        print("Enjoy!!")

    def get_session_history(self, session_id: str):
        if session_id not in self.chat_history:
            self.chat_history[session_id] = ChatMessageHistory()
        return self.chat_history[session_id]

    def get_system_prompt(self):
        system_prompt = """
            You are my assistant who will help me by organizing my thoughts.
            Your job is to ask questions that solve my goals or help me
            understand my issues better and help me gain new insights.
            Ask thought provoking questions which will challenge my
            abilities or give me a deeper perspective to myself. Always
            ask a single question at a time.
            """
        return system_prompt

    def agent_chat(self, usr_prompt, session_id="acc_setup"):
        response = self.chat_model.invoke(
            {"input": usr_prompt}, config={"configurable": {"session_id": session_id}}
        )

        # Extract the content directly if response has content attribute
        response_text = (
            getattr(response, "content", None)
            if hasattr(response, "content")
            else str(response)
        )

        return response_text


# Telegram bot functions
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome! I'm your assistant. How can I help you today?"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    response = chat_agent.agent_chat(user_text)
    await update.message.reply_text(response)


def main():
    global chat_agent
    chat_agent = QA_Agent()

    application = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    application.run_polling()


if __name__ == "__main__":
    main()


# import json
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_ollama import ChatOllama
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# import os
# import getpass

# load_dotenv()


# class QA_Agent:
#     def __init__(self):
#         # llm = ChatOllama(model="llama3.2:1b")
#         client = os.getenv("GROQ_KEY")
#         llm = ChatGroq(
#             model="llama-3.1-8b-instant",
#             api_key=client,
#             temperature=0,
#             max_tokens=None,
#             timeout=None,
#             max_retries=2,
#         )
#         print("Model Loaded")

#         qa_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", self.get_system_prompt()),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         q_chain = qa_prompt | llm

#         self.chat_history = {}
#         self.chat_model = RunnableWithMessageHistory(
#             q_chain,
#             self.get_session_history,
#             input_messages_key="input",
#             history_messages_key="chat_history",
#         )
#         print("Enjoy!!")

#     def get_session_history(self, session_id: str):
#         if session_id not in self.chat_history:
#             self.chat_history[session_id] = ChatMessageHistory()
#         return self.chat_history[session_id]

#     def reformat_json(self, json_string, path):
#         data = json.loads(json_string)
#         formatted_messages = []
#         for message in data["messages"]:
#             if message["type"] == "human":
#                 user_content = message["content"]
#                 ai_response = next(
#                     (msg["content"] for msg in data["messages"] if msg["type"] == "ai"),
#                     None,
#                 )
#                 if ai_response:
#                     formatted_messages.append(
#                         {"user_query": user_content, "ai_response": ai_response}
#                     )

#         # Creating the desired JSON structure
#         formatted_data = {"messages": formatted_messages}
#         # Convert Python dictionary back to JSON string
#         with open(path, "w") as f:
#             json.dump(formatted_data, f, indent=2)
#         return formatted_data

#     def save_history(self, path="./agent_stup.json"):
#         history = self.chat_history["acc_setup"].json()
#         self.reformat_json(history, path)
#         print(f"History saved to {path}")

#     def get_system_prompt(self):
#         system_prompt = """
#             You are my assistant who will help me by organizing my thoughts.
#             Your job is to ask questions that solve my goals or help me
#             understand my issues better and help me gain new insights.
#             Ask thought provoking questions which will challenge my
#             abilities or give me a deeper perspective to myself. Always
#             ask a single question at a time.
#             """
#         return system_prompt

#     def agent_chat(self, usr_prompt):
#         response = self.chat_model.invoke(
#             {"input": usr_prompt}, config={"configurable": {"session_id": "acc_setup"}}
#         )
#         return response


# def main():
#     chat_agent = QA_Agent()
#     print("You are now conversing with your assistant. Good Luck!")
#     print("On what thoughts do you want to discuss today: ")
#     while True:
#         prompt = input("Enter you query|('/exit' to quit session): ")
#         if prompt == "/exit":
#             print("You will have a fortuitous encounter soon, God Speed!")
#             break
#         response = chat_agent.agent_chat(prompt)
#         print(f"Assistant: {response}")
#         print("-" * 30)
#     chat_agent.save_history()


# if __name__ == "__main__":
#     main()
