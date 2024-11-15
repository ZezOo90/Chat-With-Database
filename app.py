from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os


class DatabaseConnector:
    """Handles the connection to the SQL database."""
    def __init__(self):
        self.db = None

    def connect(self, user: str, password: str, host: str, port: str, database: str):
        """Initialize database connection."""
        db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
        self.db = SQLDatabase.from_uri(db_uri)
        return self.db


class SQLAssistant:
    """Manages LLM-driven SQL query generation and response handling."""
    def __init__(self, gemini_api_key: str):
        self.gemini_api_key = gemini_api_key

    def get_sql_chain(self, db: SQLDatabase):
        """Creates a chain to generate SQL queries from user input."""
        template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
        
        <SCHEMA>{schema}</SCHEMA>
        
        Conversation History: {chat_history}
        
        Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=self.gemini_api_key)

        def get_schema(_):
            return db.get_table_info()

        return (
            RunnablePassthrough.assign(schema=get_schema)
            | prompt
            | llm
            | StrOutputParser()
        )

    def generate_response(self, user_query: str, db: SQLDatabase, chat_history: list):
        """Generates a natural language response to the SQL query."""
        sql_chain = self.get_sql_chain(db)

        response_template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, SQL query, and SQL response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        Conversation History: {chat_history}
        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}
        """
        prompt = ChatPromptTemplate.from_template(response_template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=self.gemini_api_key)

        chain = (
            RunnablePassthrough.assign(query=sql_chain).assign(
                schema=lambda _: db.get_table_info(),
                response=lambda vars: db.run(vars["query"]),
            )
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            # Attempt to invoke the chain
            return chain.invoke({
                "question": user_query,
                "chat_history": chat_history,
            })
        except Exception as e:
            # Handle API-related errors or any other exceptions
            error_message = (
                "An error occurred while processing your request. "
                "This could be due to hitting the API limit or other issues. Please try again later."
            )
            st.error(error_message)  # Display the error message in the Streamlit app
            return error_message


class ChatApp:
    """Streamlit app to interact with the SQL assistant."""
    def __init__(self):
        load_dotenv()
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.db_connector = DatabaseConnector()
        self.sql_assistant = SQLAssistant(self.gemini_api_key)
        self.init_session_state()

    def init_session_state(self):
        """Initialize Streamlit session state variables."""
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                AIMessage(content="Hello! I'm SQL assistant. Ask me anything about your database."),
            ]
        if "db" not in st.session_state:
            st.session_state.db = None

    def sidebar(self):
        """Render the sidebar for database connection settings."""
        st.sidebar.subheader("Settings")
        st.sidebar.text_input("Host", value="localhost", key="Host")
        st.sidebar.text_input("Port", value="3306", key="Port")
        st.sidebar.text_input("User", value="root", key="User")
        st.sidebar.text_input("Password", type="password", value="admin", key="Password")
        st.sidebar.text_input("Database", value="chinook", key="Database")

        if st.sidebar.button("Connect"):
            with st.spinner("Connecting to database..."):
                try:
                    st.session_state.db = self.db_connector.connect(
                        st.session_state["User"],
                        st.session_state["Password"],
                        st.session_state["Host"],
                        st.session_state["Port"],
                        st.session_state["Database"]
                    )
                    st.success("Connected to the database!")
                except Exception as e:
                    st.error(f"Failed to connect: {e}")

    def display_chat_history(self):
        """Render the chat history in the app."""
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.markdown(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

    def handle_user_query(self):
        """Process user input and generate responses."""
        user_query = st.chat_input("Type a message...")
        if user_query:
            st.session_state.chat_history.append(HumanMessage(user_query))
            with st.chat_message("Human"):
                st.markdown(user_query)

            with st.chat_message("AI"):
                if st.session_state.db:
                    response = self.sql_assistant.generate_response(
                        user_query,
                        st.session_state.db,
                        st.session_state.chat_history
                    )
                    st.markdown(response)
                    st.session_state.chat_history.append(AIMessage(content=response))
                else:
                    st.error("Database not connected. Please connect to the database first.")

    def run(self):
        """Run the Streamlit app."""
        st.set_page_config(page_title="Chat with Database", page_icon=":speech_balloon:")
        st.title("Chat with Database")
        self.sidebar()
        self.display_chat_history()
        self.handle_user_query()


# Run the app
if __name__ == "__main__":
    app = ChatApp()
    app.run()
