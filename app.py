import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Q&A Chatbot with Groq",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided chat history to maintain context and provide relevant answers."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

# Output parser (stateless, reusable across calls)
output_parser = StrOutputParser()

def generate_response(question, chat_history, model_name, temperature, max_tokens, api_key):
    """
    Generate response from Groq LLM
    
    Args:
        question: User's question
        chat_history: List of previous messages
        model_name: Selected Groq model
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        api_key: Groq API key
        
    Returns:
        Generated response or error message
    """
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        chain = prompt | llm | output_parser
        answer = chain.invoke({
            'question': question,
            'chat_history': chat_history
        })
        return answer, None
        
    except Exception as e:
        return None, str(e)

def main():
    """Main application function"""
    
    # Title and description
    st.title("🤖 GroqMind")
    st.markdown("Groq's lightning-fast LLM inference.")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key handling
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            groq_api_key = st.text_input(
                "Enter your Groq API Key:",
                type="password",
                help="Get your API key from https://console.groq.com"
            )
            if groq_api_key:
                st.success("✅ API Key entered!")
            else:
                st.warning("⚠️ Please enter your Groq API Key to continue")
                st.info("Don't have an API key? Get one at [Groq Console](https://console.groq.com)")
        else:
            st.success("✅ API Key loaded from environment!")
            # Show masked API key
            masked_key = f"{groq_api_key[:8]}...{groq_api_key[-4:]}"
            st.code(masked_key)
        
        st.divider()
        
        # Model Selection
        st.subheader("🎯 Model Settings")
        model_name = st.selectbox(
            "Select Groq Model",
            [
                "llama-3.3-70b-versatile",
                "llama-3.1-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it"
            ],
            help="Choose the LLM model for generating responses"
        )
        
        # Model info
        model_info = {
            "llama-3.3-70b-versatile": "Most capable, best for complex tasks",
            "llama-3.1-70b-versatile": "Balanced performance and speed",
            "llama-3.1-8b-instant": "Fast responses, good for simple queries",
            "mixtral-8x7b-32768": "Large context window (32K tokens)",
            "gemma2-9b-it": "Efficient and instruction-tuned"
        }
        st.caption(model_info.get(model_name, ""))
        
        st.divider()
        
        # Parameters
        st.subheader("🎛️ Response Parameters")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make output more random, lower values more focused"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=2048,
            value=512,
            step=50,
            help="Maximum length of the response"
        )
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
        
        # About section
        with st.expander("ℹ️ About"):
            st.markdown("""
            **Q&A Chatbot with Groq**
            
            This chatbot uses Groq's fast LLM inference to provide quick and accurate responses.
            
            **Features:**
            - Multiple LLM models
            - Adjustable parameters
            - Chat history
            - Fast response times
            
            **Tech Stack:**
            - Streamlit
            - LangChain
            - Groq API
            """)
    
    # Main Chat Interface
    st.divider()
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for idx, (question, answer) in enumerate(st.session_state.chat_history):
            if idx > 0:
                st.divider()
            with st.container():
                st.info(f"**You:** {question}")
                st.success(f"**Bot:** {answer}")
    
    # User input section
    st.subheader("How Can I Help You?")
    
    
    # Use form to handle enter key submission
    with st.form(key="question_form", clear_on_submit=True):
        user_input = st.text_input(
            "Your Question:",
            placeholder="Type your question here...",
            label_visibility="collapsed"
        )
        submit_button = st.form_submit_button(" Send", use_container_width=True)
    
    # Process user input
    if submit_button and user_input:
        if not groq_api_key:
            st.error("❌ Please enter your Groq API Key in the sidebar to continue.")
        else:
            with st.spinner(" Thinking..."):
                # Format chat history for LangChain (limit to last 10 messages for context)
                formatted_history = []
                for q, a in st.session_state.chat_history[-10:]:
                    formatted_history.append(HumanMessage(content=q))
                    formatted_history.append(AIMessage(content=a))
                
                response, error = generate_response(
                    user_input,
                    formatted_history,
                    model_name,
                    temperature,
                    max_tokens,
                    groq_api_key
                )
                
                if error:
                    st.error(f"❌ Error: {error}")
                    if "api_key" in error.lower() or "authentication" in error.lower():
                        st.info("💡 Please check your API key and try again.")
                else:
                    # Add to chat history
                    st.session_state.chat_history.append((user_input, response))
                    st.rerun()
    
    # Footer
    st.divider()
    st.caption("Powered by Groq  | Built with Streamlit ")

if __name__ == "__main__":
    main()
