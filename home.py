import json
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from streamlit_lottie import st_lottie_spinner
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI



def render_animation():
    path = "assets/typing_animation.json"
    with open(path,"r") as file: 
        animation_json = json.load(file) 
        return animation_json

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string


st.set_page_config(
    page_title="Softsquare AI",
    page_icon="🤖",
)

load_dotenv()

# Load Animation
typing_animation_json = render_animation()
hide_st_style = """ <style>
                    #MainMenu {visibility:hidden;}
                    footer {visibility:hidden;}
                    header {visibility:hidden;}
                    </style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown("""
    <h1 id="chat-header" style="position: fixed;
                   top: 0;
                   left: 0;
                   width: 100%;
                   text-align: center;
                   background-color: #f1f1f1;
                   z-index: 9">
        Chat with AI 💬
    </h1>
""", unsafe_allow_html=True)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'initialPageLoad' not in st.session_state:
    st.session_state['initialPageLoad'] = False

if 'selected_product_type' not in st.session_state:
    st.session_state['selected_product_type'] = 'Agrid'

if 'prevent_loading' not in st.session_state:
    st.session_state['prevent_loading'] = False

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
pinecone_index = 'agrid-document'
vector_store = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferMemory(memory_key="chat_history",
                                    max_len=50,
                                    return_messages=True,
                                    output_key='answer')

# Answer the question as truthfully as possible using the provided context, 
# and if the answer is not contained within the text below, say 'I don't know'
general_system_template = r""" 
As a dedicated Product Assistant, you are tasked with delivering detailed guidance and support to our varied user base, which includes customers, admins, developers, and managers. Your primary tools and resources include Salesforce's data model and architecture documentation, along with our product's user and admin manuals. Your role involves:
 
1. User Type Identification: Start by identifying the user type based on [USER IDENTIFICATION METHOD]. Tailor your responses to fit their specific context, enhancing the personalized support experience.
 
2. Knowledge Base Integration:
  - Dive into our product's manuals, which detail installation steps, feature explanations, and use cases on the Salesforce platform.
  - Employ keyword matching and user intent analysis for precise searches within the knowledge base.
  - Grasp the Salesforce standard object model, understanding the architecture and feature sets.
  - Analyze example use cases for insights into problem statements, configurable steps, and their solutions.
 
3. Conversation Analysis:
  - Review [conversation logs] to pinpoint keywords, error messages, and referenced features or objects.
  - Leverage this information to formulate precise queries within Salesforce and our product's documentation.
 
4. Prompting for Clarification:
  - If a user query is unclear, employ [PROMPTING STRATEGY] to gather more information or clarify their needs. A good practice is to ask questions like, “Can you specify which feature you’re using?” or “Could you describe the issue in more detail?”
 
Overall Objective: Your aim is to understand the user's issue, find solutions using the appropriate knowledge resources, and offer valuable assistance, thus resolving their concerns with our product and Salesforce, and improving their overall experience.
 
Sample User Inputs:
- Technical details or error messages for troubleshooting.
- Requirements or use cases for configuring features.
- Questions about specific product features.
 
DOs:
- Highlight the bot’s benefits briefly, such as 24/7 support and quicker problem resolution.
- Personalize responses based on the identified user type, emphasizing adaptability.
- Clarify the sources of your knowledge, reassuring users of the reliability of the information provided.
 
DON'Ts:
- Avoid overcomplication; aim for clarity and conciseness.
- Steer clear of technical jargon not understood by all user types.
 
Response Style:
- Aim for simple, human-like responses to ensure readability and clarity.
- Use short paragraphs and bullet points for easy comprehension.

If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
 ----
{context}
----
"""
general_user_template = "Question:```{question}```"

system_msg_template = SystemMessagePromptTemplate.from_template(template=general_system_template)

human_msg_template = HumanMessagePromptTemplate.from_template(template=general_user_template)
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vector_store.as_retriever(search_kwargs={'k': 2}),
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    rephrase_question = False,
    response_if_no_docs_found = "Sorry, I dont know",
    memory = st.session_state.buffer_memory,
    
)

# container for chat history
response_container = st.container()
textcontainer = st.container()


chat_history = []
with textcontainer:
    st.session_state.initialPageLoad = False
    query = st.chat_input(placeholder="Say something ... ", key="input")
    if query and query != "Menu":
        conversation_string = get_conversation_string()
        with st_lottie_spinner(typing_animation_json, height=50, width=50, speed=3, reverse=True):
            response = qa_chain({'question': query, 'chat_history': chat_history})
            chat_history.append((query, response['answer']))
            print("response:::: ",response)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response['answer'])
    st.session_state.prevent_loading = True



with response_container:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.session_state.initialPageLoad = False
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            response = f"<div style='font-size:0.875rem;line-height:1.75;white-space:normal;'>{st.session_state['responses'][i]}</div>"
            message(response,allow_html=True,key=str(i),logo=('https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/SS512X512.png'))
            if i < len(st.session_state['requests']):
                request = f"<meta name='viewport' content='width=device-width, initial-scale=1.0'><div style='font-size:.875rem'>{st.session_state['requests'][i]}</div>"
                message(request, allow_html=True,is_user=True,key=str(i)+ '_user',logo='https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/generic-user-icon-13.jpg')


