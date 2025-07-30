import os
import re
import requests
import time
from requests.auth import HTTPBasicAuth
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI
import plotly.express as px
from VoiceToText import VoiceToText as vtt
# Configure page
st.set_page_config(
    page_title="SAP Agentic AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 1rem;
    }
    .main .block-container {
            max-width: 500px;
            margin: auto;
            padding-top: 1rem;
            padding-bottom: 1rem;
    }
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #0F4C75 0%, #3282B8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        margin: 0.5rem 0 0 0;
        text-align: center;
        opacity: 0.9;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #0F4C75 0%, #3282B8 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        border: 2px solid #E1E5E9;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #3282B8;
        box-shadow: 0 0 0 3px rgba(50, 130, 184, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8F9FA;
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #3282B8;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #F8F9FA;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Loading animation */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    .loading-spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3282B8;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Chat message styling */
    .chat-message {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    
    .assistant-message {
        background-color: #F1F8E9;
        border-left: 4px solid #4CAF50;
    }
    
    /* Professional table styling */
    .dataframe {
        border: 1px solid #E1E5E9;
        border-radius: 8px;
        overflow: hidden;
    }
    
    .dataframe th {
        background-color: #F8F9FA;
        font-weight: 600;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)
# st.write(''' <style>
         
#           button {
#             box-shadow: rgba(0, 0, 0, 0.12) 0px 24px 25px, rgba(0, 0, 0, 0.12) 0px -12px 30px, rgba(0, 0, 0, 0.12) 0px 4px 6px, rgba(0, 0, 0, 0.17) 0px 12px 13px, rgba(0, 0, 0, 0.09) 0px -3px 5px;
#           }
#           .stTextInput input {
#             border: 1px solid black;
#             border-radius: 10px;
#             padding: 10px;
#           }         
#          </style>''', unsafe_allow_html=True)

# Azure OpenAI Configuration
AZURE_OPENAI_KEY = st.secrets['AZURE_OPENAI_API_KEY']
AZURE_OPENAI_VERSION = st.secrets['AZURE_OPENAI_API_VERSION']
AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_API_BASE"]
AZURE_MODEL_NAME = st.secrets["AZURE_MODEL_NAME"]

client = AzureOpenAI(
    api_key = AZURE_OPENAI_KEY, 
    api_version = AZURE_OPENAI_VERSION,
    azure_endpoint = AZURE_OPENAI_ENDPOINT
)

# Initialize the message history with a system message
messages = [
    {"role": "system", "content": "You are an expert at converting NLP queries to OData queries for SAP systems.\n"
                    "Strictly follow these rules:\n"
                    "1. Always make query by writing MfgOrderItemPlannedTotalQty for planned quantity. \n"
                    "2. Always make query by writing MfgOrderItemActualDeviationQty for delivered quantity. \n"
                    "3. Always include both `$select` and `$filter` clauses in the output.\n"
                    "4. Use `ManufacturingOrder` in `$filter` for order numbers.\n"                    
                    "5. Return only the OData query without any explanation or description.\n"},
    {"role": "user", "content": "Give me the details for order number 1000002"},
    {"role": "assistant", "content": "$select=ManufacturingOrder,MfgOrderItemPlannedTotalQty,MfgOrderItemActualDeviationQty&$filter=ManufacturingOrder eq '1000002'"}    
]

# Function to convert NL to OData query
def nlp_to_odata(nlp_query):
    """Convert a natural language query into an OData query."""
    prompt = f"""
    Convert the following natural language query into an OData query:

    Natural Language Query: "{nlp_query}"
    
    Provide only the OData query as output.
    """
    # Append the user's input to the message history
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(    
       model = AZURE_MODEL_NAME,
       messages = messages
    )
    assistant_message= response.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message})    
    return response.choices[0].message.content

# Function to make API call using OData query and get the response
def fetch_data_from_backend(odata_query, backend_url, username, password,requestheaders):
    # Construct the full URL
    full_url = f"{backend_url}{odata_query}"
    with st.expander("Query Generated"):
        st.write(full_url)
    

    # Make the GET request to the OData API
    response = requests.get(full_url, auth=HTTPBasicAuth(username, password), headers=requestheaders)  # Basic Auth
    
    # Check if the response is successful (status code 200)
    if response.status_code == 200:
        return response.json()  # Return the data in JSON format
    else:
        return {"error": f"Failed to fetch data. Status code: {response.status_code}"}

# Function to convert Microsoft JSON date format to datetime
def convert_microsoft_date(date_str):
    match = re.search(r'\d+', date_str)  # Extract digits (timestamp in milliseconds)
    return pd.to_datetime(int(match.group()), unit='ms') if match else None

# Function to convert audio to text
def audioToText():
    while True:
        # Record audio
        frames, sample_rate = vtt.record_audio()

        # Save audio to temporary file
        temp_audio_file = vtt.save_audio(frames, sample_rate)

        # Transcribe audio
        print("Transcribing...")
        text = vtt.transcribe_audio(temp_audio_file)

        # Clean up temporary file
        os.unlink(temp_audio_file)

        print("\nReady for next recording. Press PAUSE to start.")
        return text
    
# function to generate the analysis summary    
def generate_analysis_summary(df: pd.DataFrame) -> str:
    # Reduce the size of the DataFrame content to avoid token overflow
    summary_stats = df.describe().round(2).to_string()
    prompt = f"""
        You are a data analyst. Based on the following summary statistics from a dataset, write a clear, concise, and professional analysis of the data, pointing out any trends, anomalies, or interesting insights.

        Summary Statistics:
        {summary_stats}

        Write the summary in 4-5 bullet points.
            """

    response = client.chat.completions.create(
        model=AZURE_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an expert data analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.5,
        max_tokens=300
    )
    return response.choices[0].message.content.strip()

# Streamlit chatbot interface
def chatbot_interface():
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">🤖 SAP Agentic AI Assistant</h1>
        <p class="header-subtitle">Enterprise-Grade Natural Language Interface for SAP Systems</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state for textbox
    if "recognized_text" not in st.session_state:
         st.session_state.recognized_text = False   
    # selected_option = st.radio(
    # label='Choose an option:',
    # options=['Order Item', 'Order Component'],
    # horizontal=True
    # )
    col1, col2 = st.columns([8, 2])  # Textbox (4 parts), Mic Icon (1 part)
    
    # with col2:
    #     for _ in range(1):
    #         st.text("")
    #     for _ in range(1):
    #         st.text("")
    #     if st.button("🎙️"):            
    #         user_input = audioToText();
    #         st.session_state.recognized_text = user_input 
            
    
    with col1:        
        user_input = st.text_input("Enter your query",
            placeholder="Enter your query e.g., Show me details for order 1000002",
            label_visibility="collapsed")
    with col2:
        if st.button("Get Data"):
            st.session_state.recognized_text=True
    if st.session_state.recognized_text and user_input:
                # Convert the user's query into an OData query
                odata_query = nlp_to_odata(user_input)                                    
                
                backend_url = st.secrets['BACKEND_URL']
                username = st.secrets['USER_NAME']
                password = st.secrets['PASS_WORD']
                
                headers = {
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                }            
                # Fetch data from the backend
                data = fetch_data_from_backend(odata_query, backend_url, username, password, headers)
            
                if data and "d" in data:
                    df = pd.DataFrame(data["d"].get("results", []))  # Convert to DataFrame
                    df=df.drop(columns=["__metadata"],errors="ignore")
                    # The timestamp format "/Date(1738540800000)/" is a Microsoft JSON date format, which represents milliseconds since Unix epoch (January 1, 1970).
                    # Apply conversion only to columns containing timestamp format
                    for col in df.select_dtypes(include=['object']).columns:  # Check only string columns
                        if df[col].str.startswith("/Date").all():  # Ensure the column contains timestamp format
                            df[col] = df[col].apply(convert_microsoft_date)

                    # Change index to start from 1
                    df.index = range(1, len(df) + 1)
                    df.index.name = "S No"
                    matching_cols_OrderNumber = [col for col in df.columns if "ManufacturingOrder" in col]
                    matching_cols_PlannedQty = [col for col in df.columns if "MfgOrderItemPlannedTotalQty" in col]
                    matching_cols_DeliveredQty = [col for col in df.columns if "MfgOrderItemActualDeviationQty" in col]
                    
                    
                    # Rename column
                    if matching_cols_OrderNumber:                    
                        df.rename(columns={matching_cols_OrderNumber[0]: "Order Number"}, inplace=True)
                    if matching_cols_PlannedQty:                    
                        df.rename(columns={matching_cols_PlannedQty[0]: "Planned Qty"}, inplace=True)
                    if matching_cols_DeliveredQty:                    
                        df.rename(columns={matching_cols_DeliveredQty[0]: "Delivered Qty"}, inplace=True)
                    
                    with st.expander("Data Retrieved"):
                        loading_placeholder = st.empty()
                        for i in range(5):
                            loading_placeholder.write(f"⏳ Loading{'.' * i}")
                            time.sleep(0.5)
                        loading_placeholder.empty()
                        st.dataframe(df, use_container_width=True)
                    
                    
                        
                    matching_cols_QTY = [col for col in df.columns if "Qty" in col]
                    if matching_cols_QTY:                    
                        df.rename(columns={matching_cols_QTY[0]: "Qty"}, inplace=True)  
                    if not df.empty:
                        with st.expander("Graph Generated"):
                            df.rename(columns={"Qty":matching_cols_QTY[0]}, inplace=True)
                            df["Order Number"] = df["Order Number"].astype(str)
                            if (len(matching_cols_QTY)>1):                            
                                df_melt = df.melt(id_vars='Order Number', value_vars=matching_cols_QTY,  var_name="Qty", value_name="VALUE") 
                                fig = px.bar(
                                    df_melt,
                                    x="Order Number",
                                    y="VALUE",
                                    color="Qty",                                
                                    barmode="group",                                    
                                    color_discrete_sequence=["#9EE6CF", "#0e819a"],
                                )                   
                            fig.update_layout(dragmode="pan",
                                xaxis=dict(
                                    type="category",
                                    categoryorder="array",
                                    categoryarray=df_melt["Order Number"].unique().tolist()
                                )) 
                            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                        with st.expander("AI-Generated Analysis Summary"):
                            try:
                                summary_text = generate_analysis_summary(df)
                                st.markdown(summary_text)
                            except Exception as e:
                                st.error(f"Error generating summary: {e}")    
                else:
                    st.write("No data retrieved or error occurred.")            
    

if __name__ == "__main__":
    chatbot_interface()

