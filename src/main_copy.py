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
st.write(''' <style>
         
          button {
            box-shadow: rgba(0, 0, 0, 0.12) 0px 24px 25px, rgba(0, 0, 0, 0.12) 0px -12px 30px, rgba(0, 0, 0, 0.12) 0px 4px 6px, rgba(0, 0, 0, 0.17) 0px 12px 13px, rgba(0, 0, 0, 0.09) 0px -3px 5px;
          }
          .stTextInput input {
            border: 1px solid black;
            border-radius: 10px;
            padding: 10px;
          }         
         </style>''', unsafe_allow_html=True)

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
    {"role": "system", "content": "You are an expert at converting NLP queries to OData queries. Always work according to below points.\
                    1. Always include UOMTOID eq 'EA' in filter part of query. \
                    2. Always make query by writing PRDID for product. \
                    3. Always make query by writing LOCID for location. \
                    4. Always make query by writing PERIODID4_TSTAMP for weekly. \
                    5. Always make query by writing PERIODID1_TSTAMP for yearly. \
                    6. Always make query by writing PERIODID3_TSTAMP for monthly. \
                    7. Always make query by writing CONSTRAINEDDEMAND for Constrained Demand.\
                    8. Always make query by writing PRDFAMILY for Product Family. \
                    9. Always make query by writing PROJECTEDINVENTORY for projected inventory. \
                    10. Always make query by writing BACKORDERS for backorder or backorders. \
                    11. Always make query by writing SALESHISTORY for sales or sales history. \
                    12. Always add order by timestamp if query contains weekly, yearly or monthly."},
    {"role": "user", "content": "Give me the opening inventory of product Phone1 at location DC1"},
    {"role": "assistant", "content": "/ZZOPTIMIZE?$select=PRDID,LOCID,INITIALINVENTORY&$filter=UOMTOID eq 'EA' and LOCID eq 'DC1' and PRDID eq 'Phone1'"},
    {"role": "user", "content": "Give me the total Consensus Demand of product Phone1 at Customer Cust1"},
    {"role": "assistant", "content": "/ZZOPTIMIZE?$select=PRDID,CUSTID,CONSENSUSDEMAND&$filter=UOMTOID eq 'EA' and CUSTID eq 'Cust1' and PRDID eq 'Phone1'"},
    {"role": "user", "content": "Display Consensus Demand for Product Phone1 for the year 2025"},
    {"role": "assistant", "content": "/ZZOPTIMIZE?$select=PRDID,PERIODID1_TSTAMP,CONSENSUSDEMAND&$filter=UOMTOID eq 'EA' and PRDID eq 'Phone1' and PERIODID1_TSTAMP eq datetime'2025-01-01T00:00:00'"},
    {"role": "user", "content": "Show me the Products which have Consensus demand"},
    {"role": "assistant", "content": "/ZZOPTIMIZE?$select=PRDID,CONSENSUSDEMAND&$filter=UOMTOID eq 'EA' and CONSENSUSDEMAND gt 0"},
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
    # print (messages)
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
    
# Streamlit chatbot interface
def chatbot_interface():
    st.title("SAP IBP Smart Assistant")

    # Initialize session state for textbox
    if "recognized_text" not in st.session_state:
        st.session_state.recognized_text = ""   
    
    col1, col2 = st.columns([10, 1])  # Textbox (4 parts), Mic Icon (1 part)
    
    with col2:
        for _ in range(1):
            st.text("")
        for _ in range(1):
            st.text("")
        if st.button("🎙️"):            
            user_input = audioToText();
            st.session_state.recognized_text = user_input 
            print(user_input) 
    
    with col1:        
        user_input = st.text_input("Enter text or click Mic for voice based input", st.session_state.recognized_text)
    
    if st.button("Get Data"):
        if user_input:
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

                with st.expander("Data Retrieved"):
                    loading_placeholder = st.empty()
                    for i in range(5):
                        loading_placeholder.write(f"⏳ Loading{'.' * i}")
                        time.sleep(0.5)
                    loading_placeholder.empty()
                    st.dataframe(df, use_container_width=True)

                matching_cols_TSTAMP = [col for col in df.columns if "TSTAMP" in col]
                matching_cols_DEMAND = [col for col in df.columns if "DEMAND" in col]
                matching_cols_INVENTORY = [col for col in df.columns if "INVENTORY" in col]
                matching_cols_BACKORDERS = [col for col in df.columns if "BACKORDERS" in col]
                matching_cols_SALESHISTORY = [col for col in df.columns if "SALESHISTORY" in col]
                
                # Rename column
                if matching_cols_TSTAMP:                    
                    df.rename(columns={matching_cols_TSTAMP[0]: "TIMESTAMP"}, inplace=True)
                if matching_cols_DEMAND:                    
                    df.rename(columns={matching_cols_DEMAND[0]: "DEMAND"}, inplace=True)                    
                if matching_cols_INVENTORY:                    
                    df.rename(columns={matching_cols_INVENTORY[0]: "INVENTORY"}, inplace=True)
              
                
                if not df.empty:
                    with st.expander("Graph Generated"):
                        if (not matching_cols_TSTAMP) and matching_cols_INVENTORY: 
                            msg_str = "The inventory for "                        
                            if "PRDID" in df.columns:                            
                                x_value="PRDID"
                                product = df["PRDID"].iloc[0]
                                msg_str = msg_str+"product {} ".format(product)
                            if "PRDFAMILY" in df.columns:                            
                                x_value="PRDFAMILY"
                                product = df["PRDFAMILY"].iloc[0]
                                msg_str = msg_str+"product family {} ".format(product)
                            if "LOCID" in df.columns: 
                                location = df["LOCID"].iloc[0]
                                msg_str = msg_str+ "at location {} ".format(location) 
                            inventory = df["INVENTORY"].iloc[0]
                            msg_str=msg_str+"is {}".format(inventory)
                            fig = px.bar(
                                df,
                                x=x_value,
                                y="INVENTORY",                            
                                color_discrete_sequence=["#9EE6CF"],
                            )
                        if (matching_cols_TSTAMP) and matching_cols_INVENTORY: 
                            msg_str = "The inventory for "                        
                            if "PRDID" in df.columns:                            
                                x_value="PRDID"
                                product = df["PRDID"].iloc[0]
                                msg_str = msg_str+"product {} ".format(product)
                            if "PRDFAMILY" in df.columns:                            
                                x_value="PRDFAMILY"
                                product = df["PRDFAMILY"].iloc[0]
                                msg_str = msg_str+"product family {} ".format(product)
                            if "LOCID" in df.columns: 
                                location = df["LOCID"].iloc[0]
                                msg_str = msg_str+ "at location {} ".format(location) 
                            inventory = df["INVENTORY"].iloc[0]
                            msg_str=msg_str+"is {}".format(inventory)
                            fig = px.bar(
                                df,
                                x="TIMESTAMP",
                                y="INVENTORY",                            
                                color_discrete_sequence=["#9EE6CF"],
                            )
                        if (not matching_cols_TSTAMP) and matching_cols_DEMAND:
                            if "PRDID" in df.columns:
                                x_value="PRDID" 
                                product = df["PRDID"].iloc[0]
                            if "PRDFAMILY" in df.columns:                            
                                x_value="PRDFAMILY" 
                                product = df["PRDFAMILY"].iloc[0]       
                            df.rename(columns={"DEMAND":matching_cols_DEMAND[0]}, inplace=True) 
                            if (len(matching_cols_DEMAND)>1):                            
                                # Convert the DataFrame from wide to long format
                                df_melt = df.melt(id_vars='PRDID', value_vars=matching_cols_DEMAND,  var_name="DEMANDS", value_name="VALUE") 
                                print ("i am here")
                                fig = px.bar(
                                    df_melt,
                                    x="PRDID",
                                    y="VALUE",
                                    color="DEMANDS",                                
                                    barmode="group",                              
                                    color_discrete_sequence=["#9EE6CF", "#0e819a"],                        
                                )
                            else:
                                fig = px.bar(
                                df,
                                x=x_value,
                                y=matching_cols_DEMAND[0],                            
                                color_discrete_sequence=["#9EE6CF"],   
                                )                        
                            max_value = pd.to_numeric(df[matching_cols_DEMAND[0]]).max()
                            msg_str = "The highest demand for product {} is {}".format(product, max_value)                        
                        if (matching_cols_DEMAND and matching_cols_TSTAMP):
                            df.rename(columns={"DEMAND":matching_cols_DEMAND[0]}, inplace=True) 
                            if (len(matching_cols_DEMAND)>1):                            
                                # Convert the DataFrame from wide to long format
                                df_melt = df.melt(id_vars='TIMESTAMP', value_vars=matching_cols_DEMAND,  var_name="DEMANDS", value_name="VALUE") 
                                fig = px.bar(
                                    df_melt,
                                    x="TIMESTAMP",
                                    y="VALUE",
                                    color="DEMANDS",                                
                                    barmode="group",                                    
                                    color_discrete_sequence=["#9EE6CF", "#0e819a"],
                                )
                            else :
                                if "PRDFAMILY" in df.columns:
                                    fig = px.bar(
                                        df, 
                                        x="TIMESTAMP", 
                                        y=matching_cols_DEMAND[0], 
                                        color="PRDFAMILY",
                                        labels={"PRDFAMILY": "Product Family", matching_cols_DEMAND[0]: "Demand"},
                                        barmode="group",
                                        color_discrete_sequence=["#9EE6CF", "#0e819a"])
                                    # fig = px.bar(
                                    #     df, 
                                    #     x="PRDFAMILY", 
                                    #     y=matching_cols_DEMAND[0], 
                                    #     color="TIMESTAMP",
                                    #     labels={"PRDFAMILY": "Product Family", 
                                    #     matching_cols_DEMAND[0]: "Demand"})
                                else:
                                    fig = px.bar(
                                        df,
                                        x="TIMESTAMP",
                                        y=matching_cols_DEMAND[0],   
                                        color_discrete_sequence=["#9EE6CF"],  
                                    )  
                            if "PRDID" in df.columns:
                                no_product = df["PRDID"].count()
                            if "PRDFAMILY" in df.columns:
                                no_product = df["PRDFAMILY"].count()
                            max_value = pd.to_numeric(df[matching_cols_DEMAND[0]]).max()
                            timestamp = df.loc[pd.to_numeric(df[matching_cols_DEMAND[0]]) == max_value, 'TIMESTAMP'].iloc[0]
                            msg_str = "There are total {} records. The highest demand is: {} in {} ".format(no_product, max_value, timestamp)
                        if matching_cols_BACKORDERS: 
                            msg_str = "The backorders for "                        
                            if "PRDID" in df.columns:                            
                                x_value="PRDID"
                                product = df["PRDID"].iloc[0]
                                msg_str = msg_str+"product {} ".format(product)
                            if "TIMESTAMP" in df.columns:                            
                                x_value="TIMESTAMP"
                                product = df["TIMESTAMP"].iloc[0]
                                max_value = pd.to_numeric(df["BACKORDERS"]).max()
                                msg_str = msg_str+"heighest value is {} ".format(max_value)
                            else:
                                backorders = df["BACKORDERS"].iloc[0]
                                msg_str=msg_str+"is {}".format(backorders)
                            fig = px.bar(
                                df,
                                x=x_value,
                                y="BACKORDERS",                            
                                color_discrete_sequence=["#9EE6CF"],
                            )    
                        if matching_cols_SALESHISTORY: 
                            msg_str = "The sales history for "                        
                            if "PRDID" in df.columns:                            
                                x_value="PRDID"
                                product = df["PRDID"].iloc[0]
                                msg_str = msg_str+"product {} ".format(product)
                            if "TIMESTAMP" in df.columns:                            
                                x_value="TIMESTAMP"
                                product = df["TIMESTAMP"].iloc[0]
                                max_value = pd.to_numeric(df["SALESHISTORY"]).max()
                                msg_str = msg_str+"heighest value is {} ".format(max_value)
                            else:
                                saleshistory = df["SALESHISTORY"].iloc[0]
                                msg_str=msg_str+"is {}".format(saleshistory)
                            fig = px.bar(
                                df,
                                x=x_value,
                                y="SALESHISTORY",                            
                                color_discrete_sequence=["#9EE6CF"],
                            )       
                        fig.update_layout(dragmode="pan") 
                        st.plotly_chart(fig, theme="streamlit", use_container_width=True)
                        st.markdown(msg_str)
                # st.markdown(str_analysis)
            else:
                st.write("No data retrieved or error occurred.")            
        else:
            st.write("Please enter a query.")

if __name__ == "__main__":
    chatbot_interface()

# Example Usage
# nlp_query = "Give me the weekly Consensus Demand of product Phone1 at Customer Cust1."
# odata_query = nlp_to_odata(nlp_query)

# print(f"NLP Query: {nlp_query}")
# print(f"OData Query: {odata_query}")