import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchResults, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import WikipediaQueryRun
from langchain_experimental.tools import PythonREPLTool

# Initialize tools
def setup_tools():
    return [
        DuckDuckGoSearchResults(),
        WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=3)),
        ArxivQueryRun(),
        PythonREPLTool()
    ]

# Streamlit app configuration
st.set_page_config(page_title="Research Agent", layout="wide")
st.title("🔬 Multi-Tool Research Agent")
st.markdown("Integrated tools: Web Search, Wikipedia, Arxiv, Python REPL")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    openai_key = st.text_input("OpenAI API Key:", type="password")
    selected_tools = st.multiselect(
        "Enable Tools:",
        ["Web Search", "Wikipedia", "Arxiv", "Python REPL"],
        default=["Web Search", "Wikipedia"]
    )

# Agent initialization
@st.cache_resource
def create_agent(_tools, _llm):
    return initialize_agent(
        tools=_tools,
        llm=_llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

# Main interface
query = st.text_area("Enter your query:", height=150)
run_button = st.button("Execute")

if run_button and query:
    if not openai_key:
        st.error("⚠️ OpenAI API key is required!")
        st.stop()

    try:
        os.environ["OPENAI_API_KEY"] = openai_key
        llm = OpenAI(temperature=0.3, max_tokens=1000)
        
        # Filter tools based on selection
        all_tools = setup_tools()
        tool_mapping = {
            "Web Search": "DuckDuckGoSearchResults",
            "Wikipedia": "WikipediaQueryRun",
            "Arxiv": "ArxivQueryRun",
            "Python REPL": "Python_REPL"
        }
        
        enabled_tools = [
            t
            for selected_tool in selected_tools
            for t in all_tools
            if tool_mapping[selected_tool] in t.name
        ]
        
        if not enabled_tools:
            st.error("🚨 No tools selected! Please enable at least one tool.")
            st.stop()

        agent = create_agent(enabled_tools, llm)
        
        with st.spinner("🔍 Processing your query..."):
            response = agent.run(query)
            st.subheader("Final Answer")
            st.success(response)
            
            st.subheader("Execution Details")
            st.code(agent.agent.llm_chain.prompt.template)

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# Example queries
with st.expander("💡 Example Queries"):
    st.markdown("""
    - "Explain quantum computing basics using Wikipedia and recent arXiv papers"
    - "Calculate prime numbers up to 100 using Python and explain the math"
    - "Compare Wikipedia's entries on AI and machine learning"
    - "Find recent arXiv papers about neural networks and summarize them"
    """)

# Security warning
st.markdown("""
---
**⚠️ Security Note:**  
- Python REPL executes real code - use cautiously
- Avoid enabling Python REPL in public deployments
- Review [Security Guidelines](https://github.com/langchain-ai/langchain/blob/master/SECURITY.md)
""")