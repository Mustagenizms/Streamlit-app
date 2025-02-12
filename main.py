import streamlit as st

# Set the page configuration
st.set_page_config(page_title="AI Tumor Detector", page_icon="🌍", layout="wide")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: black;
    }
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 1000;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 20px;
        background-color: white;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .sticky-header .logo {
        font-size: 24px;
        font-weight: bold;
        color: #007bff;
    }
    .sticky-header .header-buttons {
        display: flex;
        gap: 10px;
    }
    .sticky-header .header-buttons button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 8px 16px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .sticky-header .header-buttons button:hover {
        background-color: #0056b3;
    }
    .main-content {
        margin: 20px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .main-content h1 {
        font-size: 32px;
        margin: 0;
    }
    .upload-button {
        background-color: #007bff;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
    }
    .upload-button:hover {
        background-color: #0056b3;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sticky Header
st.markdown(    
    """
    <div class="sticky-header">
        <div class="logo">Logo</div>
        <div class="header-buttons">
            <button>Home</button>
            <button>About</button>
            <button>Contact</button>
            <button>Settings</button>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Space between header and main content
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Main content
st.markdown(
    """
    <div class="main-content">
        <h1>AI Tumor Detector</h1>
        <button class="upload-button">Upload Scan</button>
    </div>
    """,
    unsafe_allow_html=True
)

# Space after main content
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .content {
            background-color: #C3E5FF;
        }
        .content ul {
            
        }
    </style>
    """,
    unsafe_allow_html=True
)

# next content
st.markdown(
    """
    <div class="content">
        <ul>
            <a class="content-1"><li>About</li></a>
            <a class="content-2"><li>Contact</li></a>
            <a class="content-3"><li>shdbf</li></a>
            <a class="content-4"><li>srag</li></a>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)
