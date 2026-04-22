import streamlit as st
from predict import predict_text, split_sentences

#set page config - better layout
st.set_page_config(
    page_title="❃ TruthDrift",
    layout="centered"
)

#style in css
st.markdown("""
<style>
            
    
    [data-testid="stHeader"] {
        background-color: rgba(217, 69, 69, 1); 
    }
            
    .stApp {
        background-color: rgba(255, 240, 240, 1);
    }
            
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        font-weight: bold;
        background-color: rgba(217, 69, 69, 1); 
    }
    .stTextArea textarea {
        border-radius: 8px;
    }
            
    .stButton>button:hover {
        background-color: rgba(217, 69, 69, 1);
        color: white;
    }

    .stTextArea textarea {
        border-radius: 8px;
        background-color: rgba(217, 69, 69, 0.15);  /* greyish red */
        color: black;
    }

    .stTextArea textarea:focus {
        background-color: rgba(217, 69, 69, 0.2) !important;
        outline: none;
        box-shadow: none;
    }

    .stTextArea textarea::placeholder {
        color: rgba(0, 0, 0, 0.5);
    }
            

    .stExpander > details > summary {
        background-color: rgba(217, 69, 69, 0.15);
        border-radius: 8px;
        padding: 0.5em;
        color: black;
    }


            

    .stExpander > details > summary:hover {
        background-color: rgba(217, 69, 69, 0.15);
        color: black;
    }


    .stExpander > details[open] > summary {
        background-color: rgba(217, 69, 69, 0.2);
}
</style>
""", unsafe_allow_html=True)



# Header Section
st.title("❃ TruthDrift")
st.markdown("Analyze AI-generated text to identify potentially hallucinated or incorrect statements.")
st.divider()

#input ai generated text
text = st.text_area("Paste some AI-generated text:", height=200, placeholder="Enter text...")

#analysis section
if st.button("Analyze Text"):
    if not text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing statements..."):
            sentences = split_sentences(text)
            probs = predict_text(sentences)

        st.divider()
        st.subheader("Analysis")
        
        #overal statistics display
        correct_count = sum(1 for p in probs if p > 0.4)
        total_sentences = len(sentences)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Sentences", total_sentences)
        with col2:
            st.metric("Factually Correct", correct_count)
        with col3:
            st.metric("Potential Hallucinations", total_sentences - correct_count)
            
        st.markdown("### Detailed Breakdown")
        
        for i, (sent, conf) in enumerate(zip(sentences, probs)):
            #sentence expander for more user control 
            with st.expander(f"Sentence {i+1}: {sent[:60]}{'...' if len(sent) > 60 else ''}", expanded=True):
                if conf > 0.4:
                    st.success(f"{sent}")
                    st.markdown(f"**Status:** Correct  \n**Confidence:** `{conf*100:.2f}%`")
                else:
                    st.error(f"**{sent}")
                    st.markdown(f"**Potential Status:** Hallucination  \n**Confidence:** `{(1-conf)*100:.2f}%`")