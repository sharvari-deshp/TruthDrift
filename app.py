import streamlit as st
from predict import predict_text, split_sentences


st.title("TruthDrift: AI Hallucination Detector")

text = st.text_area("Paste AI-generated paragraph:")


if st.button("Analyze"):
    
    sentences = split_sentences(text)

    
    probs = predict_text(sentences)


    for i in range(len(sentences)):
        sent = sentences[i]
        conf = probs[i]

        st.markdown(f"### Sentence {i+1}")

        if conf > 0.4:
            st.success(f"{sent}")
            st.write(f"Correct. Confidence: {conf*100:.2f}%")
        else:
            st.error(f"{sent}")
            st.write(f"Hallucinated. Confidence: {(1-conf)*100:.2f}%")