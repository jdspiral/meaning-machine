import os
from pathlib import Path

os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["HF_METRICS_OFFLINE"]     = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
os.environ["THINC_BACKEND"]          = "numpy"

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import torch
from transformers import DistilBertTokenizerFast, DistilBertModel
import spacy
from spacy import displacy
import gc

st.set_page_config(page_title="Tokenizer", layout="centered")

@st.cache_resource(show_spinner=False)
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource(show_spinner=False)
def load_bert():
    local_dir = Path(__file__).resolve().parent / "models" / "distilbert-base-uncased"
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Model folder not found: {local_dir}")
    if not (local_dir / "config.json").is_file():
        raise FileNotFoundError(f"config.json missing in: {local_dir}")

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        str(local_dir),
        local_files_only=True
    )
    tokenizer.model_max_length = 512

    model = DistilBertModel.from_pretrained(
        str(local_dir),
        local_files_only=True
    )
    model.eval()
    return tokenizer, model

nlp       = load_spacy()
tokenizer, model = load_bert()

st.title("Tokenizer")
st.markdown("##### Visualize how machines tokenize, embed, and parse your language.")

with st.form("input_form"):
    user_input = st.text_area(
        "Enter a sentence:",
        "The young student didn't submit the final report on time.",
        height=120,
    )
    submitted = st.form_submit_button("Submit")

def extract_svo(doc):
    svos = []
    for token in doc:
        if token.dep_ in ("ROOT","ccomp","xcomp") and token.pos_ == "VERB":
            subj = [w for w in token.lefts if w.dep_ in ("nsubj","nsubjpass")]
            obj  = [w for w in token.rights if w.dep_ in ("dobj","attr","oprd")]
            if subj and obj:
                svos.append((subj[0].text, token.text, obj[0].text))
    return svos

if user_input:
    st.markdown("## Tokenization (BERT-style)")
    encoded = tokenizer(
        user_input,
        return_tensors="pt",
        truncation=True,
        max_length=tokenizer.model_max_length,
        add_special_tokens=False,
    )
    token_ids = encoded["input_ids"][0].cpu().tolist()
    tokens    = tokenizer.convert_ids_to_tokens(token_ids)
    df_tokens = pd.DataFrame({
        "Index": list(range(len(tokens))),
        "Token": tokens,
        "Token ID": token_ids
    })
    st.code(" | ".join(tokens))
    st.table(df_tokens)

    st.markdown("## Subject–Verb–Object (SVO) Triples")
    doc  = nlp(user_input)
    svos = extract_svo(doc)
    if svos:
        st.table(pd.DataFrame(svos, columns=["Subject","Verb","Object"]))
    else:
        st.warning("No SVO triples detected. Try: 'The dog chased the cat.'")

    st.markdown("## POS Tagging (spaCy)")
    df_pos = pd.DataFrame([
        (t.i, t.text, t.pos_, t.dep_, t.head.text) for t in doc
    ], columns=["Index","Token","POS","Dependency","Head"])
    st.table(df_pos)

    st.markdown("## Embedding Visualization (DistilBERT)")
    with torch.no_grad():
        outputs = model(**encoded)
    embeddings = outputs.last_hidden_state[0].cpu().numpy()
    coords     = PCA(n_components=2).fit_transform(embeddings)
    pos_labels = []
    for tok in doc:
        subtoks = tokenizer.tokenize(tok.text)
        pos_labels.extend([tok.pos_] * len(subtoks))
    n = min(len(tokens), coords.shape[0], len(pos_labels))
    df_plot = pd.DataFrame({
        "x":     coords[:n,0],
        "y":     coords[:n,1],
        "token": tokens[:n],
        "type":  pos_labels[:n],
    })
    fig = px.scatter(df_plot, x="x", y="y", text="token", color="type",
                     title="Token Embeddings PCA")
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

    del outputs, embeddings, coords
    torch.cuda.empty_cache()
    gc.collect()

    st.markdown("## Dependency Structure")
    html = displacy.render(doc, style="dep", page=False)
    components.html(html, scrolling=True, height=480)

    st.markdown("## Dependency Role Glossary")
    with st.expander("Tap to expand", expanded=True):
        st.markdown("""
| **Label** | **Name**               | **Description**                                    |
|-----------|------------------------|----------------------------------------------------|
| `nsubj`   | Nominal Subject        | The noun performing the main action                |
| `aux`     | Auxiliary Verb         | A helper verb                                      |
| `neg`     | Negation Modifier      | Negates another word (e.g. “n't”)                  |
| `ROOT`    | Root                   | The sentence’s main predicate                     |
| `advmod`  | Adverbial Modifier     | Modifies a verb or clause (e.g. “on time”)         |
| `prep`    | Prepositional Modifier | Introduces a prepositional phrase (e.g. “on”)      |
| `pobj`    | Object of Prep.        | The noun in a prepositional phrase (e.g. “time”)   |
| `det`     | Determiner              | Introduces or limits a noun (e.g. “The”)          |
| `attr`    | Attribute               | Describes subject after a linking verb (e.g. “is”) |
| `amod`    | Adjectival Modifier     | An adjective modifying a noun (e.g. “young”)      |
| `compound`| Compound Modifier       | A noun modifying another noun (e.g. “final”)      |
| `cc`      | Coordinating Conj.      | Connects equal elements (e.g. “and,” “but”)       |
| `conj`    | Conjunct                | Second element joined by a CC                     |
| `ccomp`   | Clausal Complement      | Clause as object (e.g. “I think that he left”)    |
| `xcomp`   | Open Clausal Complement | Clause without its own subject (e.g. “to leave”)  |
| `mark`    | Marker                  | Introduces a subordinate clause (e.g. “that”)     |
""" )
        st.caption("Aligned with: “The young student didn't submit the final report on time.”")

st.markdown("---")

st.markdown("### 🔗 Follow & Deploy")
st.markdown("""
- 🔗 [Read Explainers](https://theperformanceage.com/s/explainers)  
- 💻 [GitHub](https://github.com/jdspiral/tokenizer)  
- ✖️ [X](https://twitter.com/joshdhathcock)  
- 📸 [Instagram](https://instagram.com/joshdhathcock)  
""")
st.caption("Built to think out loud with code by Josh Hathcock @ The Performance Age")
