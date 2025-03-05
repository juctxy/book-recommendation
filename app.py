import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity

from deep_translator import GoogleTranslator

import os
import requests
import zipfile
from datasets import load_dataset
import base64
from PIL import Image


# Load CSV dataset
dataset_url = 'https://raw.githubusercontent.com/juctxy/book-recommendation/main/novel.csv'
dataset = load_dataset('csv', data_files=dataset_url)
df = pd.DataFrame(dataset['train'])
df = df[df["Summary"].notnull()].reset_index(drop=True)

descriptions = df["Summary"].tolist()
desc_samples = [str(text) for text in descriptions]

model = SentenceTransformer("all-MiniLM-L6-v2")

desc_embeddings = model.encode(desc_samples)

ranks = df["Rank"].tolist()
max_rank = max(ranks)

# Paths
zip_url = "https://github.com/juctxy/book-recommendation/raw/main/book_illustrations.zip"  # Path to the ZIP file
zip_path = "book_illustrations.zip"  # Local path to save the ZIP file
image_folder = "book_illustrations"  # Folder to extract images

# Download the ZIP file
response = requests.get(zip_url)
with open(zip_path, 'wb') as file:
    file.write(response.content)

# Unzip if not already extracted
if not os.path.exists(image_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(image_folder)


# Function to load images from local storage
def get_local_image(title):
    filename = f"{title.replace(' ', '_').replace('/', '_')}.webp"
    image_path = os.path.join(image_folder, filename)

    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_str = base64.b64encode(img_file.read()).decode("utf-8")
        return f"data:image/webp;base64,{img_str}"
    else:
        return None  # If image is missing



# Function to calculate rank score
def calculate_rank_score(rank, max_rank):
    return 1 - (rank / max_rank)  # Normalized rank score

    
def default_top_books(language="English"):
    top_10_ranked_indices = df.nsmallest(10, 'Rank').index
    return "Some popular Novels", generate_html(top_10_ranked_indices, include_defaults=False, language=language)



def recommend_books(query, selected_categories, language="English"):
    if not query.strip():
        return default_top_books()
    translated_query = GoogleTranslator(source="vi", target="en").translate(query)
    # Filter books by selected categories
    if selected_categories:
        filtered_df = df[df['Categories'].apply(lambda x: any(cat in x.split(',') for cat in selected_categories))]
    else:
        filtered_df = df

    if filtered_df.empty:
        return "No books found with the selected categories."

    # Encode query and compute cosine similarities
    query_embedding = model.encode([translated_query])
    filtered_desc_embeddings = desc_embeddings[filtered_df.index]
    similarities = cosine_similarity(query_embedding, filtered_desc_embeddings)[0]

    # Get indices of top 10 similar books
    top_10_indices = filtered_df.index[np.argsort(similarities)[::-1][:10]]

    weighted_results = []
    for i in top_10_indices:
        sim_score = similarities[filtered_df.index.get_loc(i)]
        rank_score = calculate_rank_score(df.loc[i, 'Rank'], df['Rank'].max())
        final_score = (0.7 * sim_score) + (0.3 * rank_score)
        if final_score >= 0.4:
            weighted_results.append((i, final_score))

    # Sort by final weighted score
    weighted_results.sort(key=lambda x: x[1], reverse=True)
    for idx, score in weighted_results:
        print(f"Book: {df.loc[idx, 'Title']}, Final Score: {score}")
    selected_indices = [idx for idx, _ in weighted_results]

    return "Some novels you may like", generate_html(selected_indices, include_defaults=False, language=language)


def generate_html(selected_indices, include_defaults, language="English"):
    result_html = """
    <style>
    .novel-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        justify-content: center;
        max-width: 1300px; /* Adjust max-width for 5 cards per row */
        margin: 0 auto;
    }
    .novel-card {
        border: 1px solid #000;
        padding: 10px;
        border-radius: 5px;
        background-color: #333;
        color: #fff;
        width: calc(20% - 20px);
        text-align: center;
        cursor: pointer;
    }
    .novel-card h3 {
        font-size: 16px;
        margin-bottom: 5px;
        color: #fff;
    }
    .novel-card p {
        font-size: 12px;
        color: #ccc;
    }
    .novel-card img {
        width: 100%;
        height: auto;
        object-fit: cover;
        border-radius: 5px;
    }
    @media (max-width: 768px) {
        .novel-card {
            width: calc(50% - 10px);
        }
    }
    @media (max-width: 480px) {
        .novel-card {
            width: calc(100% - 10px);
            height: auto;
        }
    }
    </style>
    <div class="novel-container">
    """

    translator = GoogleTranslator(source="en", target="vi")

    for idx in selected_indices:
        row = df.loc[idx]
        title = row["Title"]
        author = row["Author"]
        summary = row["Summary"].replace("'", "\\'").replace("\n", "<br>")
        if language == "Vietnamese":
            summary = translator.translate(summary)  # Translate summary to Vietnamese
        rating = row["Rating"]
        rank = row["Rank"]
        chapters = row["Chapters"]
        img_data = get_local_image(title)
        if not img_data:
            continue

        result_html += f"""
        <div class="novel-card" onclick="(function(){{
            if(document.querySelector('.modal-overlay')) {{
                return;
            }}
            var d = document.getElementById('summary{idx}');
            if(d){{
                var overlay = document.createElement('div');
                overlay.className = 'modal-overlay';
                overlay.style.position = 'fixed';
                overlay.style.top = '0';
                overlay.style.left = '0';
                overlay.style.width = '100%';
                overlay.style.height = '100%';
                overlay.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
                overlay.style.zIndex = '999';

                overlay.onclick = function(event) {{
                    if (event.target === overlay) {{
                        overlay.parentNode.removeChild(overlay);
                    }}
                }};

                var m = document.createElement('div');
                m.className = 'modal-box';
                m.style.position = 'fixed';
                m.style.top = '50%';
                m.style.left = '50%';
                m.style.transform = 'translate(-50%, -50%)';
                m.style.padding = '20px';
                m.style.backgroundColor = '#333';
                m.style.borderRadius = '8px';
                m.style.maxWidth = '500px';
                m.style.width = '80%';
                m.style.boxShadow = '0 4px 8px rgba(0,0,0,0.2)';
                m.style.overflow = 'auto';
                m.innerHTML = d.innerHTML;
                m.style.color = '#fff';

                var closeButton = document.createElement('button');
                closeButton.innerText = '✕';
                closeButton.style.position = 'absolute';
                closeButton.style.top = '10px';
                closeButton.style.right = '10px';
                closeButton.style.background = 'transparent';
                closeButton.style.border = 'none';
                closeButton.style.fontSize = '20px';
                closeButton.style.cursor = 'pointer';
                closeButton.onclick = function(){{
                    overlay.parentNode.removeChild(overlay);
                }};
                m.appendChild(closeButton);
                overlay.appendChild(m);
                document.body.appendChild(overlay);
            }}
        }})()">
            <img src="{img_data}" alt="{title}">
            <h3 style="font-size:20px; margin-bottom:5px; color:#fff;">{title}</h3>
            <p style="color: white;font-size:16px;"><strong style="color: white;">Author:</strong> {author}<br>
                <strong style="color: white;">Rating:</strong> {rating}<br>
                <strong style="color: white;">Rank:</strong> {rank}<br>
                <strong style="color: white;">Chapters:</strong> {chapters}</p>
            <details id="summary{idx}" style="margin-top:5px; display:none;">
                <summary style="color:#fff;"><strong>Summary</strong></summary>
                <p style="margin-top:5px; color:#ccc;">{summary}</p>
            </details>
        </div>
        """

    result_html += "</div>"
    return result_html


with gr.Blocks(css="""
    .gradio-container {
        background-color: black !important;
        color: white !important;
    }
    .gradio-container a {
        color: white !important;
    }
/* Target all possible .gr-title containers */
    .gradio-container .gr-title,
    .gradio-container [class*="svelte-"] .gr-title {
        color: white !important;
        text-align: center !important;
        font-size: 26px !important;
        font-family: 'Source Sans Pro', sans-serif !important;
    }
    
    /* Force styles to children elements */
    .gradio-container .gr-title h3,
    .gradio-container .gr-title span,
    .gradio-container .gr-title a {
        color: inherit !important;
        font-size: inherit !important;
        font-family: inherit !important;
        text-decoration: none;
    }
    
    /* Specific footer styling */
    .gradio-container .gr-title[style*="26px"] {
        font-size: 26px !important;
        margin-top: 20px;
    }
    /* Fix footer color */
    .gradio-container .gr-title .prose,
    .gradio-container .gr-title .prose * {
        color: white !important;
    }
    
    /* Force underline for links */
    .gradio-container .gr-title a {
        text-decoration: underline !important;
    }
    
    /* Override Gradio's last-child margin */
    .gradio-container .gr-title .prose :last-child {
        margin-bottom: 0 !important;
        color: white !important;
    }
    .gr-row,
    .gr-row * {
        background-color: black !important;
        color: white !important;
        outline: none !important;
        box-shadow: none !important;
    }
    .gr-checkboxgroup, .gr-checkboxgroup * {
        background-color: black;
        color: white;
    }
    .gr-checkboxgroup label {
        background-color: black;
        color: white;
    }
    .gr-checkboxgroup input[type="checkbox"] {
        background-color: black;
        color: white;
        border: 1px solid white;
    }
    .gr-button {
        background-color: black !important;
        color: white !important;
        border: 1px solid white !important;
        cursor: pointer !important;
    }
    .gr-button:hover {
        background-color: #222 !important;
    }
    /* Remove all focus outlines */
    input:focus,
    textarea:focus,
    select:focus,
    button:focus {
        outline: none !important;
        box-shadow: none !important;
        border-color: white !important;
    }
""") as demo:
    title_state = gr.State("Some popular Novel")

    gr.Markdown(
        "### Huy's Brilliant Library: Web Novel Corner",
        elem_classes="gr-title"
    )

    # Query input
    query_input = gr.Textbox(
        lines=1,
        placeholder="Enter your book query...",
        label="Query",
        elem_classes="gr-row"
    )

    split_categories = df['Categories'].apply(lambda x: x.split(',') if isinstance(x, str) else []).explode()
    unique_categories = split_categories.str.strip().unique()
    unique_categories = sorted(unique_categories)
    category_filter = gr.CheckboxGroup(
        choices=unique_categories,
        label="Select Categories",
        elem_classes="gr-checkboxgroup"
    )

    # Language selector
    language_selector = gr.Radio(
        choices=["English", "Vietnamese"],
        label="Select Language",
        value="English",  # Default language
        elem_classes="gr-row"
    )

    # Search button
    recommend_button = gr.Button("Search", elem_classes="gr-button")

    # Markdown title and HTML output for recommendations
    title_markdown = gr.Markdown(elem_id="title", elem_classes="gr-title")
    output_html = gr.HTML()

    # Event handlers
    query_input.submit(
        fn=lambda query, categories, language: recommend_books(query, categories, language) if query.strip() else default_top_books(language),
        inputs=[query_input, category_filter, language_selector],
        outputs=[title_state, output_html]
    )

    recommend_button.click(
        fn=lambda query, categories, language: recommend_books(query, categories, language) if query.strip() else default_top_books(language),
        inputs=[query_input, category_filter, language_selector],
        outputs=[title_state, output_html]
    )

    # Language toggle event
    language_selector.change(
        fn=lambda query, categories, language: recommend_books(query, categories, language) if query.strip() else default_top_books(language),
        inputs=[query_input, category_filter, language_selector],
        outputs=[title_state, output_html]
    )

    # Initial load
    demo.load(
        fn=lambda language: default_top_books(language),
        inputs=[language_selector],
        outputs=[title_state, output_html]
    )

    # Title update
    title_state.change(
        fn=lambda x: f"""<div class="gr-title">{x}</div>""",
        inputs=[title_state],
        outputs=[title_markdown]
    )

    # Layout
    gr.Row(
        query_input,
        category_filter,
        language_selector,
        recommend_button,
        elem_classes="gr-row"
    )
    
    gr.Markdown(
        """Hope you find some novels you love. Enjoy!<br>
        <a href="https://www.webnovelworld.org/home" target="_blank" style="text-decoration: underline !important;">Check it out here</a>""",
        elem_classes="gr-title"
    )

demo.launch(share=True)
