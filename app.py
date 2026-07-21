import gradio as gr
from transformers import pipeline
import os

# Try loading local model, otherwise fallback to Hub
MODEL_DIR = "./models/transformer_ner/"
if os.path.exists(MODEL_DIR):
    model_id = MODEL_DIR
    print(f"Loading local model from {model_id}")
else:
    model_id = "Afifi333/NER-DistilBERT"
    print(f"Local model not found. Loading fallback from Hub: {model_id}")

try:
    # Use aggregation_strategy to merge B- and I- tags properly
    ner_pipeline = pipeline("ner", model=model_id, aggregation_strategy="simple")
except Exception as e:
    print(f"Error loading model: {e}")
    ner_pipeline = None

def predict_entities(text):
    if not ner_pipeline:
        return [(text, "Error: Model not loaded")]
        
    predictions = ner_pipeline(text)
    
    # Process predictions for Gradio HighlightedText format: [(text_span, label), ...]
    result = []
    last_idx = 0
    
    for entity in predictions:
        # Get start and end character indices
        start = entity['start']
        end = entity['end']
        
        # Add text before the entity as non-labeled (None)
        if start > last_idx:
            result.append((text[last_idx:start], None))
            
        # Add the entity with its label
        label = entity['entity_group']
        result.append((text[start:end], label))
        
        last_idx = end
        
    # Add remaining text after the last entity
    if last_idx < len(text):
        result.append((text[last_idx:], None))
        
    return result

# Color map for entities
color_map = {
    "PER": "#fca5a5",
    "ORG": "#93c5fd",
    "LOC": "#86efac",
    "MISC": "#fde047"
}

examples = [
    ["Hugging Face Inc. is a company based in New York City. Its headquarters are in DUMBO, therefore very close to the Manhattan Bridge which is visible from the window."],
    ["My name is Sarah and I work at Google in London."],
    ["The World Health Organization warned about the new virus in Geneva."],
    ["Elon Musk founded SpaceX in Hawthorne, California."],
    ["I bought a new Apple iPhone 15 Pro yesterday from the store in Times Square."]
]

with gr.Blocks(title="Named Entity Recognition", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Named Entity Recognition (NER) Demo")
    gr.Markdown("This app extracts Named Entities (Persons, Organizations, Locations, and Miscellaneous) from text using a Transformer-based model (DistilBERT/BERT).")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=5, 
                placeholder="Enter text here...", 
                label="Input Text"
            )
            submit_btn = gr.Button("Analyze Entities", variant="primary")
            
        with gr.Column():
            output = gr.HighlightedText(
                label="Extracted Entities",
                color_map=color_map,
                show_legend=True
            )
            
    submit_btn.click(fn=predict_entities, inputs=input_text, outputs=output)
    
    gr.Examples(
        examples=examples,
        inputs=input_text,
        outputs=output,
        fn=predict_entities,
        cache_examples=False
    )
    


if __name__ == "__main__":
    demo.launch()

