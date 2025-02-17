import streamlit as st
import torch
from model_handler import ModelHandler
from training_utils import Trainer
from visualization import plot_training_metrics
from dataset_handler import DatasetHandler

st.set_page_config(
    page_title="LLM Fine-tuning Lab",
    page_icon="🔬",
    layout="wide"
)

def main():
    st.title("🔬 LLM Fine-tuning Laboratory")
    
    # Sidebar for model selection and configuration
    st.sidebar.header("Model Configuration")
    
    model_name = st.sidebar.selectbox(
        "Select Base Model",
        ["facebook/opt-125m", "EleutherAI/gpt-neo-125M", "google/flan-t5-small"]
    )
    
    peft_method = st.sidebar.selectbox(
        "Select PEFT Method",
        ["LoRA", "Prefix Tuning", "P-Tuning"]
    )
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dataset Configuration")
        
        uploaded_file = st.file_uploader("Upload Training Data (CSV)", type=["csv"])
        
        if uploaded_file:
            dataset_handler = DatasetHandler()
            dataset = dataset_handler.load_dataset(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            st.write("Dataset Preview:")
            st.dataframe(dataset.head())
    
    with col2:
        st.subheader("Training Parameters")
        
        learning_rate = st.slider("Learning Rate", 1e-6, 1e-3, 1e-4, format="%.6f")
        num_epochs = st.number_input("Number of Epochs", min_value=1, max_value=50, value=3)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=32, value=8)

    # Model loading and training section
    if st.button("Initialize Model"):
        with st.spinner("Loading model..."):
            model_handler = ModelHandler(model_name)
            model = model_handler.load_model()
            st.session_state['model'] = model
            st.success(f"Model {model_name} loaded successfully!")

    if 'model' in st.session_state and uploaded_file:
        if st.button("Start Fine-tuning"):
            trainer = Trainer(
                model=st.session_state['model'],
                dataset=dataset,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=num_epochs,
                peft_method=peft_method
            )
            
            # Training progress container
            progress_container = st.empty()
            metrics_container = st.empty()
            
            # Training loop with progress updates
            for epoch, metrics in trainer.train():
                progress = (epoch + 1) / num_epochs
                progress_container.progress(progress)
                
                # Update metrics visualization
                fig = plot_training_metrics(metrics)
                metrics_container.plotly_chart(fig, use_container_width=True)
            
            st.success("Fine-tuning completed!")

if __name__ == "__main__":
    main()
