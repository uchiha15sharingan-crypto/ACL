
## CARE Pipeline

This repository contains a two-stage pipeline for explanation-aware clinical trait classification using large language models.

- CARE.py – training, validation, and test evaluation

- inference.py – inference on unseen data using a trained checkpoint

### Data

The pipeline operates on **CSV-based conversational data**.

### Expected Format

Each CSV contains therapist–patient conversations.  
Only **therapist (`T`) utterances** are used for prediction.

### Required Columns

-   `ID` _(or `conv_id` + `turn_index`)_
    
-   `Utterance` _(or `utterance`)_
    
-   `Type` / `speaker` (`T` or `P`)
    
-   Ordinal labels in `{-2, -1, 0, 1, 2}` for:
    
    -   Non-Judgmental Language
        
    -   Warmth and Encouragement
        
    -   Respect for Autonomy
        
    -   Active Listening
        
    -   Reflecting Feelings
        
    -   Situational Appropriateness
        

Conversation context is automatically constructed using a **sliding window** over previous turns.

----------

## Code

### CARE.py

End-to-end **training and evaluation pipeline**.

-   Generates **RAG-based explanations** using an explainer LLM
    
-   Trains a **hierarchical LoRA-based classifier**
    
-   Evaluates using **Quadratic Weighted Kappa (QWK)** and accuracy
    
-   Saves best model checkpoints and confusion matrices
    

Run:

`python CARE.py` 

----------

### inference.py

**Inference pipeline for unseen data**.

-   Uses training data as a **retrieval knowledge base**
    
-   Generates explanation-aware inputs
    
-   Loads a trained checkpoint and predicts clinical trait scores
    
-   Saves predictions, explanations, and evaluation plots
    

Run:

`python inference.py` 

----------

## Setup

Create an environment and install dependencies:

`pip install torch transformers sentence-transformers peft scikit-learn pandas numpy tqdm matplotlib seaborn` 

A **GPU is recommended** for both training and inference.
