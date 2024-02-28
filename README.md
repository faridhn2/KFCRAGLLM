# KFCRAGLLM
This project is to set up a large language model using RAG, which uses the Mistral model and answers questions related to the menu based on KFC menu data.
## Overview
This repository contains code for KFCRAG, a tool for querying information about KFC menus using a Retrieval-Augmented Generation (RAG) model. The code is organized into classes, each serving a specific functionality.
## Classes
<b>ChromaShip</b>: Represents a document in the Chroma vector store. It has attributes for content and optional metadata.

<b>KFCRAG</b>: The main functionality class. It includes methods for loading data from a JSON file, embedding documents into the Chroma vector store, creating a RAG model, generating a text pipeline, and setting up a Question-Answering (QA) chain.
## Usage
To use the code, follow these steps:

* Ensure you have the required dependencies installed (see the imports at the beginning of the code).
* Create an instance of the KFCRAG class, optionally specifying a JSON file path.
* Call the query method with a <b>query</b> string to get information from the KFC menus.

## Example
```
if __name__ == '__main__':
    kfc_rag = KFCRAG()
    print(kfc_rag.query('Give me an orange chocolate milkshake, medium'))
```
You can use this code :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144w89-CQuYm-Y-zkDQINK-MTBTMxT1jV?usp=sharing]

## Dependencies
* PyTorch
* Hugging Face Transformers
* Pandas
* LangChain library

## Notes
* The code uses a pre-trained RAG model ("mistralai/Mistral-7B-Instruct-v0.2") and a Chroma vector store.
* Data is loaded from a JSON file containing KFC menu information.

## Test 
I asked the trained model six sentences that fit the data, and the results were excellent in all six cases, which can be seen in the table below.
| Question |Response   |
|  :---:  |  :---: | 
| Can I get a Whopper? | No, you cannot get a Whopper at KFC as it is a different fast food chain from Burger King, where the Whopper is sold.   | 
|  :---:  | :---:  | 
| Give me a Veggie Tender, medium, with salad | Based on the given menu codes and contents, the Veggie Tender meal comes with a Veggie Tender (V2), a large drink from the Drinks section, and one side dish chosen from the Side dishes section. However, the question asks for a medium Veggie Tender instead of a large one. Unfortunately, there is no information provided in the context about the size options for the Veggie Tender itself. Therefore, I cannot provide a definitive answer to the question as it is currently stated. If more information were available about the size options for the Veggie Tender, I would be happy to help construct an order based on that information. As it stands, I can only suggest ordering a Veggie Tender meal with a medium Veggie Tender if such an option exists. Additionally, the question asks for a salad as a side dish, but the context does not specify which side dish is included in the Veggie Tender meal. Again, if more information were available, I could provide a more complete answer. But based on the context given, I can only recommend ordering a Veggie Tender meal with a medium Veggie Tender and asking for a salad as a side dish, assuming that those options are available.   | 
| :---:  | :---:  | 
| Hi, do you have cola? | No, you cannot get a Whopper at KFC as it is a different fast food chain from Burger King, where the Whopper is sold.   | 
|  :---:   | :---:  | 
| Can I get a Whopper? | No, you cannot get a Whopper at KFC as it is a different fast food chain from Burger King, where the Whopper is sold.   | 
|  :---:   | :---:  | 
| Can I get a Whopper? | No, you cannot get a Whopper at KFC as it is a different fast food chain from Burger King, where the Whopper is sold.   | 
|  :---:  | :---:  | 
| Can I get a Whopper? | No, you cannot get a Whopper at KFC as it is a different fast food chain from Burger King, where the Whopper is sold.   | 

