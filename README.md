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
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/144w89-CQuYm-Y-zkDQINK-MTBTMxT1jV?usp=sharing)

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
| Can I get a Whopper? | No, there is no Whopper option on this menu. The options available are the Original Burger Boxmeal (M27), Colonel Burger Meal (M9), and another Original Burger Meal (M12).  | 
|  :---:  | :---:  | 
| Give me a Veggie Tender, medium, with salad | Based on the given menu codes and items, the correct order would be "M20" for the menu code which includes a Veggie Tender as one of its options, followed by "choosing 1 side dish" which could be a salad, and specifying the size as "medium" for the Veggie Tender itself. Therefore, the final order would be "M20, choose 1 from Side dishes (salad), medium Veggie Tender". The total price for this order would be $30.00.   | 
| :---:  | :---:  | 
| Hi, do you have cola? | Yes, we have Pepsi with a code D2 and it costs $2.8. However, it is currently not available.   | 
|  :---:   | :---:  | 
| Give me an orange chocolate milkshake, medium | I'm sorry, neither of the given items match your request for an orange chocolate milkshake, medium. The first item is a Chocolate Sundae with no size or type specified and the second item is plain Ice Cream also without any size or type specified. To order an orange chocolate milkshake, medium, you would need to check if it's available on the menu and then place an order for it accordingly.   | 
|  :---:   | :---:  | 
| Give me the gluten free burger options | Based on the provided information, there are no gluten-free burger options available as all burgers listed have allergens that include wheat or soy.  | 
|  :---:  | :---:  | 
| How many calories does the Colonel have? | The Colonel refers to the Colonel Burger and Colonel Stacker items in this context. Each of these items has a nutritional information object with a calorie count of 150. Therefore, the Colonel has a total of 150 calories (for one item). Since the question doesn't ask for the number of calories for multiple items or meals, the answer is simply "The Colonel has 150 calories." or "150 calories" if spoken out loud.   | 

You can use this code for testing :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1vZZ4KKRIWDwCyJDLFoN9Qu6ld27mTEa0?usp=sharing)

## Test Time
I evaluated the processing speed of these six questions in several stages and the result is reported in the table below
| Question |Processing Time   |
|  :---:  |  :---: | 
| Can I get a Whopper? | 2.56 s ± 340 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)  | 
|  :---:  | :---:  | 
| Give me a Veggie Tender, medium, with salad | The slowest run took 5.31 times longer than the fastest. 16.3 s ± 7.24 s per loop (mean ± std. dev. of 7 runs, 1 loop each) | 
| :---:  | :---:  | 
| Hi, do you have cola? | 3.85 s ± 314 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)  | 
|  :---:   | :---:  | 
| Give me an orange chocolate milkshake, medium | 4.63 s ± 1.68 s per loop (mean ± std. dev. of 7 runs, 1 loop each)  | 
|  :---:   | :---:  | 
| Give me the gluten free burger options | 3.4 s ± 1.02 s per loop (mean ± std. dev. of 7 runs, 1 loop each)  | 
|  :---:  | :---:  | 
| How many calories does the Colonel have? | 1.5 s ± 121 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)  | 

You can use this code for speed testing  :
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1amGO0apUdK4MTWU3NpUp752YMl5FlBX-?usp=sharing)

## Suggestion to speed up
In this code, we used the language model locally, and the faster method is to use server-based models, such as using HoggingFace endpoint APIs, which will give us a much higher speed.
