# Sentiment Analysis for Amazon Baby-Products
Here I have built two models  
a) Based on count of all the words (Word_count_model.ipynb)  
b) Based on count of some selected words (Selected_words_model.ipynb)  
# Requirements
Graphlab Create  
Anaconda2  
IPython Notebook
#
Download the amazon_baby.sframe dataset file.
# Accuracy of the models
The accuracy of the word_count_model is 91.6%. While the accuracy of the selected_words_model is 84.3%.  
The less accuracy of the selected_words_model is because in some reviews none of the selected words are present, so the accuracy can
be increased by adding more words to the selected words list. 
