import graphlab

#uploading dataset
product=graphlab.SFrame('amazon_baby.sframe')
#visualizing the dataset
product

graphlab.canvas.set_target('ipynb')
product.show()

#creating a new column that contains the word count of the review
product['word_count']=graphlab.text_analytics.count_words(product['review'])

#shows the most frequent items purchased
product['name'].show()

#reviews of the most purchased item 'Vulli the Giraffe Teether'
gr=product[product['name']=='Vulli Sophie the Giraffe Teether']
gr['rating'].show(view='Categorical')


#selcted words through which we will buld our model
selected_words = ['awesome', 'great', 'fantastic', 'amazing', 'love', 'horrible', 'bad', 'terrible', 'awful', 'wow', 'hate']

#functions to return the value/count of the key/selected words which is stored in the form of dictionary
def awesome_count(word_count):
    if 'awesome' in word_count:
        return (word_count['awesome'])
    else:
        return 0;
def great_count(word_count):
    if 'great' in word_count:
        return (word_count['great'])
    else:
        return 0;
def fantastic_count(word_count):
    if 'fantastic' in word_count:
        return (word_count['fantastic'])
    else:
        return 0;
def amazing_count(word_count):
    if 'amazing' in word_count:
        return (word_count['amazing'])
    else:
        return 0;
def love_count(word_count):
    if 'love' in word_count:
        return (word_count['love'])
    else:
        return 0;
def horrible_count(word_count):
    if 'horrible' in word_count:
        return (word_count['horrible'])
    else:
        return 0;
def bad_count(word_count):
    if 'bad' in word_count:
        return (word_count['bad'])
    else:
        return 0;
def terrible_count(word_count):
    if 'terrible' in word_count:
        return (word_count['terrible'])
    else:
        return 0;
def awful_count(word_count):
    if 'awful' in word_count:
        return (word_count['awful'])
    else:
        return 0;
def wow_count(word_count):
    if 'wow' in word_count:
        return (word_count['wow'])
    else:
        return 0;
def hate_count(word_count):
    if 'hate' in word_count:
        return (word_count['hate'])
    else:
        return 0;

#columns for each word count
product['awesome']=product['word_count'].apply(awesome_count)
product['great']=product['word_count'].apply(great_count)
product['fantastic']=product['word_count'].apply(fantastic_count)
product['amazing']=product['word_count'].apply(amazing_count)
product['love']=product['word_count'].apply(love_count)
product['horrible']=product['word_count'].apply(horrible_count)
product['bad']=product['word_count'].apply(bad_count)
product['terrible']=product['word_count'].apply(terrible_count)
product['awful']=product['word_count'].apply(awful_count)
product['wow']=product['word_count'].apply(wow_count)
product['hate']=product['word_count'].apply(hate_count)

product.show()

#splitting the data into train_set and test_set
train_data,test_data=product.random_split(.8,seed=0)


#Building the model
selected_words_model=graphlab.logistic_classifier.create(train_data,target='sentiment',features=selected_words,validation_set=test_data)

#Evaluating the model
print selected_words_model.evaluate(test_data)

#applying the model
gr['predicted_sentiment']=selected_words_model.predict(gr,output_type='probability')
gr=gr.sort('predicted_sentiment',ascending=False)
gr
