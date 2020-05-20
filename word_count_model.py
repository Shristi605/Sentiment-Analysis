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

#Building a sentiment classifier
product=product[product['rating']!=3] #removing reviews given as 3 as it doesn't specify whether the product has positive review or negative
product['sentiment']=product['rating']>=4

#splitting the data into train_set and test_set
train_data,test_data=product.random_split(.8,seed=0)


#Building the model
model=graphlab.logistic_classifier.create(train_data,target='sentiment',features=['word count'],validation_set=test_data)

#Evaluating the model
print model.evaluate(test_data)

model.evaluate(test_data,metric='roc_curve')

model.show(view='Evaluation')

#applying the model
gr['predicted_sentiment']=model.predict(gr,output_type='probability')
gr=gr.sort('predicted_sentiment',ascending=False)
gr
