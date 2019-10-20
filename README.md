# NATURAL-LANGUAGE-PROCESSING
The dataset contains reviews and liked column depicting the type of review,good or bad.
Here a tsv file has been used instead of csv because the reviews may contain commas and that would be confusing.
Firstly the data has been cleaned by removal of punctuations and other irrelevant words. Moreover similar words has been converted to a single word.
Finally a bag of words model has been created using CountVectoriser which is basically provides a sparse matrix of words. Now it has been fed into a naive bayes classifier for the classification process.
An accuracy of about 74% has been achieved 
