Baseline -
Giving our baseline.py script no argument on the command line will make it
automate to having as input testing.json and as output final_test_baseline.json.
Thus if you would like to change the data path, the first argument is the input
and the second argument is the desired output.

Final -
parse.py will parse the training data and testing data. Update the variables
for parseTrain and parseTest to customize the training and testing data. To
parse the development data, also run the function parseTrain.

wordEmbeddings.py is the code we used to run word embeddings. However, you do
not need to run this anymore because we've already included the glove file
with all the pretrained embeddings in the folder. We did not include the original
glove.6B.50d.txt file in the submission folder so wordEmbeddings.py will not run.

embeddings.py will use InferSent and save the sentence embeddings. Same as above,
it will not run because we did not include the InferSent folder. Instead, we
included the pickle files in the embeddings folder.

we also have a tags.py file that we used to get all the POS tag sequence of
each sentence and question. We've included it in the final submission folder,
but again, for convenience we just added the pos tag files in the data folder.

features.py contains all the functions for extracting features. It is called
trainFeatures.py or testFeatures.py is called. Run trainFeatures.py and
testFeatures.py to extract the features in order to create the input for running
the final model. The files will be dumped in the features folder.

final.py is the code that trains the logistic regression model and dumps the
output prediction.
