# ImageCaptioning
It is a project based on captioning the image by using deep learning techniques like LSTM and VGG16
Part-1 Understanding Problem
Dataset and its structure
Commonly known datasets that can be used for trianing purpose:
Flickr8K
Flick30K
Fick100K
MSCOCO
Each dataset may have there own structure of dataset. For Flickr_8K dataset, all the images of training, validation and test set are in one folder. It contains 3 different files i.e Flickr_8k.trainImages.txt, Flickr_8k.testImages.txt , Flickr_8k.devImages.txt corresponding to each type of dataset i.e train, test and validation set, each file having file_name of images conatined in each dataset.
For example, in Flick8k, Flickr_8k.trainImages.txt file contains file_ids of images in training set. Name of image file is its image id.
All the images are in same folder. So to parse images of training dataset, first read trianImages.txt file, read line by line image id and load corresponding image from image dataset folder.
Each image is given 5 different captions by 5 different humans. This is because an image can be described in multiple ways.
These captions are stored in 'Flickr8k.token.txt'. Each line of file contains a caption corresponding to an image. And for one image, there are 5 lines representing 5 captions for one image file.
Research papers and resources to follow
https://arxiv.org/pdf/1411.4555.pdf
https://arxiv.org/pdf/1412.2306.pdf
https://www.youtube.com/watch?v=yCC09vCHzF8 (Video lecture by author of paper 2)
https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/
How can we define a single training example?
First thing that comes into mind is that how are we going to restructure our dataset in order to train it.

We are going to generate one word at a time inorder to generate complete sentence. To generate each word, we will provide 2 types of inputs.

Image
Sentence that has already been predicted so that model can use the context and predict next word.
How our training sample is going to look?

We have to add 2 special tokens to each captions that represents start of sentence and end of sentence.
Then we need to split each caption and image pair in multiple data samples.image1.png
So corresponding to a single image and a caption, we are going to generate multiple data samples.

Structure of a generic model for image captioning
https://arxiv.org/pdf/1703.09137.pdf Depending upon the choice, your structure of model will change.

Commonly used architecture
image2.png

Extracting image features
To extract image features, we can use CNN network. Either we can make our own CNN network or we can use concept of transfer learning. There are a lot of pre-trained models available to extract image features. For example, VGG16 model contains 16 layers which is used to classify image in 1 out of 1000 classes. Last layer of this model is used for classification, so we can capture the output of second last layer (which is a vector of size 4096 for a single image) as it will represent image features in form of numbers without classifying them in classes.
So, we can pass each image in our dataset through this network and store the results corresponding to image id in a file.
Q: Why are we extracting image features before training and storing them in files when we can extract them while training too?
Ans: For one image, there are 5 captions and each caption can have max length of 32-34 words (observation from dataset). So, for each image, no. of training samples can reach up to 5 * 34 = 170, and each time VGG model will generate same output as image is same. As, VGG16 model is quite a big model, we want to reduce this time by computing these features once and using them directly from dump file instead of using VGG model for each sample to predict image features.
Providing text features
One very common way of defining our words features is that first we define our vocabulary and then one hot encode each word as a vector of vocab_size.
It has several drawbacks( see this https://www.youtube.com/watch?v=JKpm3DMSSMI and further few lectures to understand word embeddings completely) and thus we are going to use word embeddings to represent our words.
We will also need to define a maximum length which a sentence can take so that we could define size of our imputs because we are going to input a group of words to our model and expect a single word output.
For example, in Flickr8K dataset, maximum length of sentence is around 34. Let us say we have a map of words to integer as {"a": 1, "girl": 2, "running": 3, "near": 4}, so to encode a sentence 'a girl', we will make an array of size 32 (maximum length of sentence), put 1st two integers using word to index mapping as [1,2] and append this with zeros which represent no word. So, our input vector representing a word is going to look like [1,2,0,0,0,...] and so on of size 34. This input will be fed to a word embedding layer.
We need not change each word to its corresponding one hot encoding as we are not using one hot encoded features. We are going to feed word to index mapping for each word of sentence to word embedding layer (let us say it has n units) and it will convert each word to n sized feature array. So, its input is a vector of size (max_length of sentence) and it will change this to (max_length_of_sentence,n).
What is encoder in such models?
The neural networks that changes the any input in its features representation using vector of numbers is encoder. For example, we want to use image to predict words. As image directly can't tell what should be the word, we want to use its feature to help us decide the next word. And thus the network of layers used to change image or any other type of input in its feature representation is known as encoders.

What is decoder?
The combination of layers/neural network that takes feature representation provided by encoder as its own input and predicts the next word, is known as decoder.

Sample architecture that you can try to build
image3.png

Part-2 Predictions and Evaluating Result
BLEU Score for Evaluating Model:
It is the most widely used automated method of determining the quality of machine translation. The BLEU metric scores a translation on a scale of 0 to 1, but is frequently displayed as a percentage value. The closer to 1, the more the translation correlates to a human translation.

https://www.youtube.com/watch?v=DejHQYAGb7Q Andrew NG explaining BLEU score and why we should use this parameter to grade our trained model instead of precision or accuracy.

Let us say we predicted a sentence S and we have actual 5 correct sentences S1, S2, S3, S4 and S5, then BLEU score considers all the 5 correct sentences and then evaluates correctness of predicted sentence S. It uses the concept of n-grams i.e a contiguous sequence of n items from a given sample of text. BLEU-1 score corresponds to 1-gram score, BLEU-2 to 2-grams and so on. We generally go up to 3 or 4 n-grams to check model's correctness. Higher the value of n in n-grams, the score of model would go low.

NOTE:
While training, we will not be evaluating BLEU score at the end of each epoch, instead we will use accuracy for that part because calculating BLEU score after each epoch is going to take a lot of time. So, we will use it only after training of model ends.

Predicting Sentences
Method 1: Greedy Technique
In this method, in each iteration, we will pass model 2 inputs, first is image feature and second is set of already predicted words. Our model will give us the probability of each word in vocabulary to be selected as next word. We can select the word having maximum probability and add to our set of words already predicted in a sequence. And in next iteration we pass this as input to model. We can repeat this unless we get 'endseq' token as output of model which represents that the sentence has ended. This is greedy technique because we are selecting word havig maximum probability in each step and rejecting all other words.

Method 2: Beam Search Technique
In this technique, we first define a parameter called beam size (b). In each iteration we select top b candidates having max. probability.

Let us say, path proabilities (product of probabilities of all words predicted in a candidate sentence) for them are {p1,p2,p3,...pb} for set of sentences {s1,s2,s3,...,sb} where each sentence can have one or more words.

In 1st iteration we select top b words, and make each of them as candidate sentences ({s1,s2,s3,...,sb}).
In following iteration:
Make a new set of candidates (NS)
For each candidate sentence from old candidates ({s1,s2,...sb}):
Convert sentence to input feature and predict output. Again select top b words and add them to current sentence and push each sentence to new candidates set (NS).
Now we have new candidates set having b^2 candidates. Select top b candidates having maximum path probabilities.
Update old candidates set with newly selected top b candidates.
Do step 2 untill all candidates see 'endseq' token at the end.
Intuition: How is it better than greedy technique?
Assume that in ith iteration, we have two words (W1, W2) having nearly same probabilities i.e 0.3 and 0.32. We selected word W2 with 0.32 and in next iteration after selected W2, all the words get probability in 0.1 to 0.2 range which means our model is not so sure about which word to select.

Had we selected W1 instead, in next iteration our model might have predicted words with more confidence and hence in long term selecting W1 might prove to be more beneficial.

image4.png

In above example, if we used greedy search or with beam search with beam size = 1, we might have ended up predicting ABB having path probability 0.216 (0.6*0.6*0.6).

But with beam size = 2, our prediction will be BBB with path probability 0.324 (0.4*0.9*0.9)

Hence, it is a heuristic technique of seaching a good acceptable answer among all the possible states.

Note:
Beam search with beam size = 1 is equivalent to greedy search.
