Working:

I create two variables called train and test. train is the filehandle for the training file and test is the filehandle for the testing file.
I create 2 dictionaries, namely mydict and mydict1.
mydict is a nested dictionary, whose keys are the cities given in the training file, and the value for each city is a dictionary, whose keys are the words which occur in the tweets from that city and the values are the number of times the word occurs in the tweets from that city.
For example: {'San_Francisco':{'Hello':10}} means that the word 'Hello' occurs 10 times in the tweets from San Francisco.
mydict1 is a dictionary to calculate the the P(L|Wn) values for each city. Its keys are the city names and values are the P(L|Wn) values for each city
I created a list of stopwords, in which I inserted the most common stop words I could think of.
I created a function called removespcl which takes a word as input and makes all the letters as lower case and also removes any non-alphanumeric (such as exclamations and underscores) except '#' (because hashtags are important!).
The program reads the training file line by line, storing each line at a time in a variable called line. A count variable maintains the number of lines in the file.
For each line:
  The line is then split into an array of words.
   Since the first word is the city, it is stored in a variable called city, and the rest of the words are stored in a variable called tweet.
   For each word in the tweet:
      The word is passed to the removespcl function where all the capital letters are converted to lower case and special characters are removed.
      The program then inserts the words or updates their values in mydict according to the description given in line 5.
Now, I remove all the stopwords in mydict.
The program then reads the testing file line by line, storing each line at a time in a variable called line.
For each line:
  The line is split into an array of words.
  Since the first word is the city, it is stored in a variable called rcity, and the rest of the words are stored in a variable called sample.
  For each word in the tweet:
      The word is passed to the removespcl function. 
      For each city, the value of the city in mydict1 is multiplied with the probability of that word given that city (P(W|L)) as the number of times the word appears in the tweets from that city divided by the sum of the number of times all the words appear in that city.
      There are speical cases, where a word in the training file is not available in the testing file for that city. For such cases, I multiply the value of the city by one over a million for each such word.
      By the end of this set of iterations, mydict1 contains the cities and the product of each P(Wn|L) for Wn=each word in the line. 
      I then multiply this by P(L), which is the probability of the location, which is calculated by dividing the number of times the city appears in the training set of tweets divided by the total number of tweets in the training set to get P(L|Wn) for each city.
  I then choose the city with the max P(L|Wn) value as the predicted city.
  If the predicted city is the same as the actual city (stored in rcity), the count of correct is incremented, otherwise the count of wrong is incremented. This is used to calcuate the final accuracy.
To display the top 5 words for each city, I have to display the words in the mydict[city] dictionary for which the values are the highest, for each city. I did not know exactly how to do this in an efficient manner, so I took a reference from https://stackoverflow.com/questions/7197315/5-maximum-values-in-a-python-dictionary .

Design decisions:

1. Changed all the uppercase letters to lowercase.
2. Removed all the special characters, i.e., the non-alphanumeric characters, except '#' because hashtags are important and can serve as an important identifier.
3. Removed some common stop words from the tweets.
4. There are cases where in the testing file, there may be tweets which do not contain anything except the city, i.e., there is just the city name and no tweet. For such cases, I decided to write those tweets directly in the output file without predicting their cities.
5. There are cases where in the testing file, there are some words which do not appear in the training file for a particular city. For such cases, while calculating the probability, I took P(the word | the city) to be 1/1000000.
