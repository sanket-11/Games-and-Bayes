#!/usr/bin/env python3

import sys
import math

train = open(sys.argv[1],'r',encoding='ISO-8859-1')
test = open(sys.argv[2],'r',encoding='ISO-8859-1')
output = open(sys.argv[3],'w')
mydict={}
mydict1={}
cities={}
correct=0
wrong=0
count=0
stopwords=['','i','me','my','we','our','you','your','he','him','his','she','her','it','its','they','them','their','what','who','this','that','these','those','am','is','are','was','were','be','being','has','have','had','do','does','did','a','an','the','and','but','if','or','as','while','of','at','by','for','with','to','from','in','out','again','then','here','there','when','where','why','how','all','any','more','no','only','so','than','too','can','will']

def removespcl(word):
	temp=""
	word=word.lower()
	for letter in word:
		if letter.isalnum() or letter=='#':
			temp+=letter
	return temp

for line in train:
	count+=1
	i=line.split()
	city=i[0]
	tweet=i[1:]
	if city in cities.keys():
		cities[city]=cities[city]+1
	else:
		cities[city]=1
	for word in tweet:
		word=removespcl(word)
		if city in mydict.keys():
			if word in mydict[city].keys():
				mydict[city][word]+=1
			else:
				mydict[city][word]=1
		else:
			mydict[city]={}
			mydict[city][word]=1


for city in cities.keys():
	for word in stopwords:
		if word in mydict[city]:
			mydict[city].pop(word)

for line in test:
	sen=line.split()
	rcity=sen[0]
	sample=sen[1:]
	if len(sample)==0:
		output.write(city)
		break
	for word in sample:
		word=removespcl(word)
		for city in cities.keys():
			if city in mydict1.keys():
				if word in mydict[city].keys():
					mydict1[city]=mydict1[city]*(mydict[city][word]/sum(mydict[city].values()))
				else:
					mydict1[city]=mydict1[city]*(1/1000000)
			else:
				if word in mydict[city].keys():
					mydict1[city]=mydict[city][word]/sum(mydict[city].values())
				else:
					mydict1[city]=1/1000000
	for city in mydict1.keys():
		mydict1[city]=mydict1[city]*(cities[city])
	city = max(mydict1,key=mydict1.get)
	output.write("%s %s" %(city,line))
	if city==rcity:
		correct+=1
	else:
		wrong+=1
	mydict1.clear()



acc=(correct/(correct+wrong))*100
print("Accuracy = %d%%" %acc)

for city in cities.keys():
	print(city)
	print(sorted(mydict[city],key=mydict[city].get,reverse=True)[:5])
