import requests
import json
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import nltk as nk
import matplotlib.pyplot as plt

#apikey : 1d8d625be3d443f8b3ad9f5d56fd5c5f

def process(raw):
    tokens = word_tokenize(raw)
    words = [w.lower() for w in tokens] #convert all words to lowercase
    porter = nk.PorterStemmer() 
    stemmed_tokens = [porter.stem(t) for t in words] #remove morphological affixes from words, leaving only the word stem.

    stop_words = set(stopwords.words('english')) 
    filtered_tokens = [w for w in stemmed_tokens if not w in stop_words] #remove stop words

    count = nk.defaultdict(int) #creation of dictionary
    for word in filtered_tokens: #count occurance of each token
        count[word]+=1
    return count;

def cos_sim(a,b):
    dot_product = np.dot(a,b)
    #Frobenius norm - square root of the sum of the absolute squares of its elements
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product/(norm_a*norm_b);

def getSimilarity(dict1,dict2):
    all_words_list = []
    for key in dict1:
        all_words_list.append(key)
    for key in dict2:
        all_words_list.append(key)
    all_words_list_size = len(all_words_list)
    #create array of integers
    v1 = np.zeros(all_words_list_size,dtype=np.int)
    v2 = np.zeros(all_words_list_size,dtype=np.int)
    i=0
    for key in all_words_list:
        v1[i] = dict1.get(key,0)
        v2[i] = dict2.get(key,0)
        i+=1
    return cos_sim(v1,v2);

def getDataFromAPI(url,fileName):
    response = requests.get(url)
    data = json.loads(response.text)
    file = open(fileName,"w")
    json.dump(data,file)
    file.close()
    return;    

def sortArticles(datastore,Articles):
    i=1
    for x in datastore['articles']:
        Articles.append(process(x['title']+" "+x['description']))
        print("Article",i,":")
        print(x['title']+" "+x['description'])
        print()
        i+=1
    print()
    return;

def plotGraph(maxSimilarity):
    values = [float(v) for v in maxSimilarity.values()]
    keys = [x for x in maxSimilarity.keys()]
    left = list(range(1,len(maxSimilarity)+1))
    plt.bar(left, values, tick_label = keys, width = 0.8)
    plt.ylabel('Similarity in %')
    plt.xlabel('Articles TOI->GNI')
    plt.title('Maximum Similarity of Each TOI Article with GNI Article')
    plt.xticks(list((float(a)+.3) for a in range(0,7)),rotation=10)
    plt.savefig("Graph.png")
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    return;

if __name__ == '__main__':
    url1 = ('https://newsapi.org/v2/top-headlines?'
            'sources=the-times-of-india&'
            'apiKey=1d8d625be3d443f8b3ad9f5d56fd5c5f')

    url2 = ('https://newsapi.org/v2/top-headlines?'
            'sources=google-news-in&'
            'apiKey=1d8d625be3d443f8b3ad9f5d56fd5c5f')

    getDataFromAPI(url1,"TOINews.json")
    getDataFromAPI(url2,"GNINews.json")

    file = open("TOINews.json","r")
    datastore1 = json.load(file)
    file.close()

    file = open("GNINews.json","r")
    datastore2 = json.load(file)
    file.close()

    TOIArticles = []
    GNIArticles = []
    
    
    print("Articles from Times of India:")
    print("============================================================================================================================")
    sortArticles(datastore1,TOIArticles)
    
    print("Articles from Google News India:")
    print("============================================================================================================================")
    sortArticles(datastore2,GNIArticles)
    
    i = 0
    k = 0
    similarity = {}
    similarArticles = []
    maxSimilarity = {}

    for Article1 in TOIArticles:
        i+=1
        j=1
        maximum=0
        for Article2 in GNIArticles:
            value=round(getSimilarity(Article1,Article2)*100,2)
            if value>maximum:
                maximum = value
                k = "Article "+str(i)+" TOI and Article "+str(j)+" GNI :"+str(value)
            similarity["Article "+str(i)+" TOI and Article "+str(j)+" GNI :"] = value
            print("Similarity between Article",(i)," of TOI and Article",(j),"of GNI:",value,"%")
            j+=1
        maximum = k.split(":")
        maxSimilarity[maximum[0]] = maximum[1]
                    
                    
    print("*****************************************************************************************************************************")

    print("Maximum Similarity of TOI Articles With GNI Articles : ")
    for Article in maxSimilarity:
        print(Article," : ",maxSimilarity[Article],"%")

    plotGraph(maxSimilarity)

