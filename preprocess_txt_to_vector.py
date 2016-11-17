from nltk.tokenize import word_tokenize
import sqlite3
import os
import numpy as np
import sys 
reload(sys)

try:
    NumReviews_train = int(sys.argv[1])
    NumReviews_test = int(sys.argv[2])
    print(NumReviews_train)
except:
    NumReviews_train = 10000
    NumReviews_test = 10000

rand = np.random.permutation(NumReviews_train + NumReviews_test)+1;

sys.setdefaultencoding('utf-8')
conn=sqlite3.connect('database.sqlite')
cur=conn.cursor()
# txt_wv=open('Word2Vec/text_words.csv',"w")
txt=open('text_words.csv',"w")
smy=open('summary_words.csv',"w")
os.system("echo \"\" > text_words.csv && echo \"\" >  summary_words.csv")

for i in range(1,NumReviews_train):
    exec_str='select Text, Summary from Reviews where rowid='+str(rand[i])
    cur.execute(exec_str) 
    tmp=cur.fetchall()
    words_txt=word_tokenize(tmp[0][0])
    words_smy=word_tokenize(tmp[0][1])
    
    for j in xrange(len(words_txt)):
        txt.write(words_txt[j]+str('\n'))
    txt.write("\n")

    for j in xrange(len(words_smy)):
        smy.write(words_smy[j]+"\n")
    smy.write("\n")

    strw = ""
    for j in words_txt:
        strw += " " + j
    strw += "\n"

    for j in words_smy:
        strw += " " + j
    strw += "\n"

    # txt_wv.write(strw)

    if i%1000==0:
        print('Preprocessed '+str(i)+' Reviews for TRAINING')


# txt_wv=open('Word2Vec/test_text_words.csv',"w")
txt=open('test_text_words.csv',"w")
smy=open('test_summary_words.csv',"w")
os.system("echo \"\" > test_text_words.csv && echo \"\" >  test_summary_words.csv")
for i in range(NumReviews_train, NumReviews_test+NumReviews_train):
    exec_str='select Text, Summary from Reviews where rowid=' + str(rand[i])
    cur.execute(exec_str) 
    tmp=cur.fetchall()
    words_txt=word_tokenize(tmp[0][0])
    words_smy=word_tokenize(tmp[0][1])
    
    for j in xrange(len(words_txt)):
        txt.write(words_txt[j]+str('\n'))
    txt.write("\n")

    for j in xrange(len(words_smy)):
        smy.write(words_smy[j]+"\n")
    smy.write("\n")

    strw = ""
    for j in words_txt:
        strw += " " + j
    strw += "\n"

    for j in words_smy:
        strw += " " + j
    strw += "\n"

    # txt_wv.write(strw)

    if i%1000==0:
        print('Preprocessed '+str(i-NumReviews_train)+' Reviews for TEST')
