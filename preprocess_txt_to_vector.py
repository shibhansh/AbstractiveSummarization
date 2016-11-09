from nltk.tokenize import word_tokenize
import sqlite3
import os
import sys
reload(sys)

try:
    NumReviews = int(sys.argv[1])
    print(NumReviews)
except:
    NumReviews = 10000

sys.setdefaultencoding('utf-8')
conn=sqlite3.connect('database.sqlite')
cur=conn.cursor()
txt_wv=open('Word2Vec/text_words.csv',"w")
txt=open('text_words.csv',"w")
smy=open('summary_words.csv',"w")
os.system("echo \"\" > text_words.csv && echo \"\" >  summary_words.csv")

for i in range(1,NumReviews):
    exec_str='select Text, Summary from Reviews where rowid='+str(i)
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

    txt_wv.write(strw)

    if i%100==0:
        print('Evaluated '+str(i)+' Reviews')
