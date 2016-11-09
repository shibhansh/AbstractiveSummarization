from nltk.tokenize import word_tokenize
import sqlite3
import os
import sys

reload(sys)

sys.setdefaultencoding('utf-8')
conn=sqlite3.connect('database.sqlite')
cur=conn.cursor()
txt=open('Word2Vec/text_words.csv',"w")
os.system("echo \"\" > text_words.csv && echo \"\" >  summary_words.csv")
for i in range(1,10000):
    exec_str='select Text, Summary from Reviews where rowid='+str(i)
    cur.execute(exec_str) 
    tmp=cur.fetchall()
    words_txt=word_tokenize(tmp[0][0])
    words_smy=word_tokenize(tmp[0][1]) 
    
    strw = ""
    for j in words_txt:
        strw += " " + j
    strw += "\n"

    for j in words_smy:
        strw += " " + j
    strw += "\n"

    txt.write(strw)

    if i%100==0:
        print('Evaluated '+str(i)+' Reviews')
