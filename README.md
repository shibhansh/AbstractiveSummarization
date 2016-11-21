# AbstractiveSummarization
Food Review Summarization

Download the database from kaggle and store it as 'database.sqlite' in the same directory as the code 
Run 
```python new_preprocess.py n1 n2 ```
Example
```python new_preprocess.py 50000 50000 ```
where n1 is the number of reviews to train on and n2 is the number of test reviews.

To train create a directory named 'Models' and simply run
 ```th train.lua```
 train.lua saves the model after every three epochs in the folder Models/ with the file name model<numberOfReviews>
 i.e. Models/model120000 if 6 epochs have been completed with number of reviews = 20000
 
 Copy the model you wish to evaluate into the file model_tmp (not Models/model_tmp)
 
 To test/evaluate, open torch, require eval.lua, and then run eval() or predict(<string>)
 
 ```
 th
 require('eval.lua')
 eval()
 predict("Did not like this coffee . Hated it . Very bad taste .")
```
Note the spaces between punctuation marks. i.e "coffee ." or "yummy !" rather than "coffee." and "... yummy!"
Evaluate will create three files evalText.txt, predicted_summary.txt, target_summary.txt.
- evalText.txt contains the original Reviews one on each line,
- target_summary.txt and predicted_summary.txt contain the corresponding original and predicted summaries on the same line number.

To evaluate further and get quantative data these files can be used.
