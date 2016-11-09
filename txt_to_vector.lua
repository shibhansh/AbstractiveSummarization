require 'torch'
print('Importing w2vutils')
wv=require 'w2vutils'

print('Finished importind w2vutils')

print('Scanning Text Reviews and generating word vectors ...')
file_text = io.open('text_words.csv','r')
all_doc={}
ind_doc={}
for line in file_text:lines() do
    if line=="" then
        table.insert(all_doc,ind_doc)
        ind_doc={}
    else
        table.insert(ind_doc,wv:word2vec(line))
    end
end
torch.save('text_wordvectors.t7',all_doc)
print('Done with Text Reviews')
file_text:close()

print('Scanning Summaries and generating word vectors ...')
file_text = io.open('summary_words.csv','r')
all_doc={}
ind_doc={}
for line in file_text:lines() do
    if line=="" then
        table.insert(all_doc,ind_doc)
        ind_doc={}
    else
        table.insert(ind_doc,wv:word2vec(line))
    end
end


torch.save('summary_wordvectors.t7',all_doc)
print('Done with Summaries')
print('Exiting')
