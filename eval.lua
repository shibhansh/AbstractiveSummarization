require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
---------------------------------

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
----------------------------------

file_txt=io.open('test_text_words.csv','r')
file_smy=io.open('test_summary_words.csv','r')

batchSize = 5
numDocuments = 100
hiddenSize = 1000


loaded_model = torch.load('./model_tmp')
model = loaded_model.model
Vocab = loaded_model.Vocab
Index2Vocab = loaded_model.Index2Vocab

--*************************************************************

function nextBatch()
    local encoderInputs, decoderInputs, decoderTargets = {}, {}, {}

    sentBatch_text = {}
    sentBatch_smy={}
    local max_len_text = 0
    local max_len_smy = 0 
    local j=0, j2

    for j=1,batchSize do 
        local sent_txt = {}
        local sent_smy = {}

        line = file_txt:read():lower()
        while line ~= "" and line~=nil do 
            table.insert(sent_txt,line)
            line = file_txt:read()
            if line ~= nil then
                line = line:lower()
            end
        end
        table.insert(sentBatch_text,sent_txt)
        
        if max_len_text < #sent_txt then
            max_len_text = #sent_txt
        end

        table.insert(sent_smy, '<go>')
        line = file_smy:read():lower()
        while line ~= "" and line~= nil  do     
            table.insert(sent_smy, line)
            line = file_smy:read()
            if line~=nil then
                line = line:lower()
            end
        end 
        table.insert(sent_smy, '<eos>')
        table.insert(sentBatch_smy,sent_smy)
        
        if max_len_smy < #sent_smy then
            max_len_smy = #sent_smy
        end 
    end 

    local encoderInputs = torch.IntTensor(max_len_text, batchSize):fill(0)
    local decoderInputs = torch.IntTensor(max_len_smy-1,batchSize):fill(0)
    local decoderTargets= torch.IntTensor(max_len_smy-1,batchSize):fill(0)

    for j=1,batchSize do
        eosOffset = max_len_text - #sentBatch_text[j] --left paddding

        for j2=1,#sentBatch_text[j]do   
            if  Vocab[sentBatch_text[j][j2]] ~= nil then 
                encoderInputs[j2+eosOffset][j] = Vocab[sentBatch_text[j][j2]]
            else
                encoderInputs[j2+eosOffset][j] = Vocab['<unknown>']
            end

        end

        --trimmedEosToken = sentBatch_smy[j]:sub(1,-2)
        for j2=1,#sentBatch_smy[j]-1 do
            if Vocab[sentBatch_smy[j][j2]] ~= nil then 
                decoderInputs[j2][j] = Vocab[sentBatch_smy[j][j2]]
            else
                decoderInputs[j2][j] = Vocab['<unknown>']
            end
        end
        

        --trimmedGoToken = sentBatch_smy[j]:sub(2,-1)
        for j2=2,#sentBatch_smy[j] do
            if Vocab[sentBatch_smy[j][j2]] ~= nil then
                decoderTargets[j2-1][j] = Vocab[sentBatch_smy[j][j2]]
            else
                decoderTargets[j2-1][j] = Vocab['<unknown>']
            end
        end
    end
    

    if encoderInputs:dim()==0 then
        return -1
    end

    --print('-------------------------------------------')
    --print(encoderInputs,decoderInputs,decoderTargets)
    --print('-----------------------------------------asasas-')
    if encoderInputs:size()[1] > 100 then
        encoderInputs=encoderInputs:sub(-100,-1)
    end
    return encoderInputs, decoderInputs, decoderTargets
end 
--*************************************************************


--*************************************************************
eval = function()
    evalText = io.open('evalText.txt','w')
    evalUserSmy = io.open('target_summary.txt','w')
    evalPredSmy = io.open('predicted_summary.txt','w')

    for i = 1,numDocuments/batchSize do
        model.encoder:forget()
        model.decoder:forget()
        print('Evaulated for : ' .. i*batchSize .. 'Reviews')
        local encoderInputs, decoderInputs, decoderTargets = nextBatch()
       
        for batchSize_ind = 1,batchSize do
            str_sent = "" 
            enc_batch_size = encoderInputs:select(2,batchSize_ind)

            for ii = 1,(#enc_batch_size)[1] do   
                if enc_batch_size[ii] ~= 0 then
                    local word = Index2Vocab[enc_batch_size[ii]]
                    str_sent = str_sent .. " " .. word
                end
            end 
            evalText:write(str_sent .. '\n')

            str_sent = "" 
            decoder_batch_size = decoderTargets:select(2,batchSize_ind)

            for ii = 1,(#decoder_batch_size)[1] do
                if decoder_batch_size[ii] ~= 0 then
                    local word = Index2Vocab[decoder_batch_size[ii]]
                    str_sent = str_sent .. " " .. word
                end
            end 
            evalUserSmy:write(str_sent .. '\n')

            wordIds, probabilities = model:eval(encoderInputs:select(2,batchSize_ind))
            str_sent=""
            for _tmp, ind in ipairs(wordIds) do 
                local word = Index2Vocab[ind[1]]
                str_sent = str_sent .." ".. word
            end
            evalPredSmy:write(str_sent .. "\n")

        end
    end
    evalText:close()
    evalUserSmy:close()
    evalPredSmy:close()
end

--*************************************************************

predict = function(text)
    local text = text:split(' ')
    if #text > 100 then
        print('Too big a sentence')
        return
    end

    local encoderInputs = torch.Tensor(#text)

    for ii=1,#text do
        text[ii] = text[ii]:lower()
        if Vocab[text[ii]] ~= nil then
            encoderInputs[ii] = Vocab[text[ii]]
        else
            print('Unknown words present in the sentence. Aborting...')
            return
        end
    end

    local wordIds, probabilities = model:eval(encoderInputs)

    str_sent=""
    for _tmp, ind in ipairs(wordIds) do 
        local word = Index2Vocab[ind[1]]
        str_sent = str_sent .." ".. word
    end
    print(str_sent .. " : " )
    model.encoder:forget()
    model.decoder:forget()
end


--*************************************************************
test = function()
    print('test')

    file_txt = io.open('test_text_words.csv','r')
    file_smy = io.open('test_summary_words.csv','r')


    for i = 1, numDocuments/batchSize do	
        wordIds, decoderTargets = eval()
        model.decoder:forget()
        model.encoder:forget()

        if i*batchSize%100 == 0 then
            print('error for Batch ' .. i .. ' is ' .. " Number Reviews : ".. i*batchSize .. " : ")
        end
    end
    print('Done')
end

