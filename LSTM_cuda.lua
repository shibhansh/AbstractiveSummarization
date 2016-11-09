require 'rnn'
require 'optim'
require 'seq2seq'
require 'cutorch'
require 'cunn'
---------------------------------

cutorch.setDevice(1)
torch.setdefaulttensortype('torch.CudaTensor')
print('Running with CUDA on GPU')
----------------------------------

file_txt=io.open('text_words.csv','r')
file_smy=io.open('summary_words.csv','r')

batchSize = 10
numDocuments = 50
hiddenSize = 1000

print('Begin')

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
feval = function(x_new)
    if x ~= x_new then
        x:copy(x_new)
    end
     
    dl_dx:zero()

    --print('---->getting NextBatch()')
    encoderInputs, decoderInputs, decoderTargets = nextBatch()
    --print('---->Done getting NextBatch()')

    --forward pass
    --print('---->encoder forward')
    encoderOutput = model.encoder:forward(encoderInputs) 
    --print('---->DONE encoder forward')

    --print('---->forwardConnect')
    model:forwardConnect(encoderInputs:size(1))
    --print('---->DONE encoder forward')

    --print('---->decoder forward')
    decoderOutput = model.decoder:forward(decoderInputs)
    --print('---->DONE decoder forward')

    --print('---->criterion forward')
    local loss_x = model.criterion:forward(decoderOutput, decoderTargets)
    --print('---->DONE criterion forward')
   
    --backward pass
    --print('---->criterion backward')
    dloss_doutput = model.criterion:backward(decoderOutput, decoderTargets)
    --print('---->DONE criterion backward')

    --print('---->decoder backward')
    model.decoder:backward(decoderInputs, dloss_doutput)
    --print('---->DONE decoder backward')

    --print('---->backwardConnect')
    model:backwardConnect()
    --print('---->DONE backwardConnect')
 
    
    --print('---->encoder backward')
    model.encoder:backward(encoderInputs, encoderOutput:zero())
    --print('---->DONE encoder backward') 


    --print(encoderInputs)
    wordIds, probabilities = model:eval(encoderInputs:select(2,10))
    for _tmp, ind in ipairs(wordIds) do 
        local word = Index2Vocab[wordId[1]]
        print(word)
    end

    print('---->DONE with feval')
    return loss_x, dl_dx
end

--*************************************************************

function getVocab()
    numDocumentsTraversed = 0
    VocabSize = 0
    Vocab = {}
    Index2Vocab = {}
    file_txt=io.open('text_words.csv','r')
    file_smy=io.open('summary_words.csv','r')

    --insert <eos>, <go> and <unknown> tokens

    Vocab['<go>'] = 1
    Vocab['<eos>'] = 2
    Vocab['<unknown>'] = 3
    Index2Vocab[1] = '<go>'
    Index2Vocab[2] = '<eos>'
    Index2Vocab[3] = '<unknown>'
    VocabSize = 3

    line = file_txt:read():lower()
    while line~=nil do
        if Vocab[line] == nil then 
            VocabSize = VocabSize + 1
            Vocab[line] = VocabSize
            Index2Vocab[VocabSize] = line
        end
        
       line = file_txt:read()
       if (line ~= nil) then
           line=line:lower()
       end

       if line == "" then
           numDocumentsTraversed = numDocumentsTraversed +1
       end
    
       if numDocumentsTraversed > numDocuments then 
           break
       end
    end

    numDocumentsTraversed = 0
    line = file_smy:read():lower()
    while line~=nil do 
        if Vocab[line] == nil then
            VocabSize = VocabSize + 1
            Vocab[line] = VocabSize 
            Index2Vocab[VocabSize] = line
        end
            --print(':::',sentBatch_smy[j][j2],"::",j,j2,#decoderTargets)
    
        line = file_smy:read()
        if(line ~= nil) then
            line=line:lower()
        end
       
        if line == "" then
           numDocumentsTraversed = numDocumentsTraversed +1
       end

       if numDocumentsTraversed > numDocuments then 
           break
       end
    end
end

--*************************************************************

--Get Vocab Table 
print("Getting Vocabulary...")
getVocab()
print("Done")

print('Building Model')
model = Seq2Seq(VocabSize,hiddenSize)
model.goToken = Vocab['<go>']
model.eosToken = Vocab['<eos>']

if batchSize > 1 then 
    model.criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(nn.ClassNLLCriterion(),1))
else
    model.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())
end

model:cuda()

print('Done')

-- get weights and gradient of loss wrt weights from the model
print('Getting Parameters')
x, dl_dx = model:getParameters()
print('Done')

print('Getting sgd_params')
sgd_params = {
    learningRate = 5e-1,--changed from 1e-2
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0.5
}
print('Done')

print('Training')

for j=1,10000 do
	print("EPOCH: "..j)
	file_txt = io.open('text_words.csv','r')
	file_smy = io.open('summary_words.csv','r')

    
    for i = 1, numDocuments/batchSize do	
	_, fs = optim.sgd(feval,x, sgd_params)
        
    model.decoder:forget()
    model.encoder:forget()

    print('error for Batch ' .. sgd_params.evalCounter  .. ' is ' .. " Number Reviews : ".. sgd_params.evalCounter*batchSize .. " : ".. fs[1])

	if sgd_params.evalCounter%1000 == 0 then
        torch.save('model_tmp', model)
    end
end
end
print('Done')
