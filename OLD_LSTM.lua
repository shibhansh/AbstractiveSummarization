require 'rnn'
require 'optim'
---------------------------------
print('Importing w2vutils')
wv=require 'w2vutils'
print('Done importing w2vutils')
---------------------------------
--batchSize = 50
batch_count=0
rho = 10000 
inputSize = 300
hiddenSize1 = 2000
hiddenSize2 = 2000 
hiddenSize3 = 600 
outputSize = 300 
seriesSize = 10000

--*************************************************************
print('Building the Model')
model = nn.Sequential()
model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize1, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize1,hiddenSize2 )))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize2, hiddenSize3, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize3, outputSize)))
--model:dropout(0.5)
criterion = nn.SequencerCriterion(nn.MSECriterion())
print('Done Building the model')
--*************************************************************

------------------------------------------
file_txt = io.open('text_words.csv','r')
file_smy = io.open('summary_words.csv','r')
------------------------------------------

function nextBatch()
    local inputs, targets = {}, {}
    ----------------------------------------------
    print('NEXTBATCH: Scanning Text Reviews and generating word vectors ...')

    line = file_txt:read()
    while line ~= "" and line~=nil do 
        --print(line)
        tmp=wv:word2vec(line)
        --print('==read________________________1')
        tmp2=torch.DoubleTensor(tmp:size()):copy(tmp)
        --print('NEXTBATCH: Before Insert')
        table.insert(inputs,tmp2)
        --print('NEXTBATCH: After Insert')
        tmp=nil
        tmp2=nil 
        line=file_txt:read()
    end
    table.insert(inputs, torch.DoubleTensor(300):zero())
    print('NEXTBATCH: Done with Text Reviews')

    print('NEXTBATCH: Scanning Summaries and generating word vectors ...')
    line = file_smy:read()
    while line ~= "" and line~= nil  do
        --print(line)
        tmp=wv:word2vec(line)
        --print('==read____________________________2')
        tmp2=torch.DoubleTensor(tmp:size()):copy(tmp)
        --print('NEXTBATCH: Before Insert')
        table.insert(targets,tmp2)
        --print('NEXTBATCH: After Insert')
        tmp1=nil
        tmp2=nil 
        line = file_smy:read()
    end 
    table.insert(inputs, torch.DoubleTensor(300):zero())
    print('NEXTBATCH: Done with Summaries')
    print('NEXTBATCH: Exiting')
    ------------------------------------------------
    return inputs, targets
end
--*************************************************************

-- get weights and gradient of loss wrt weights from the model
print('Getting Parameters')
x, dl_dx = model:getParameters()
print('Done')
--*************************************************************
feval = function(x_new)
    -- copy the weight if are changed
    if x ~= x_new then
        x:copy(x_new)
    end

    -- select a training batch
    print('FEVAL: Getting new Batch')
    local inputs, targets = nextBatch()
    print('FEVAL: Got New Batch')
    -- reset gradients (gradients are always accumulated, to accommodate
    -- batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    print('FEVAL: Going Forward')
    --print(inputs)
    local prediction=torch.Tensor(outputSize)
    for i = 1,#inputs do 
        prediction = model:forward({inputs[i]})
    end
    print('FEVAL: done with inputs')

    print('FEVAL: Calculating Loss and going Backward')
    --print('____________loss________________')
    local loss_x = criterion:forward({prediction[1]},{targets[1]})
    --print(prediction[1]:size())
    --print(targets[1]:size())
    --print(inputs)
    model:backward({inputs[#inputs]}, criterion:backward({prediction[1]}, {targets[1]}))
    
    for i=2,#targets do 
        --print('_______________________')
        --print(i)
        prediction=model:forward({prediction[1]})
        local loss_x = criterion:forward({prediction[1]},{targets[i]})
        model:backward({prediction[1]}, criterion:backward({prediction[1]}, {targets[i]}))
    end 
    print('FEVAL: Done')
   -- print('FEVAL: Going Backward')
   --model:backward(inputs, criterion:backward(prediction, targets))
   --print('FEVAL: Done')
   batch_count=batch_count+1
    print('FEVAL: ---------------Done with feval: count: '..batch_count..'; Loss: '..loss_x)
    return loss_x, dl_dx
end
--*************************************************************

print('Getting sgd_arams')
sgd_params = {
    learningRate = 1e-2,
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}
print('Done')

print('Training')
for i = 1, 9990 do
    -- train a mini_batch of batchSize in parallel
    _, fs = optim.sgd(feval,x, sgd_params)

    if sgd_params.evalCounter % 100 == 0 then
        print(model)
        torch.save('model_tmp', model)
        print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
    end
end
print('Done')
