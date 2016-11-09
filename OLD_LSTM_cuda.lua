require 'rnn'
require 'optim'
---------------------------------
print('Importing w2vutils')
--wv=require 'w2vutils'
print('Done importing w2vutils')

require 'cutorch'
require 'cunn'

cutorch.setDevice(1)
--torch.setdefaulttensortype(torch.CudaTensor)
print('Running with CUDA on GPU')
----------------------------------

--batchSize = 50
batch_count=0
--rho = 10000
rho = 5 
inputSize = 300
--lstmOutputSize1 = 10
--lstmLayerSize1 =100
--hiddenSize1 = lstmOutputSize1*lstmLayerSize1
hiddenSize1=20000
hiddenSize2 = 2000
hiddenSize3 = 2000
outputSize = 300 

print('Begin')

--*************************************************************
print('Building the Model')
model = nn.Sequential()
--model:add(nn.Sequencer(LSTM_Layer_1))
LSTM_Layer_1=nil
print('added lstm layer 1')
--model:add(nn.Sequencer(nn.FastLSTM(inputSize, hiddenSize1, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize1,hiddenSize2 )))
model:add(nn.Sequencer(nn.FastLSTM(hiddenSize2, hiddenSize3, rho)))
model:add(nn.Sequencer(nn.Linear(hiddenSize3, outputSize)))
--model:dropout(0.5)
criterion = nn.SequencerCriterion(nn.MSECriterion())
model=model:cuda()
criterion=criterion:cuda()
print('Done Building the model')
--*************************************************************

function nextBatch()
    local inputs, targets = {}, {}
    print('	    ----------------------------------------------')
    line = file_txt:read()
    while line ~= "" and line~=nil do 
  	tmp=wv:word2vec(line)
        tmp2=torch.DoubleTensor(tmp:size()):copy(tmp)
	
--	for i=1,lstmLayerSize1-1 do
--		tmp2=torch.cat(tmp2,tmp2)	
--	end
        table.insert(inputs,tmp2)
        tmp=nil
        tmp2=nil 
        line=file_txt:read()
    end

    table.insert(inputs, torch.DoubleTensor(300):zero()) 
    line = file_smy:read()
    
    while line ~= "" and line~= nil  do
        tmp=wv:word2vec(line)
        tmp2=torch.DoubleTensor(tmp:size()):copy(tmp)
        table.insert(targets,tmp2)
        tmp1=nil
        tmp2=nil 
        line = file_smy:read()
    end 
 
    table.insert(inputs, torch.DoubleTensor(300):zero())
    ------------------------------------------------
    return inputs, targets
end
--*************************************************************

print('---------')
--*************************************************************
feval = function(x_new)
    -- copy the weight if are changed
    if x ~= x_new then
        x:copy(x_new)
    end

    -- select a training batch
    local inputs, targets = nextBatch()

print('paospasope')
    -- reset gradients (gradients are always accumulated, to accommodate
    -- batch methods)
    dl_dx:zero()

    -- evaluate the loss function and its derivative wrt x, given mini batch
    local prediction=torch.Tensor(outputSize)

    for i = 1,#inputs do 
        prediction = model:forward({inputs[i]:cuda()})
    end

    local loss_x = criterion:forward({prediction[1]:cuda()},{targets[1]:cuda()})
    model:backward({inputs[#inputs]:cuda()}, criterion:backward({prediction[1]:cuda()}, {targets[1]:cuda()}))

    for i=2,#targets do 
        prediction=model:forward({prediction[1]:cuda()})
        local loss_x = criterion:forward({prediction[1]:cuda()},{targets[i]:cuda()})
        model:backward({prediction[1]:cuda()}, criterion:backward({prediction[1]:cuda()}, {targets[i]:cuda()}))
    end 
    batch_count=batch_count+1
  
    return loss_x, dl_dx
end
--*************************************************************



-- get weights and gradient of loss wrt weights from the model
print('Getting Parameters')
x, dl_dx = model:getParameters()
print('Done')

print('Getting sgd_arams')
sgd_params = {
    learningRate = 1e-1,--changed from 1e-2
    learningRateDecay = 1e-4,
    weightDecay = 0,
    momentum = 0
}
print('Done')

print('Training')
for j=1,10 do
	print("EPOCH: "..j)
	------------------------------------------
	file_txt = io.open('text_words.csv','r')
	file_smy = io.open('summary_words.csv','r')
	------------------------------------------
	for i = 1, 19990 do
	    -- train a mini_batch of batchSize in parallel
	    _, fs = optim.sgd(feval,x, sgd_params)

	    if sgd_params.evalCounter % 100 == 0 then
		torch.save('model_tmp', model)
		print('error for iteration ' .. sgd_params.evalCounter  .. ' is ' .. fs[1] / rho)
	    end
	end
end
print('Done')
