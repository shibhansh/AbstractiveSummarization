model.encoder = nn.Sequential()
model.encoder:add(nn.Sequencer(nn.Linear(model.vocabSize, model.hiddenSize)))
--model.encoder:add(nn.LookupTableMaskZero(model.vocabSize, model.hiddenSize))
model.encoderLSTM = nn.FastLSTM(model.hiddenSize, model.hiddenSize):maskZero(1)
model.encoder:add(nn.Sequencer(model.encoderLSTM))
--model.encoder:add(nn.Select(1,-1))

model.decoder = nn.Sequential()
--model.decoder:add(nn.LookupTableMaskZero(model.vocabSize, model.hiddenSize))
model.decoderLSTM = nn.FastLSTM(model.hiddenSize, model.hiddenSize):maskZero(1)
model.decoder:add(nn.Sequencer(model.decoderLSTM))
model.decoder:add(nn.Sequencer(nn.MaskZero(nn.Linear(model.hiddenSize, model.vocabSize),1)))
--model.decoder:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(),1)))

model.encoder:zeroGradParameters()
model.decoder:zeroGradParameters()

