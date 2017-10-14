require 'nn'
require 'optim'

opt = {
    name = 'lstmv2', -- 'lstm', 'lstmv2',
    param_path = './checkpoints/lstmv2.3.para.t7',
    log = '',
    num_nonz = 3, -- 3,4,5,6,7,8,9
    gpu = 0,
    batch_size = 200,
    test_size = 1000,
    -- task related parameters
    -- task: y = Ax, given A recovery sparse x from y
    dataset = 'uniform', -- type of non-zero elements: uniform ([-1,-0.1]U[0.1,1]), unit (+-1)
    num_nonz = 3, -- number of non-zero elemetns to recovery: 3,4,5,6,7,8,9,10
    input_size = 20, -- dimension of observation vector y
    output_size = 100, -- dimension of sparse vector x

    -- model hyper parameters
    model = 'lstmv2', -- model: lstm, lstmv2, gru, gruv2
    rnn_size = 425, -- number of units in RNN cell
    num_layers = 2, -- number of stacked RNN layers
    num_unroll = 11, -- number of RNN unrolled time steps
}
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
-- print(opt)

torch.setnumthreads(4)
opt.manualSeed = torch.random(1, 10000) -- fix seed
-- print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

num_nonz = opt.num_nonz
batch_size = opt.batch_size
-- print(opt)
-- assert(opt.gpu > 0, 'please run on gpu')
require 'nngraph'
require 'cunn'
if opt.gpu > 0 then
    cutorch.setDevice(opt.gpu)
end
-- net = torch.load('./checkpoints.h425.uniform/' .. opt.checkpoints)
get_lstm = require ('model.' .. opt.model .. '.lua')
net = get_lstm(opt)
if opt.gpu == 0 then
    net:float()
else
    net:cuda()
end
paras, gradParas = net:getParameters()

print('Loading parameters')
paras:copy(torch.load(opt.param_path))
print('Done: network have ' .. paras:size(1) .. ' parameters')

matio = require 'matio'
if opt.gpu == 0 then
    batch_data = torch.FloatTensor(batch_size, opt.input_size)
    batch_zero_states = torch.FloatTensor(batch_size, opt.num_layers * opt.rnn_size * 2) -- init_states for lstm
else
    batch_data = torch.CudaTensor(batch_size, opt.input_size)
    batch_zero_states = torch.CudaTensor(batch_size, opt.num_layers * opt.rnn_size * 2) -- init_states for lstm
end
batch_label = torch.zeros(batch_size, opt.num_nonz) -- for MultiClassNLLCriterion LOSS
batch_zero_states:zero()
AccM, AccL, AccS = unpack(require 'accuracy')

mat_A = matio.load('./data/matrix_corr_unit_20_100.mat')['A']:t():float()
batch_X = torch.Tensor(batch_size, 100)
batch_n = torch.Tensor(batch_size, num_nonz)
local function gen_batch() -- batch_data, batch_label generating
    local bs = batch_size
    local len = 100 / num_nonz * num_nonz
    local perm = torch.randperm(100)[{{1,len}}]
    for i = 1, bs * num_nonz / len do
        perm = torch.cat(perm, torch.randperm(100)[{{1,len}}])
    end
    batch_label:copy(perm[{{1, bs * num_nonz}}]:reshape(bs, num_nonz))
    batch_X:zero()

    if opt.dataset == 'uniform' then
        batch_n:uniform(-0.4,0.4)
        batch_n[batch_n:gt(0)] = batch_n[batch_n:gt(0)] + 0.1
        batch_n[batch_n:le(0)] = batch_n[batch_n:le(0)] - 0.1
    end
    if opt.dataset == 'unit' then
        batch_n:uniform(-1,1)
        batch_n[batch_n:gt(0)] = 1
        batch_n[batch_n:le(0)] = -1
    end
    for i = 1, bs do
        for j = 1, num_nonz do
            batch_X[i][batch_label[i][j]] = batch_n[i][j]
        end
    end
    batch_data:copy(batch_X * mat_A)
end

tm = torch.Timer()

-- valid
tm:reset()
nbatch = 0
valid_accs = 0
valid_accl = 0
valid_accm = 0
for i = 1, opt.test_size, batch_size do
    gen_batch()
    local pred_prob = net:forward({batch_data, batch_zero_states})[1]:float()
    if (i==1) then
        matio.save('test_samples.mat',{X = batch_X:float(), label = batch_label:float(), predict = pred_prob:float()})
    end
    batch_accs = AccS(batch_label[{{},{1,num_nonz}}], pred_prob)
    valid_accs = valid_accs + batch_accs
    valid_accl = valid_accl + AccL(batch_label[{{},{1,num_nonz}}], pred_prob)
    valid_accm = valid_accm + AccM(batch_label[{{},{1,num_nonz}}], pred_prob)
    nbatch = nbatch + 1
end

print(("Test %d samples Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f"):format(opt.test_size, tm:time().real,
    valid_accs / nbatch, valid_accl / nbatch, valid_accm / nbatch))
