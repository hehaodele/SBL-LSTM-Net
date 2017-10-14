require 'nn'
require 'optim'

opt = {
    -- training hyper parameters
	gpu = 1, -- gpu id
	batch_size = 250, -- training batch size
    lr = 0.001, -- basic learning rate
    lr_decay_startpoint = 250, -- learning rate from which epoch
    num_epochs = 400, -- total training epochs
    max_grad_norm = 5.0,
    clip_gradient = 4.0,

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
print(opt)

torch.setnumthreads(4)
opt.manualSeed = torch.random(1, 10000)
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

num_nonz = opt.num_nonz
batch_size = opt.batch_size

LOSS = (require 'MultiClassNLLCriterion')()
myrmsprop = require 'myrmsprop'

assert(opt.gpu > 0, 'please run on gpu')
require 'nngraph'
require 'cunn'
cutorch.setDevice(opt.gpu)

get_lstm = require ('model.' .. opt.model .. '.lua')
net = get_lstm(opt)
net:cuda();

paras, gradParas = net:getParameters()
print('network have ' .. paras:size(1) .. ' parameters')

matio = require 'matio'
-- if opt.dataset == 'uniform' then
--     data_file = ('../data/data_'.. opt.dataset .. '_d_'.. opt.num_nonz ..'.bo.mat')
--     print(data_file)
--     data = matio.load(data_file)
--     print(data)
--     train_data = data['Output'][{{1,600000}}]:float()
--     train_label = data['Label'][{{1,600000}}]:float() + 1
--     valid_data = data['Output'][{{600001,700000}}]:float()
--     valid_label = data['Label'][{{600001,700000}}]:float() + 1
--     print('Loading data done!')
--     train_size = train_data:size(1)
--     valid_size = valid_data:size(1)
-- end
-- if opt.dataset == 'unit' then
    train_size = 600000
    valid_size = 100000
    valid_data = torch.zeros(valid_size, opt.input_size)
    valid_label = torch.zeros(valid_size, opt.num_nonz)
-- end

batch_data = torch.CudaTensor(batch_size, opt.input_size)
batch_label = torch.zeros(batch_size, opt.num_nonz) -- for MultiClassNLLCriterion LOSS
batch_zero_states = torch.CudaTensor(batch_size, opt.num_layers * opt.rnn_size * 2) -- init_states for lstm
if opt.model == 'gru' or opt.model == 'gruv2' then
    batch_zero_states:resize(batch_size, opt.num_layers * opt.rnn_size) -- init_states for gru
end
batch_zero_states:zero()

AccM, AccL, AccS = unpack(require 'accuracy')



err = 0
function fx(x)
	gradParas:zero()
	local pred_prob = net:forward({batch_data, batch_zero_states})[1]:float()
	err = LOSS:forward(pred_prob, batch_label)
	local df_dpred = LOSS:backward(pred_prob, batch_label)
	net:backward({batch_data, batch_zero_states}, {df_dpred:cuda(), batch_zero_states})
	gradParas:clamp(-4.0, 4.0)
	local gnorm = gradParas:norm()
	if gnorm > opt.max_grad_norm then
		gradParas:mul(opt.max_grad_norm / gnorm)
	end
	return err, gradParas
end

function do_fx(x)
	local pred_prob = net:forward({batch_data, batch_zero_states})[1]:float()
	err = LOSS:forward(pred_prob, batch_label)
	return err
end

opt.model_all = opt.model .. '.l_' .. opt.num_layers .. '.t_' .. opt.num_unroll .. '.rnn_' .. opt.rnn_size
logger_file = opt.model_all .. '.' .. opt.dataset .. '.' .. num_nonz .. '.log'
logger = io.open(logger_file, 'w')
for k,v in pairs(opt) do logger:write(k .. ' ' .. v ..'\n') end
logger:write('network have ' .. paras:size(1) .. ' parameters' .. '\n')
logger:close()

mat_A = matio.load('./data/matrix_corr_unit_20_100.mat')['A']:t():float()
batch_X = torch.Tensor(batch_size, 100)
batch_n = torch.Tensor(batch_size, num_nonz)
local function gen_batch()
-- generate training data
-- batch_data, batch_label generating
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

-- generate a fixed validation set
print('building validation set')
for i = 1, valid_size, batch_size do
    gen_batch()
    valid_data[{{i,i+batch_size-1},{}}]:copy(batch_data)
    valid_label[{{i,i+batch_size-1},{}}]:copy(batch_label)
end
print('done')

best_valid_accs = 0
base_epoch = opt.lr_decay_startpoint
base_lr = opt.lr
optimState = {
    learningRate = 0.001,
    weightDecay = 0.0001,
}
tm = torch.Timer()
for epoch = 1, opt.num_epochs  do
    -- learing rate self-adjustment
    if epoch > 250 then
        optimState.learningRate = base_lr / (1 + 0.06 * (epoch - base_epoch))
        if(epoch % 50 == 0) then base_epoch = epoch; base_lr = base_lr * 0.25; end
    end
    logger = io.open(logger_file, 'a')
	-- train
	train_accs = 0
	train_accl = 0
	train_accm = 0
	train_err = 0
	nbatch = 0
	tm:reset()
	for i = 1, train_size, batch_size do
        gen_batch()
		myrmsprop(fx, paras, optimState)
		batch_accs = AccS(batch_label[{{},{1,num_nonz}}], net.output[1]:float())
		batch_accl = AccL(batch_label[{{},{1,num_nonz}}], net.output[1]:float())
		batch_accm = AccM(batch_label[{{},{1,num_nonz}}], net.output[1]:float())
		train_accs = train_accs + batch_accs
		train_accl = train_accl + batch_accl
		train_accm = train_accm + batch_accm
		train_err = train_err + err
		nbatch = nbatch + 1
		if nbatch % 512 == 1 then
			print(('%.4f %.4f %.4f err %.4f'):format(batch_accs, batch_accl, batch_accm, err))
		end
	end
    print(("Train [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f"):format(epoch, tm:time().real,
    	train_accs / nbatch, train_accl / nbatch, train_accm / nbatch, train_err / nbatch))
   	logger:write(("Train [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f\n"):format(epoch, tm:time().real,
    	train_accs / nbatch, train_accl / nbatch, train_accm / nbatch, train_err / nbatch))

	-- eval
	tm:reset()
	nbatch = 0
	valid_accs = 0
	valid_accl = 0
	valid_accm = 0
	valid_err = 0
	for i = 1, valid_size, batch_size do
		batch_data:copy(valid_data[{{i,i+batch_size-1},{}}])
		batch_label[{{},{1,num_nonz}}]:copy(valid_label[{{i,i+batch_size-1},{}}])
		do_fx()
		batch_accs = AccS(batch_label[{{},{1,num_nonz}}], net.output[1]:float())
		batch_accl = AccL(batch_label[{{},{1,num_nonz}}], net.output[1]:float())
		batch_accm = AccM(batch_label[{{},{1,num_nonz}}], net.output[1]:float())
		valid_accs = valid_accs + batch_accs
		valid_accl = valid_accl + batch_accl
		valid_accm = valid_accm + batch_accm
		valid_err = valid_err + err
		nbatch = nbatch + 1
	end
    print(("Valid [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f"):format(epoch, tm:time().real,
    	valid_accs / nbatch, valid_accl / nbatch, valid_accm / nbatch, valid_err / nbatch))
    logger:write(("Valid [%d] Time %.3f s-acc %.4f l-acc %.4f m-acc %.4f err %.4f\n"):format(epoch, tm:time().real,
    	valid_accs / nbatch, valid_accl / nbatch, valid_accm / nbatch, valid_err / nbatch))

	if valid_accs > best_valid_accs then
		best_valid_accs = valid_accs
   		print('saving model')
   		logger:write('saving model\n')
        torch.save('./checkpoints/'.. opt.model..'.' .. num_nonz .. '.para.t7', paras) -- clearState may lead 'stack overflow' when num_unroll=33
   	end
    if epoch % 100 == 0 then
    	print('saving model')
   		-- paras, gradParas = nil, nil
        -- torch.save('./checkpoints/'.. opt.model..'.' .. num_nonz .. '.' .. epoch .. '.t7', net:clearState())
   		-- paras, gradParas = net:getParameters()
        torch.save('./checkpoints/'.. opt.model..'.' .. num_nonz .. '.' .. epoch .. '.para.t7', paras) -- clearState may lead 'stack overflow' when num_unroll=33
    end
    logger:close()
    if epoch == opt.lr_decay_startpoint then
        optimState = {
            learningRate = 0.001,
            weightDecay = 0.0001,
        }
    end
end