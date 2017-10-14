require 'nngraph'

all_layers = {}


function build_lstm_stacks(opt, input, prev_hs, prev_cs)
	local input_size, rnn_size, num_layers = opt.input_size, opt.rnn_size, opt.num_layers
	local next_hs = {}
	local next_cs = {}
	local layers = {}
	for L = 1, num_layers do
		local x
		if L == 1 then
			x = input
			x_size = input_size
		else
			x = next_hs[L-1]
			x_size = rnn_size
		end
		local l_i2h = nn.Linear(x_size, 4 * rnn_size)
		local l_h2h = nn.Linear(rnn_size, 4 * rnn_size)
		local l_bn = nn.BatchNormalization(4 * rnn_size)
		all_layers['l_i2h_' .. L] = l_i2h
		all_layers['l_h2h_' .. L] = l_h2h
		all_layers['l_bn_' .. L] = l_bn
		table.insert(layers, l_i2h)
		table.insert(layers, l_h2h)
		table.insert(layers, l_bn)

		local prev_c = prev_cs[L]
		local prev_h = prev_hs[L]

		local i2h = x - l_i2h
		local h2h = prev_h - l_h2h
		-- local all_sums = {i2h, h2h} - nn.CAddTable() - l_bn - nn.Reshape(4, rnn_size)
		local all_sums = {i2h, h2h} - nn.CAddTable() - nn.Reshape(4, rnn_size)
		local n1, n2, n3, n4 = (all_sums - nn.SplitTable(2)):split(4)
	    -- decode the gates
	    local in_gate = n1 - nn.Sigmoid()
	    local forget_gate = n2 - nn.Sigmoid()
	    local out_gate = n3 - nn.Sigmoid()
	    -- decode the write inputs
	    local in_transform = n4 - nn.Tanh()
	    -- update cells and hidden
        local next_c = nn.CAddTable()({
        	nn.CMulTable()({forget_gate, prev_c}),
        	nn.CMulTable()({in_gate, in_transform}),
      	})
  	    local next_h = nn.CMulTable()({out_gate, next_c - nn.Tanh()})
  	    next_hs[L] = next_h
  	    next_cs[L] = next_c
	end
	return next_hs, next_cs, layers
end

function do_share_parameters(layer, shared_layer)
	-- print('sharing  ' .. torch.type(layer) .. ' parameters')
	if layer.weight then
		layer.weight:set(shared_layer.weight)
		layer.gradWeight:set(shared_layer.gradWeight)
	end
	if layer.bias then
		layer.bias:set(shared_layer.bias)
		layer.gradBias:set(shared_layer.gradBias)
	end
end

function build_lstm_unrollnet(opt)
	local num_unroll, num_layers, rnn_size = opt.num_unroll, opt.num_layers, opt.rnn_size
	local init_hs = {}
	local init_cs = {}
	local input = - nn.Identity()
	local init_states_input = - nn.Identity()
	local init_states = init_states_input - nn.Reshape(num_layers * 2, rnn_size)
	local init_states_lst = {nn.SplitTable(2)(init_states):split(num_layers * 2)}
	print(#init_states_lst)
	for i = 1, num_layers do
		init_hs[i] = init_states_lst[i*2 - 1]:annotate{name = 'init_hidden_' .. i}
		init_cs[i] = init_states_lst[i*2]:annotate{name = 'init_cell_' .. i}
	end
	local outputs = {}
	-- unroll the lstm_stacks for many times
	local now_hs, now_cs = init_hs, init_cs
	local layers = {}
	for i = 1, num_unroll do
		now_hs, now_cs, layers[i] = build_lstm_stacks(opt, input, now_hs, now_cs)
		outputs[i] = now_hs[#now_hs] -- take topest lstm layer's hidden as the output
		for L = 1, num_layers do
			now_hs[L]:annotate{name='hid_'..i..'_'..L}
			now_cs[L]:annotate{name='cell_'..i..'_'..L}
		end
		-- print(layers[i])
	end
	local out_states_lst = {}
	for i = 1, num_layers do
		out_states_lst[i*2-1] = now_hs[i]
		out_states_lst[i*2] = now_cs[i]
	end
	local out_states = out_states_lst - nn.JoinTable(1,1)
	-- share weight
	for i = 2, num_unroll do
		for j = 1, #layers[i] do
			do_share_parameters(layers[i][j], layers[1][j])
		end
	end
	collectgarbage()
	local output = outputs - nn.JoinTable(1,1) -- concat the output of lstm in each time step as a big output
	return input, output, init_states_input, out_states
end

function get_lstm_net(opt)
	local num_unroll, num_layers, rnn_size, output_size = opt.num_unroll, opt.num_layers, opt.rnn_size, opt.output_size
	local lstm_input, lstm_output, init_states, out_states = build_lstm_unrollnet(opt)
	local l_pred_l = nn.Linear(num_unroll * rnn_size, output_size)
	local l_pred_bn = nn.BatchNormalization(output_size)
	local pred = lstm_output - l_pred_l
	-- - l_pred_bn - nn.ReLU(true)
	all_layers['l_pred_l'] = l_pred_l
	all_layers['l_pred_bn'] = l_pred_bn
	return nn.gModule({lstm_input:annotate{name='input'}, init_states:annotate{name='init_states'}},
		{pred:annotate{name='pred'}, out_states:annotate{name='out_states'}}), all_layers
end

local function test()
	local opt = {
		input_size = 20,
		output_size = 100,
		rnn_size = 100,
		num_layers = 3,
		num_unroll = 5,
	}
	local net = get_lstm_net(opt)
	-- graph.dot(net.bg, 'bg', './bg')
	-- graph.dot(net.fg, 'fg', './fg')
	local x = torch.rand(5,20)
	local z = torch.zeros(5,opt.rnn_size * opt.num_layers * 2)
	local y = net:forward({x,z})
    print(y[1]:mean(),y[1]:min(),y[1]:max())
    print(net:getParameters():size())
end

test()
return get_lstm_net, all_layers