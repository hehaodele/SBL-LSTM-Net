require 'nngraph'

all_layers = {}


function build_gru_stacks(opt, input, prev_hs)
    local input_size, rnn_size, num_layers = opt.input_size, opt.rnn_size, opt.num_layers
    local next_hs = {}
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
        local l_i2h = nn.Linear(x_size, 3 * rnn_size)
        local l_h2h = nn.Linear(rnn_size, 3 * rnn_size)
        table.insert(layers, l_i2h)
        table.insert(layers, l_h2h)

        local prev_h = prev_hs[L]
        local i2h = x - l_i2h - nn.Reshape(3, rnn_size)
        local h2h = prev_h - l_h2h - nn.Reshape(3, rnn_size)

        local n1, n2, n3 = (i2h - nn.SplitTable(2)):split(3)
        local m1, m2, m3 = (h2h - nn.SplitTable(2)):split(3)
        -- decode the gates
        local reset_gate = nn.CAddTable()({n1, m1}) - nn.Sigmoid()
        local update_gate = nn.CAddTable()({n2, m2}) - nn.Sigmoid()
        -- decode the write inputs
        local in_transform = nn.CAddTable()({n3, nn.CMulTable()({reset_gate, m3})}) - nn.Tanh()
        -- update cells and hidden
        local next_h = nn.CAddTable()({
            nn.CMulTable()({update_gate - nn.MulConstant(-1) - nn.AddConstant(1), prev_h}),
            nn.CMulTable()({update_gate, in_transform}),
        })
        next_hs[L] = next_h
    end
    return next_hs, layers
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

function build_gru_unrollnet(opt)
    local num_unroll, num_layers, rnn_size = opt.num_unroll, opt.num_layers, opt.rnn_size
    local init_hs = {}
    local input = - nn.Identity()
    local init_states_input = - nn.Identity()
    local init_states = init_states_input - nn.Reshape(num_layers, rnn_size)
    local init_states_lst = {nn.SplitTable(2)(init_states):split(num_layers)}
    print(#init_states_lst)
    for i = 1, num_layers do
        init_hs[i] = init_states_lst[i]:annotate{name = 'init_hidden_' .. i}
    end
    local outputs = {}
    -- unroll the gru_stacks for many times
    local now_hs = init_hs
    local layers = {}
    for i = 1, num_unroll do
        now_hs, layers[i] = build_gru_stacks(opt, input, now_hs)
        outputs[i] = now_hs[#now_hs] -- take topest gru layer's hidden as the output
        for L = 1, num_layers do
            now_hs[L]:annotate{name='hid_'..i..'_'..L}
        end
        -- print(layers[i])
    end
    local out_states_lst = {}
    for i = 1, num_layers do
        out_states_lst[i] = now_hs[i]
    end
    local out_states = out_states_lst - nn.JoinTable(1,1)
    -- share weight
    for i = 2, num_unroll do
        for j = 1, #layers[i] do
            do_share_parameters(layers[i][j], layers[1][j])
        end
    end
    collectgarbage()
    local output = outputs - nn.JoinTable(1,1) -- concat the output of gru in each time step as a big output
    return input, output, init_states_input, out_states
end

function get_gru_net(opt)
    local num_unroll, num_layers, rnn_size, output_size = opt.num_unroll, opt.num_layers, opt.rnn_size, opt.output_size
    local gru_input, gru_output, init_states, out_states = build_gru_unrollnet(opt)
    local l_pred_l = nn.Linear(num_unroll * rnn_size, output_size)
    local pred = gru_output - l_pred_l
    all_layers['l_pred_l'] = l_pred_l
    all_layers['l_pred_bn'] = l_pred_bn
    return nn.gModule({gru_input:annotate{name='input'}, init_states:annotate{name='init_states'}},
        {pred:annotate{name='pred'}, out_states:annotate{name='out_states'}}), all_layers
end

local function test()
    opt = {
        input_size = 20,
        output_size = 100,
        rnn_size = 680,
        num_layers = 2,
        num_unroll = 11,
    }
    net = get_gru_net(opt)
    pms, gms = net:getParameters()
    print(pms:size())
    -- graph.dot(net.bg, 'bg', './bg')
    -- graph.dot(net.fg, 'fg', './fg')
    -- x = torch.rand(5,20)
    -- z = torch.zeros(5,opt.rnn_size * opt.num_layers)
    -- y = net:forward({x,z})
    -- print(y[1]:mean(),y[1]:min(),y[1]:max())
end
-- test()
return get_gru_net