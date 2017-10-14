require 'nngraph'
-- gated feedback gru
core_layers = {}
function build_gf_gru_stacks(opt, input, prev_hs)
    local input_size, rnn_size, num_layers = opt.input_size, opt.rnn_size, opt.num_layers
    local next_hs = {}
    local layers = {}
    local prev_H = prev_hs - nn.JoinTable(1,1)
    for L = 1, num_layers do
        local x
        if L == 1 then
            x = input
            x_size = input_size
        else
            x = next_hs[L-1]
            x_size = rnn_size
        end
        local l_i2h = nn.Linear(x_size, 2 * rnn_size)
        local l_h2h = nn.Linear(rnn_size, 2 * rnn_size)
        local l_bn = nn.BatchNormalization(2 * rnn_size)
        core_layers['l_i2h_' .. L] = l_i2h
        core_layers['l_h2h_' .. L] = l_h2h
        core_layers['l_bn_' .. L] = l_bn
        table.insert(layers, l_i2h)
        table.insert(layers, l_h2h)
        table.insert(layers, l_bn)

        local prev_h = prev_hs[L]

        local i2h = x - l_i2h
        local h2h = prev_h - l_h2h
        local all_sums = {i2h, h2h} - nn.CAddTable() - nn.Reshape(2, rnn_size)
        local n1, n2 = (all_sums - nn.SplitTable(2)):split(2)
        -- decode the gates
        local reset_gate = n1 - nn.Sigmoid()
        local update_gate = n2 - nn.Sigmoid()
        -- docode the global gates
        local global_gates = {}
        for i = 1, num_layers do
            local l_wg_iL = nn.Linear(x_size, rnn_size)
            local l_ug_iL = nn.Linear(rnn_size * num_layers, rnn_size)
            table.insert(layers, l_wg_iL)
            table.insert(layers, l_ug_iL)
            local i2L_gate = {x - l_wg_iL, prev_H - l_ug_iL} - nn.CAddTable() - nn.Sigmoid()
            if(i == L) then
                core_layers['l_wg_LL_' .. L] = l_wg_iL
                core_layers['l_ug_LL_' .. L] = l_ug_iL
            end
            global_gates[i] = i2L_gate
        end
        -- decode the write inputs
        local l_wc = nn.Linear(x_size, rnn_size)
        table.insert(layers, l_wc)
        core_layers['l_wc_' .. L] = l_wc
        local in_from_input = x - l_wc
        local in_from_prev_hs = {}
        for i = 1, num_layers do
            local l_uc_iL = nn.Linear(rnn_size, rnn_size)
            table.insert(layers, l_uc_iL)
            if(i == L) then
                core_layers['l_uc_LL_' .. L] = l_uc_iL
            end
            in_from_prev_hs[i] = nn.CMulTable()({global_gates[i], prev_hs[i] - l_uc_iL})
        end
        local in_from_prev_hs_after_reset = {in_from_prev_hs - nn.CAddTable(), reset_gate} - nn.CMulTable()
        local in_transform = {in_from_input, in_from_prev_hs_after_reset} - nn.CAddTable() - nn.Tanh()
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

function build_gf_gru_unrollnet(opt)
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
        now_hs, layers[i] = build_gf_gru_stacks(opt, input, now_hs)
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

function get_gf_gru_net(opt)
    local num_unroll, num_layers, rnn_size, output_size = opt.num_unroll, opt.num_layers, opt.rnn_size, opt.output_size
    local gru_input, gru_output, init_states, out_states = build_gf_gru_unrollnet(opt)
    local l_pred_l = nn.Linear(num_unroll * rnn_size, output_size)
    local l_pred_bn = nn.BatchNormalization(output_size)
    local pred = gru_output - l_pred_l
    core_layers['l_pred_l'] = l_pred_l
    core_layers['l_pred_bn'] = l_pred_bn
    return nn.gModule({gru_input:annotate{name='input'}, init_states:annotate{name='init_states'}},
        {pred:annotate{name='pred'}, out_states:annotate{name='out_states'}}), core_layers
end

local function test()
    opt = {
        input_size = 20,
        output_size = 100,
        rnn_size = 455,
        num_layers = 2,
        num_unroll = 11,
    }
    net = get_gf_gru_net(opt)
    pms, gms = net:getParameters()
    print(pms:size())
    -- graph.dot(net.bg, 'bg', './bg')
    -- graph.dot(net.fg, 'fg', './fg')
    -- x = torch.rand(5, opt.input_size)
    -- z = torch.zeros(5, opt.rnn_size * opt.num_layers)
    -- y = net:forward({x,z})
    -- print(y[1]:mean(),y[1]:min(),y[1]:max())
end

-- test()
return get_gf_gru_net