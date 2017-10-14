-- match number
local function Accuracy(label, pred_prob)
    local num_nonz = label:size(2)
    local _, pred = pred_prob:topk(num_nonz, true)
    pred = pred:float()
    local t_score = torch.zeros(label:size())
    for i = 1, num_nonz do
        for j = 1, num_nonz do
            t_score[{{},i}]:add(label[{{},i}]:eq(pred[{{},j}]):float())
        end
    end
    return t_score:mean()
end

-- loose match
local function Accuracy2(label, pred_prob)
    local num_nonz = label:size(2)
    local _, pred = pred_prob:topk(20, true)
    pred = pred:float()
    local t_score = torch.zeros(label:size())
    for i = 1, num_nonz do
        for j = 1, 20 do
            t_score[{{},i}]:add(label[{{},i}]:eq(pred[{{},j}]):float())
        end
    end
    return t_score:mean()
end

-- strict match
local function Accuracy3(label, pred_prob)
    local num_nonz = label:size(2)
    local _, pred = pred_prob:topk(num_nonz, true)
    pred = pred:float()
    local t_score = torch.zeros(label:size())
    for i = 1, num_nonz do
        for j = 1, num_nonz do
            t_score[{{},i}]:add(label[{{},i}]:eq(pred[{{},j}]):float())
        end
    end
    return t_score:sum(2):eq(num_nonz):sum() * 1. / pred:size(1)
end

return {Accuracy, Accuracy2, Accuracy3}