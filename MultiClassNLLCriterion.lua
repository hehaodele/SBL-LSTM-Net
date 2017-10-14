local MultiClassNLLCriterion, Criterion  = 
torch.class('nn.MultiClassNLLCriterion', 'nn.Criterion')

function MultiClassNLLCriterion:__init()
	Criterion.__init(self)
	self.lsm = nn.LogSoftMax()
	self.nll = nn.ClassNLLCriterion()
end

function MultiClassNLLCriterion:updateOutput(input, target)
	self.lsm:updateOutput(input)
	self.output = 0
	for i = 1, target:size(2) do
		self.nll:updateOutput(self.lsm.output, target[{{},{i}}]:squeeze())
		self.output = self.output + self.nll.output
	end
	return self.output
end

function MultiClassNLLCriterion:updateGradInput(input, target)
	for i = 1, target:size(2) do
		self.nll:updateGradInput(self.lsm.output, target[{{},{i}}]:squeeze())
		if i == 1 then
			self.gradInput = self.nll.gradInput:clone()
		else
			self.gradInput:add(self.nll.gradInput)
		end
	end
	self.lsm:updateGradInput(input, self.gradInput)
	self.gradInput = self.lsm.gradInput
	return self.gradInput
end

return nn.MultiClassNLLCriterion