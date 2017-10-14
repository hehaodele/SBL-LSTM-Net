function rmsprop(opfunc, x, config, state)
   -- (0) get/update state
   local config = config or {}
   local state = state or config
   local lr = config.learningRate or 0.002
   local gamma1 = config.gamma1 or 0.95
   local gamma2 = config.gamma2 or 0.9
   local epsilon = config.epsilon or 1e-4
   local wd = config.weightDecay or 0
   local mfill = config.initialMean or 0

   -- (1) evaluate f(x) and df/dx
   local fx, dfdx = opfunc(x)


   -- (3) initialize mean square values and square gradient storage
   if not state.n then
      state.n = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.g = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.delta = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(mfill)
      state.tmp = torch.Tensor():typeAs(x):resizeAs(dfdx):fill(0)
   end

   -- (4) calculate new (leaky) mean squared values
   state.n:mul(gamma1)
   state.n:addcmul(1.0 - gamma1, dfdx, dfdx)
   state.g:mul(gamma1)
   state.g:add(1.0 - gamma1, dfdx)
   state.delta:mul(gamma2)

   -- state.test = state.n - torch.cmul(state.g, state.g)
   state.tmp = torch.sqrt(state.n - torch.cmul(state.g, state.g) + epsilon)
   state.delta:addcdiv(-lr, dfdx, state.tmp)
   -- (2) weight decay
   if wd ~= 0 then
      state.delta:add(-lr * wd, x)
   end
   -- (5) perform update
   x:add(state.delta)
   -- return x*, f(x) before optimization
   return x, {fx}
end

return rmsprop