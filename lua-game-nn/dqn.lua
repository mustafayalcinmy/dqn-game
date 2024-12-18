require 'torch'
require 'nn'
require 'optim'

local DQN = {}
local targetNetwork = nn.Sequential()
DQN.stateSize = 10 -- Number of state inputs
DQN.actionSize = 2 -- Number of possible actions
DQN.hiddenSize = 256 -- Size of the hidden layer
DQN.gamma = 0.95 -- Discount factor
DQN.learningRate = 0.0005 -- Learning rate
DQN.epsilon = 0.2 -- Exploration rate
DQN.targetUpdateFreq = 100 -- Frequency of target network updates
DQN.replayBuffer = {} -- Replay buffer to store experiences
DQN.batchSize = 32 -- Size of batches for training

-- Define the Q-Network
DQN.network = nn.Sequential()
DQN.network:add(nn.Linear(DQN.stateSize, DQN.hiddenSize))
DQN.network:add(nn.ReLU())
DQN.network:add(nn.Linear(DQN.hiddenSize, DQN.actionSize))

-- Copy the network to targetNetwork
for i = 1, #DQN.network.modules do
    targetNetwork:add(DQN.network.modules[i]:clone())
end

-- Define the loss function
DQN.criterion = nn.MSECriterion()

-- Function to select an action using epsilon-greedy strategy
function DQN:selectAction(state)
    if math.random() < self.epsilon then
        return math.random(1, self.actionSize) -- Random action
    else
        local qValues = self.network:forward(state)
        local _, action = torch.max(qValues, 1)
        return action[1]
    end
end

-- Function to add experience to replay buffer
function DQN:addExperience(state, action, reward, nextState)
    table.insert(self.replayBuffer, {state, action, reward, nextState})
    if #self.replayBuffer > 10000 then
        table.remove(self.replayBuffer, 1)
    end
end

-- Function to train the Q-Network with batch updates
function DQN:train(state, action, reward, nextState)
    -- Add experience to the replay buffer
    self:addExperience(state, action, reward, nextState)

    -- Wait until buffer has enough experiences for a batch
    if #self.replayBuffer < self.batchSize then
        return
    end

    -- Sample a batch of experiences
    local batch = {}
    for i = 1, self.batchSize do
        local idx = math.random(1, #self.replayBuffer)
        table.insert(batch, self.replayBuffer[idx])
    end

    -- Batch processing setup
    local states = torch.Tensor(self.batchSize, self.stateSize)
    local targets = torch.Tensor(self.batchSize, self.actionSize)
    
    for i, experience in ipairs(batch) do
        local s, a, r, s_next = unpack(experience)
        states[i] = s
        
        local qValues = self.network:forward(s)
        targets[i] = qValues:clone()
        
        local nextQValues = targetNetwork:forward(s_next)
        local maxNextQ = torch.max(nextQValues)
        targets[i][a] = r + self.gamma * maxNextQ
    end

    -- Perform gradient descent
    self.network:zeroGradParameters()
    local output = self.network:forward(states)
    local loss = self.criterion:forward(output, targets)
    local gradOutput = self.criterion:backward(output, targets)
    self.network:backward(states, gradOutput)
    self.network:updateParameters(self.learningRate)

    -- Update the target network periodically
    if self.step % self.targetUpdateFreq == 0 then
        for i = 1, #self.network.modules do
            if self.network.modules[i].weight then
                targetNetwork.modules[i].weight:copy(self.network.modules[i].weight)
            end
            if self.network.modules[i].bias then
                targetNetwork.modules[i].bias:copy(self.network.modules[i].bias)
            end
        end
    end
    
    self.step = self.step + 1
end

DQN.step = 1

return DQN
