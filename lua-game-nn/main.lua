-- Import necessary libraries
_G.love = require("love")
local DQN = require("dqn")
require 'torch'
require 'nn'
require 'optim'

-- Game state and initial settings
local cube = { 
    x = 100, 
    y = 400, 
    size = 20, 
    speed = 0, 
    gravity = 350, 
    jump = -250, 
    xSpeed = 0, 
    xAcceleration = 100, 
    friction = 10 
}

local camera = { 
    y = 0, 
    speed = 100 
}

local state = "play"
local score = 0
local gaps = {}
local obstacleFrequency = 300
local gapWidth = 100
local blockHeight = 10
local generation = 1
local maxGenerations = 10000

-- Generate initial obstacles
function generateObstacles()
    local startY = cube.y + 200
    for i = 1, 5 do
        local blockY = startY - i * obstacleFrequency
        local gapX = math.random(0, love.graphics.getWidth() - gapWidth)
        table.insert(gaps, { x = gapX, y = blockY, passed = false })
    end
end

-- Update obstacles dynamically
function updateObstacles()
    for i = #gaps, 1, -1 do
        if gaps[i].y > camera.y + love.graphics.getHeight() then
            table.remove(gaps, i)
        end
    end
    while #gaps < 5 do
        local lastY = gaps[#gaps] and gaps[#gaps].y or 0
        local newY = lastY - obstacleFrequency
        local newGapX = math.random(0, love.graphics.getWidth() - gapWidth)
        table.insert(gaps, { x = newGapX, y = newY, passed = false })
    end
end

-- Reset the game
function resetGame()
    cube.x, cube.y, cube.speed, score = 100, 400, 0, 0
    camera.y = 0
    gaps = {}
    generateObstacles()
end

function love.load()
    love.graphics.setFont(love.graphics.newFont(32))
    generateObstacles()
end

function love.update(dt)
    if state == "play" then
        local currentState = torch.Tensor({
            cube.x, 
            cube.y, 
            cube.speed, 
            gaps[1] and gaps[1].x - cube.x or 0,
            gaps[1] and gaps[1].y - cube.y or 0,
            gaps[2] and gaps[2].x - cube.x or 0,
            gaps[2] and gaps[2].y - cube.y or 0,
            gaps[3] and gaps[3].x - cube.x or 0,
            gaps[3] and gaps[3].y - cube.y or 0,
            camera.y
        })

        local action = DQN:selectAction(currentState)

        if action == 1 then
            cube.xSpeed = -cube.xAcceleration
            cube.speed = cube.jump
        elseif action == 2 then
            cube.xSpeed = cube.xAcceleration
            cube.speed = cube.jump
        end

        cube.speed = cube.speed + cube.gravity * dt
        cube.y = cube.y + cube.speed * dt
        cube.x = cube.x + cube.xSpeed * dt

        local targetY = cube.y - love.graphics.getHeight() / 2
        camera.y = camera.y + (targetY - camera.y) * camera.speed * dt

        local reward = 0
        for _, gap in ipairs(gaps) do
            if cube.y + cube.size > gap.y and cube.y < gap.y + blockHeight then
                if cube.x + cube.size < gap.x or cube.x > gap.x + gapWidth then
                    state = "gameover"
                    reward = -1
                elseif not gap.passed then
                    gap.passed = true
                    score = score + 1
                    reward = 1
                end
            end
        end

        if cube.y > camera.y + love.graphics.getHeight() then
            state = "gameover"
            reward = -1
        end

        local nextState = torch.Tensor({
            cube.x,
            cube.y,
            cube.speed,
            gaps[1] and gaps[1].x - cube.x or 0,
            gaps[1] and gaps[1].y - cube.y or 0,
            gaps[2] and gaps[2].x - cube.x or 0,
            gaps[2] and gaps[2].y - cube.y or 0,
            gaps[3] and gaps[3].x - cube.x or 0,
            gaps[3] and gaps[3].y - cube.y or 0,
            camera.y
        })

        DQN:train(currentState, action, reward, nextState)
        updateObstacles()
    end

    if state == "gameover" then
        if generation < maxGenerations then
            generation = generation + 1
            resetGame()
            state = "play"
        else
            love.event.quit()
        end
    end
end

function love.draw()
    love.graphics.translate(0, -camera.y)

    love.graphics.setColor(0, 1, 0)
    love.graphics.rectangle("fill", cube.x, cube.y, cube.size, cube.size)

    love.graphics.setColor(1, 0, 0)
    for _, gap in ipairs(gaps) do
        love.graphics.rectangle("fill", 0, gap.y, gap.x, blockHeight)
        love.graphics.rectangle("fill", gap.x + gapWidth, gap.y, love.graphics.getWidth() - gap.x - gapWidth, blockHeight)
    end

    love.graphics.setColor(1, 1, 1)
    love.graphics.print("Score: " .. score, 10, camera.y + 10)
    love.graphics.print("Generation: " .. generation, 10, camera.y + 50)
end

function love.keypressed(key)
    if state == "gameover" and key == "r" then
        resetGame()
        state = "play"
        generation = 1
    end
end