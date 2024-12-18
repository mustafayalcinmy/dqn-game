-- button.lua
local Button = {}
Button.__index = Button

function Button:new(y, width, height, text, color, textColor, onClick)
    -- Calculate horizontal position to center the button
    local x = (love.graphics.getWidth() - width) / 2
    return setmetatable({
        x = x,
        y = y,
        width = width,
        height = height,
        text = text,
        color = color,
        textColor = textColor,
        onClick = onClick or function() end -- Default to a no-op function
    }, Button)
end

function Button:draw()
    -- Draw the button
    love.graphics.setColor(self.color)
    love.graphics.rectangle("fill", self.x, self.y, self.width, self.height)

    -- Draw the button text, vertically and horizontally centered
    love.graphics.setColor(self.textColor)
    local font = love.graphics.getFont()
    local textHeight = font:getHeight(self.text)
    love.graphics.printf(self.text, self.x, self.y + (self.height - textHeight) / 2, self.width, "center")
end

function Button:isHovered(mx, my)
    -- Check if the mouse is over the button
    return mx > self.x and mx < self.x + self.width and
        my > self.y and my < self.y + self.height
end

function Button:click(mx, my)
    -- If hovered, execute the button's onClick function
    if self:isHovered(mx, my) then
        self.onClick()
    end
end

return Button
