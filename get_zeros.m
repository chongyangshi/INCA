function [ones] = get_ones(input_set, goal_set)
if size(input_set, 2) ~= size(goal_set, 2)
    ones = NaN;
    return
end
ones = [];
for i = 1:size(input_set, 2)
    if goal_set(i) == 0
        ones = [ones input_set(:, i)];
    end
end
return