function [mse] = get_mse_som(som, som_in, som_target)
if size(som_in, 2) ~= size(som_target, 2)
    mse = NaN;
    return
end
results = sim(som, som_in);
out_size = size(results, 1);
out_values = [];
for r_i = 1:size(results, 2)
    output = results(:, r_i);
    out_values = [out_values (out_size - find(output)) / out_size];
    % Same as get_misclassification_som, inverted.
end
mse = immse(som_target, out_values);