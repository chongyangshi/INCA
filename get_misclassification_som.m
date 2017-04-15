function [missclass_rate] = get_misclassification_som(som, som_in, som_target, mapstd_ts_reverse)
if size(som_in, 2) ~= size(som_target, 2)
    missclass_rate = NaN;
    return
end
misclass_count = 0;
total_count = size(som_in, 2);
out = sim(som, som_in);
if exist('mapstd_ts_reverse', 'var')
    out = mapstd('reverse', out, mapstd_ts_reverse);
end
for som_i = 1:size(som_in, 2)
    output = out(:,som_i);
    if find(output) < (size(output, 1) / 2)
        classification = 1;
    else
        classification = 0;
    end
    % Visual obersvation with all ones and all zeros inputs report that
    % the positioning appears to be reversed on one dimension.
    % Therefore the classification is reversed.
    if round(som_target(som_i)) ~= round(classification)
        misclass_count = misclass_count + 1;
    end
end
missclass_rate = misclass_count / total_count * 100;