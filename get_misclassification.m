function [misclass_rate] = get_misclassification(results, targets)
if size(results, 2) ~= size(targets, 2)
    misclass_rate = NaN;
    return
end
total_count = 0;
misclassification_count = 0;
for i = 1:size(results, 2)
    total_count = total_count + 1;
    if round(results(i)) ~= round(targets(i))
        misclassification_count = misclassification_count + 1;
    end
end
misclass_rate = misclassification_count / total_count;