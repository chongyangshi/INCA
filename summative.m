datasets = {};
datafiles = {'datatraining.txt'; 'datatest.txt'; 'datatest2.txt'};

for i = 1:size(datafiles, 1)
    file = fopen(datafiles{i});
    data = textscan(file, '%s%s%f%f%f%f%f%f', 'Delimiter', ',', 'HeaderLines', 1);
    date_times = strrep(data{2}, '"', '');
    time_stamps = datenum(date_times, 'yyyy-mm-dd HH:MM:SS');
    time_values = []; % No time
    %time_values = [time_stamps']; % Time Stamp
    %time_values = [cellfun(@str2num, strrep(data{1}, '"', ''))']; % Time Sequence
    if ~isempty(time_values)
        values = im2double(mat2gray(time_values));
    else
        values = time_values;
    end
    for j = 3:7
        values = [values; data{j}'];
    end
    results = [data{8}'];
    datasets{end+1} = {values, results};
    file = fclose(file);
end

training_set_in = datasets{1}{1};
training_set_out = datasets{1}{2};
testing_set_in = [datasets{2}{1} datasets{3}{1}];
testing_set_out = [datasets{2}{2} datasets{3}{2}];
%{
fprintf('============================================================\n');
fprintf('Pre-testing multilayer perceptron networks...\n');
trial_network_sizes = {[5 3 2], [6 4 2], [10 5 5], [20 10 10], [6 4]};
for n_size = 1:size(trial_network_sizes, 2)
    network_size = cell2mat(trial_network_sizes(n_size));
    
    test_lm = feedforwardnet(network_size, 'trainlm');
    [test_lm, test_lm_record] = train(test_lm, training_set_in, training_set_out);
    fprintf('%s LM Training MSE: %f.\n', mat2str(network_size), test_lm_record.best_perf);
    result = test_lm(testing_set_in);
    perf = perform(test_lm, testing_set_out, result);
    fprintf('%s LM Validation MSE: %f.\n', mat2str(network_size), perf);
    fprintf('%s LM Validation Actual Misclassification Rate: %f.\n', mat2str(network_size), get_misclassification(testing_set_out, result));
    
    test_gd = feedforwardnet([10 5 5], 'traingd');
    [test_gd, test_gd_record] = train(test_gd, training_set_in, training_set_out);
    fprintf('%s GD Training MSE: %f.\n', mat2str(network_size), test_gd_record.best_perf);
    result = test_gd(testing_set_in);
    perf = perform(test_gd, testing_set_out, result);
    fprintf('%s GD Validation MSE: %f.\n', mat2str(network_size), perf);
    fprintf('%s GD Validation Actual Misclassification Rate: %f.\n', mat2str(network_size), get_misclassification(testing_set_out, result));
    
    test_gdm = feedforwardnet([10 5 5], 'traingdm');
    [test_gdm, test_gdm_record] = train(test_gdm, training_set_in, training_set_out);
    fprintf('%s GDM Training MSE: %f.\n', mat2str(network_size), test_gdm_record.best_perf);
    result = test_gdm(testing_set_in);
    perf = perform(test_gdm, testing_set_out, result);
    fprintf('%s GDM Validation MSE: %f.\n', mat2str(network_size), perf);
    fprintf('%s GDM Validation Actual Misclassification Rate: %f.\n', mat2str(network_size), get_misclassification(testing_set_out, result));
end

fprintf('============================================================\n');
fprintf('Pre-testing RBF...\n');
rbf_goals = [0.1 0.05 0.01 0.001];
for rbf_g = 1:size(rbf_goals, 2)
    rbf_goal = rbf_goals(rbf_g);
    test_rb = newrb(training_set_in, training_set_out, rbf_goal, 1.0, size(training_set_in, 2), 25);
    fprintf('%f RBF Training MSE: %f.\n', rbf_goal, rbf_goal);
    rbf_out = test_rb(testing_set_in);
    result = immse(rbf_out, testing_set_out);
    fprintf('%f RBF Validation MSE: %f.\n', rbf_goal, result);
    fprintf('%f RBF Validation Actual Misclassification Rate: %f.\n', rbf_goal, get_misclassification(testing_set_out, rbf_out));
end

fprintf('============================================================\n');
fprintf('Pre-testing SOM...\n');
trial_som_sizes = {[2 1] [5 1] [10 1] [20 1] [30 1] [40 1] [50 1] [60 1] [70 1] [80 1] [90 1] [100 1]};
for som_s = 1:size(trial_som_sizes, 2)
    som_size = cell2mat(trial_som_sizes(som_s));
    test_som = selforgmap(som_size);
    test_som = train(test_som, training_set_in);
    fprintf('%s SOM Training MSE: %f.\n', mat2str(som_size), get_mse_som(test_som, training_set_in, training_set_out));
    fprintf('%s SOM Validation MSE: %f.\n', mat2str(som_size), get_mse_som(test_som, testing_set_in, testing_set_out));
    fprintf('%s SOM Validation Actual Misclassification Rate: %f.\n', mat2str(som_size), get_misclassification_som(test_som, testing_set_in, testing_set_out));
end
%}

proper_training_in = datasets{1}{1};
proper_training_out = datasets{1}{2};
door_closed_test_in = datasets{2}{1};
door_closed_test_out = datasets{2}{2};
door_open_test_in = datasets{3}{1};
door_open_test_out = datasets{3}{2};

% Time check -- uncoment and comment lines at start of script
som_size = [20 1];
test_som = selforgmap(som_size);
test_som = train(test_som, proper_training_in);
fprintf('%s SOM Training MSE: %f.\n', mat2str(som_size), get_mse_som(test_som, proper_training_in, proper_training_out));
fprintf('%s SOM Validation 1 MSE: %f.\n', mat2str(som_size), get_mse_som(test_som, door_closed_test_in, door_closed_test_out));
fprintf('%s SOM Validation 1 Actual Misclassification Rate: %f.\n', mat2str(som_size), get_misclassification_som(test_som, door_closed_test_in, door_closed_test_out));
fprintf('%s SOM Validation 2 MSE: %f.\n', mat2str(som_size), get_mse_som(test_som, door_open_test_in, door_open_test_out));
fprintf('%s SOM Validation 2 Actual Misclassification Rate: %f.\n', mat2str(som_size), get_misclassification_som(test_som, door_open_test_in, door_open_test_out));

