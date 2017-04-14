% Data processing file for INCA Spring Formative.
% Written by Chongyang Shi.

% Read file.
source_file = csvread('parkinsons_updrs.data', 1, 0);
picked_data = source_file;

% Variable classifications: -1 for data excluded from statistical
% analysis, 0 for numerical input data, 1 for discrete input data,
% 2 for result data.
classifications = [-1, 0, 1, 0, -1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]';

% Motor UPDRS is removed from data set as instructed. Subject # is also
% not considered at this stage, since while it is useful in distinguishing
% the affinity between data rows, training neural networks on individual
% patients only may produce a bias. Sex of patients might be significant,
% as Parkinsons disease have different progression patterns between the two
% genders. Therefore it is worth training with data from different gender
% separately, or maybe use -1 and 1 as effect coding?


% Fetch male and female candidates separately to be grouped.
subjects_seen = [];
males = {};
females = {};
current_person = [];
for i = 1:size(source_file, 1)
    if ~ismember(source_file(i, 1), subjects_seen)
        subjects_seen = [subjects_seen source_file(i, 1)];
        if ~isempty(current_person)
            if current_person(1, 3) == 0
                males{end+1} = current_person;
            else
                females{end+1} = current_person;
            end
        end   
        current_person = [source_file(i, :)];
    else
        current_person = [current_person; source_file(i, :)];
    end
end

% Group the candidates into groups of five.
counter = 0;
male_groups = {};
groups_needed = 4;
current_group = [];
for i = 1:size(males, 2)
    counter = counter + 1;
    current_group = [current_group; males{i}];
    if counter == 5
        male_groups{end+1} = current_group;
        current_group = [];
        counter = 0;
        if size(male_groups, 2) >= groups_needed
            break
        end
    end
end

counter = 0;
female_groups = {};
groups_needed = 2;
current_group = [];
for i = 1:size(females, 2)
    counter = counter + 1;
    current_group = [current_group; females{i}];
    if counter == 5
        female_groups{end+1} = current_group;
        current_group = [];
        counter = 0;
        if size(female_groups, 2) >= groups_needed
            break
        end
    end
end

% Loop through each group.
groups = [male_groups female_groups];
processed_groups = {};

for group = 1:size(groups, 2)
    
    fprintf('==================================================\n');
    if group <= 4
        group_gender = 'males';
    else
        group_gender = 'females';
    end
    fprintf('Preprocessing Group %d, which is a group of 5 %s.\n', group, group_gender);
    
    % Remove data not being considered.
    chosen_data = groups{group};
    num_removed = 0;
    for i = 1:size(classifications, 1)
        if classifications(i) == -1
            chosen_data(:,i-num_removed) = [];
            num_removed = num_removed + 1;
        end
    end

    % Create two sets of inputs: voice only, and all inputs.
    voice_only_data = [chosen_data(:, 5:size(chosen_data, 2)) chosen_data(:, 4)];
    all_variable_data = [chosen_data(:, 2) chosen_data(:, 1) chosen_data(:, 3) chosen_data(:, 5:size(chosen_data,2)) chosen_data(:, 4)];

    % New orderings:
    % Voice only: 
    %   1. Jitter(%), 2. Jitter(Abs), 3. Jitter:RAP, 4. Jitter:PPQ5, 5. Jitter:DDP
    %   6. Shimmer, 7. Shimmer(dB), 8. Shimmer:APQ3, 9. Shimmer:APQ5, 10. Shimmer:APQ11, 11. Shimmer:DDA
    %   12. NHR, 13. HNR
    %   14. RPDE
    %   15. DFA
    %   16. PPE
    %   17. total_UPDRS (result)
    % All variables:
    %   1. sex
    %   2. age
    %   3. test_time
    %   4. Jitter(%), 5. Jitter(Abs), 6. Jitter:RAP, 7. Jitter:PPQ5, 8. Jitter:DDP
    %   9. Shimmer, 10. Shimmer(dB), 11. Shimmer:APQ3, 12. Shimmer:APQ5, 13. Shimmer:APQ11, 14. Shimmer:DDA
    %   15. NHR, 16. HNR
    %   17. RPDE
    %   18. DFA
    %   19. PPE
    %   20. total_UPDRS (result)

    % Scale and preprocess input data (except the binary variable gender).
    % Voice only: 1:16
    for i = 12:16
        column = voice_only_data(:, i);
        column_scaled = (column - min(column)) / (max(column) - min(column));
        voice_only_data(:, i) = column_scaled;
    end

    % Perform PCA on voice data, separately for jitter and shimmer.
    [jitter_std, ~] = mapstd(voice_only_data(:,1:5));
    [jitter_pca, jitter_pca_config] = processpca(jitter_std',0.0001);
    [shimmer_std, ~] = mapstd(voice_only_data(:,6:11));
    [shimmer_pca, shimmer_pca_config] = processpca(shimmer_std', 0.0001);
    fprintf('Voice data PCA eliminated %d in jitter and %d in shimmer. \n', jitter_pca_config.xrows - jitter_pca_config.yrows, shimmer_pca_config.xrows - shimmer_pca_config.yrows);
    voice_only_data = [jitter_pca' shimmer_pca' voice_only_data(:, 12:end)];
    voice_input = voice_only_data(:,1:(end-1))';
    voice_target = voice_only_data(:,end)';

    % All data: 2:19
    for i = 2:3
        column = all_variable_data(:, i);
        column_scaled = (column - min(column)) / (max(column) - min(column));
        all_variable_data(:, i) = column_scaled;
    end
    
    for i = 15:19
        column = all_variable_data(:, i);
        column_scaled = (column - min(column)) / (max(column) - min(column));
        all_variable_data(:, i) = column_scaled;
    end

    % Perform PCA on all data, separately for jitter and shimmer.
    [jitter_std, ~] = mapstd(all_variable_data(:,4:8));
    [jitter_pca, jitter_pca_config] = processpca(jitter_std',0.0001);
    [shimmer_std, ~] = mapstd(all_variable_data(:,9:14));
    [shimmer_pca, shimmer_pca_config] = processpca(shimmer_std', 0.0001);
    fprintf('All data PCA eliminated %d in jitter and %d in shimmer. \n', jitter_pca_config.xrows - jitter_pca_config.yrows, shimmer_pca_config.xrows - shimmer_pca_config.yrows);
    all_variable_data = [all_variable_data(:, 1:3) jitter_pca' shimmer_pca' all_variable_data(:, 15:end)];
    all_input = all_variable_data(:,1:(end-1))';
    all_target = all_variable_data(:,end)';
    
    fprintf('==================================================\n');
    
    processed_groups{end+1} = {voice_input voice_target all_input all_target};
end

LM_Voice_MSEs = [];
LM_All_MSEs = [];
BR_Voice_MSEs = [];
BR_All_MSEs = [];
SCG_Voice_MSEs = [];
SCG_All_MSEs = [];
LM_Voice_MSEs_Training = [];
LM_All_MSEs_Training = [];
BR_Voice_MSEs_Training = [];
BR_All_MSEs_Training = [];
SCG_Voice_MSEs_Training = [];
SCG_All_MSEs_Training = [];

for n = 1:10
    
    fprintf('Beginning test run #%d.\n', n);
    
    for group = 1:size(processed_groups, 2)

        % Train with training data set.
        if mod(group, 2) ~= 0

            %LM Voice Only
            fprintf('Training LM for Group %d, voice data only.\n', group);
            current_lm_network_voice = feedforwardnet([20 10 10 10], 'trainlm');
            current_lm_network_voice.trainParam.max_fail = 50;
            current_lm_network_voice.trainParam.goal = 0.1;
            [current_lm_network_voice, current_lm_record] = train(current_lm_network_voice, processed_groups{group}{1}, processed_groups{group}{2});
            fprintf('MSE: %f.\n', current_lm_record.best_perf);
            LM_Voice_MSEs_Training = [LM_Voice_MSEs_Training current_lm_record.best_perf];
            
            %LM All Inputs
            fprintf('Training LM for Group %d, all input data.\n', group);
            current_lm_network_all = feedforwardnet([20 10 10 10], 'trainlm');
            current_lm_network_all.trainParam.max_fail = 50;
            current_lm_network_voice.trainParam.goal = 0.1;
            [current_lm_network_all, current_lm_record] = train(current_lm_network_all, processed_groups{group}{3}, processed_groups{group}{4});
            fprintf('MSE: %f.\n', current_lm_record.best_perf);
            LM_All_MSEs_Training = [LM_All_MSEs_Training current_lm_record.best_perf];
            
            %BR Voice Only
            fprintf('Training BR for Group %d, voice data only.\n', group);
            current_br_network_voice = feedforwardnet([20 10 10 10], 'trainbr');
            current_br_network_voice.trainParam.goal = 0.1;
            [current_br_network_voice, current_br_record] = train(current_br_network_voice, processed_groups{group}{1}, processed_groups{group}{2});
            fprintf('MSE: %f.\n', current_br_record.best_perf);
            BR_Voice_MSEs_Training = [BR_Voice_MSEs_Training current_br_record.best_perf];
            
            %BR All Inputs
            fprintf('Training BR for Group %d, all input data.\n', group);
            current_br_network_all = feedforwardnet([20 10 10 10], 'trainbr');
            current_br_network_all.trainParam.goal = 0.1;
            [current_br_network_all, current_br_record] = train(current_br_network_all, processed_groups{group}{3}, processed_groups{group}{4});
            fprintf('MSE: %f.\n', current_br_record.best_perf);
            BR_All_MSEs_Training = [BR_All_MSEs_Training current_br_record.best_perf];
            
            %SCG Voice Only
            fprintf('Training SCG for Group %d, voice data only.\n', group);
            current_scg_network_voice = feedforwardnet([20 10 10 10], 'trainscg');
            current_scg_network_voice.trainParam.goal = 0.1;
            [current_scg_network_voice, current_scg_record] = train(current_scg_network_voice, processed_groups{group}{1}, processed_groups{group}{2});
            fprintf('MSE: %f.\n', current_scg_record.best_perf);
            SCG_Voice_MSEs_Training = [SCG_Voice_MSEs_Training current_scg_record.best_perf];
            
            %SCG All Inputs
            fprintf('Training SCG for Group %d, all input data.\n', group);
            current_scg_network_all = feedforwardnet([20 10 10 10], 'trainscg');
            current_scg_network_all.trainParam.goal = 0.1;
            [current_scg_network_all, current_scg_record] = train(current_scg_network_all, processed_groups{group}{3}, processed_groups{group}{4});
            fprintf('MSE: %f.\n', current_scg_record.best_perf);
            SCG_All_MSEs_Training = [SCG_All_MSEs_Training current_scg_record.best_perf];
        % Evaluate with testing data set.
        else

            % LM Voice Only
            result = current_lm_network_voice(processed_groups{group}{1});
            perf = perform(current_lm_network_voice, processed_groups{group}{2}, result);
            fprintf('MSE of LM for Group %d (evaluation pair of Group %d), voice data only is: %f.\n', group, group - 1, perf);
            LM_Voice_MSEs = [LM_Voice_MSEs perf];

            % LM All Inputs
            result = current_lm_network_all(processed_groups{group}{3});
            perf = perform(current_lm_network_all, processed_groups{group}{4}, result);
            fprintf('MSE of LM for Group %d (evaluation pair of Group %d), all input data is: %f.\n', group, group - 1, perf);
            LM_All_MSEs = [LM_All_MSEs perf];

            % BR Voice Only
            result = current_br_network_voice(processed_groups{group}{1});
            perf = perform(current_br_network_voice, processed_groups{group}{2}, result);
            fprintf('MSE of BR for Group %d (evaluation pair of Group %d), voice data only is: %f.\n', group, group - 1, perf);
            BR_Voice_MSEs = [BR_Voice_MSEs perf];

            % BR All Inputs
            result = current_br_network_all(processed_groups{group}{3});
            perf = perform(current_br_network_all, processed_groups{group}{4}, result);
            fprintf('MSE of BR for Group %d (evaluation pair of Group %d), all input data is: %f.\n', group, group - 1, perf);
            BR_All_MSEs = [BR_All_MSEs perf];

            % SCG Voice Only
            result = current_scg_network_voice(processed_groups{group}{1});
            perf = perform(current_scg_network_voice, processed_groups{group}{2}, result);
            fprintf('MSE of SCG for Group %d (evaluation pair of Group %d), voice data only is: %f.\n', group, group - 1, perf);
            SCG_Voice_MSEs = [SCG_Voice_MSEs perf];

            % SCG All Inputs
            result = current_scg_network_all(processed_groups{group}{3});
            perf = perform(current_scg_network_all, processed_groups{group}{4}, result);
            fprintf('MSE of SCG for Group %d (evaluation pair of Group %d), all input data is: %f.\n', group, group - 1, perf);
            SCG_All_MSEs = [SCG_All_MSEs perf];

        end;
    end;
end;


fprintf('----------------------------------------\n');
fprintf('Training Set Results:\n');
titles = {'LM Voice Only'; 'LM All Inputs'; 'BR Voice Only'; 'BR All Inputs'; 'SCG Voice Only'; 'SCG All Inputs'};
all_data = [LM_Voice_MSEs_Training; LM_All_MSEs_Training; BR_Voice_MSEs_Training; BR_All_MSEs_Training; SCG_Voice_MSEs_Training; SCG_All_MSEs_Training];
for n = 1:size(all_data, 1)
    male_data = [];
    female_data = [];
    for i = 1:size(all_data, 2)
        if mod(i, 3) == 0
            female_data = [female_data all_data(n, i)];
        else
            male_data = [male_data all_data(n, i)];
        end
    end
    fprintf('%s: both genders mean %f, variance %f; males mean %f, variance %f; females mean %f, variance %f.\n', titles{n}, mean(all_data(n, :)), var(all_data(n, :)), mean(male_data), var(male_data), mean(female_data), var(female_data));
end;
fprintf('----------------------------------------\n');


fprintf('----------------------------------------\n');
fprintf('Comparison Set Results:\n');
titles = {'LM Voice Only'; 'LM All Inputs'; 'BR Voice Only'; 'BR All Inputs'; 'SCG Voice Only'; 'SCG All Inputs'};
all_data = [LM_Voice_MSEs; LM_All_MSEs; BR_Voice_MSEs; BR_All_MSEs; SCG_Voice_MSEs; SCG_All_MSEs];
for n = 1:size(all_data, 1)
    male_data = [];
    female_data = [];
    for i = 1:size(all_data, 2)
        if mod(i, 3) == 0
            female_data = [female_data all_data(n, i)];
        else
            male_data = [male_data all_data(n, i)];
        end
    end
    fprintf('%s: both genders mean %f, variance %f; males mean %f, variance %f; females mean %f, variance %f.\n', titles{n}, mean(all_data(n, :)), var(all_data(n, :)), mean(male_data), var(male_data), mean(female_data), var(female_data));
end;
fprintf('----------------------------------------\n');

format long g;
goals = [100 50 25 10 5 1 0.1 0.01];
voice_results = {};
all_results = {};
voice_neurons = {};
all_neurons = {};
for group = 1:size(processed_groups, 2)
    if mod(group, 2) ~= 0
        voice_rbfs = {};
        all_rbfs = {};
        voice_neurons{group} = [];
        all_neurons{group} = [];
        for j = 1:size(goals, 2)
            fprintf('RBF Training for group %d with a MSE goal of %f.\n', group, goals(j));
            vrb = newrb(processed_groups{group}{1}, processed_groups{group}{2}, goals(j), 1.0, 9999, 1000);
            voice_rbfs{j} = vrb;
            arb = newrb(processed_groups{group}{3}, processed_groups{group}{4}, goals(j), 1.0, 9999, 1000);
            all_rbfs{j} = arb;
            voice_neurons{group} = [voice_neurons{group}; size(vrb.IW{1}, 1)];
            all_neurons{group} = [all_neurons{group}; size(arb.IW{1}, 1)];
        end;
    else
        voice_results{group} = [];
        all_results{group} = [];
        fprintf('RBF Simulating for group %d.\n', group);
        for j = 1:size(goals, 2) 
            voice_simulate = sim(voice_rbfs{j}, processed_groups{group}{1});
            voice_mse = immse(processed_groups{group}{2}, voice_simulate);
            voice_results{group} = [voice_results{group}; voice_mse];
            all_simulate = sim(all_rbfs{j}, processed_groups{group}{3});
            all_mse = immse(processed_groups{group}{4}, all_simulate);
            all_results{group} = [all_results{group}; all_mse];
        end;
        
    end;
end;

colours = ['y'; 'm'; 'c'; 'r'; 'g'; 'b'];
voice_only_fig = figure('Name','RBF Network MSE on Voice Only Data');
voice_only_legends = [];
all_input_fig = figure('Name','RBF Network MSE on All Inputs Data');
all_input_legends = [];
for group = 1:size(processed_groups, 2)
    if mod(group, 2) == 0
        fprintf('MSEs for Group %d testing on Group %d with voice data only is:\n', group, group - 1);
        disp(voice_results{group}');
        fprintf('Neurons:\n');
        disp(voice_neurons{group - 1}');
        fprintf('MSEs for Group %d testing on Group %d with all input data is:\n', group, group - 1);
        disp(all_results{group}');
        fprintf('Neurons:\n');
        disp(all_neurons{group - 1}');
        figure(voice_only_fig);
        title('RBF Network MSE on Voice Only Data');
        xlabel('Number of neurons in the RBF network');
        ylabel('MSE on unseen data');
        plot(voice_neurons{group - 1}', voice_results{group}' , ['-' colours(group) 'x']); hold on;
        axis([0,800,9,50000]);
        voice_only_legends = [voice_only_legends; 'Group' num2str(group)];
        figure(all_input_fig);
        title('RBF Network MSE on All Inputs Data');
        xlabel('Number of neurons in the RBF network');
        ylabel('MSE on unseen data');
        plot(all_neurons{group - 1}', all_results{group}' , ['-' colours(group) 'x']); hold on;
        axis([0,800,0,4000]);
        all_input_legends = [all_input_legends; 'Group' num2str(group)];
    end;
end;


figure(voice_only_fig);
legend(voice_only_legends); hold on;
figure(all_input_fig);
legend(all_input_legends);
