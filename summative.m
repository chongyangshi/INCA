datasets = {};
datafiles = {'datatraining.txt'; 'datatest.txt'; 'datatest2.txt'};

for i = 1:size(datafiles, 1)
    file = fopen(datafiles{i});
    data = textscan(file, '%s%s%f%f%f%f%f%f', 'Delimiter', ',', 'HeaderLines', 1);
    date_times = strrep(data{2}, '"', '');
    time_stamps = datenum(date_times, 'yyyy-mm-dd HH:MM:SS');
    values = [time_stamps];
    for j = 3:8
        values = [values data{j}];
    end
    datasets{end+1} = values;
    file = fclose(file);
end

