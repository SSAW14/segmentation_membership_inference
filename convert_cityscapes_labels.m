%%% Script for converting Mapillary ground truth labels into Cityscapes
%%% Creator: Yang He
%%% Contact: yang.he@cispa.saarland or yang@mpi-inf.mpg.de

clear
clc

% please change the path for mapillary dataset in the following
mapillary_path = '/path_to_mapillary';

% set cityscapes_labels as the output folder by default
output_folder = 'cityscapes_labels';

% start to convert
labels_mapillary = {[13, 41, 24], [15,2], [17], [6], [3], [45, 47], [48], [50], [30], [29], [27], [19], [20,21,22], [55], [61], [54], [58], [57], [52]};

labels = 255*ones(1,66);

for i = 1:19
    cls = labels_mapillary{i};
    for j = 1:length(cls)
        labels(cls(j)+1) = i-1;
    end
end

folders = {'training' , 'validation'};

for i = 1:length(folders)
    if ~exist(sprintf('%s/%s/%s', mapillary_path, folders{i}, output_folder))
        mkdir(sprintf('%s/%s/%s', mapillary_path, folders{i}, output_folder));
    end

    files = dir(sprintf('%s/%s/labels/*.png', mapillary_path, folders{i}))
    for j = 1:length(files)
        name_file = files(j).name;
        gt = imread(sprintf('%s/%s/labels/%s', mapillary_path, folders{i}, name_file));
        gt_out = 255*ones(size(gt));
        for k = 1:66
            gt_out(gt==k-1) = labels(k);
        end
        imwrite(uint8(gt_out),sprintf('%s/%s/%s/%s', mapillary_path, folders{i}, output_folder, name_file));
    end
end
