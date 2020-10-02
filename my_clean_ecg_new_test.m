%params.type_detector = 'pan-tompkins';
params.type_detector = 'DWT';
params.type_thresh = 'percent';
%params.type_thresh = 'fixed';
params.dist_thresh = 0.8;
params.a_coeff  = 200;        
params.a_offset = -100;
params.do_plot = true;


%load('data\data_test.mat');


i = 4;
shape = [3, 5];
j_start = 31;
%shape = [1, 1];
%j_start = 12;


j = [j_start : (j_start + prod(shape) - 1)];


j_resh = reshape(j, shape(1), shape(2));

for j_resh_i = 1:size(j_resh, 1)
    for j_resh_j = 1:size(j_resh, 2)
        
        currpos = shape(2) * (j_resh_i-1) + j_resh_j;
        subplot(shape(1), shape(2), currpos);
        
        ecg_resampled = resample(data_train{i}{j(currpos)}, 5, 3); % resample from 300 to 500 Hz

        %ecg_clean_seg = my_clean_ecg_new(ecg_resampled,500, params); % clean ECG
        my_clean_ecg_new(ecg_resampled,500, params); % clean ECG
        
    end
end

% val_clean = reshape(ecg_clean_seg, numel(ecg_clean_seg),1);    % unify all segments again
% 
% data_train{i}{j} = val_clean;                                  % overwrite all signal
% num_det{i}(j) = size(ecg_clean_seg, 2);
% num_det_rel{i}(j) = num_det{i}(j) / numel(data_train_nonclean{i}{j});
