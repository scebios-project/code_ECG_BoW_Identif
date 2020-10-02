clear all
close all

% Add local paths
addpath(genpath('encoding'));
addpath(genpath('pwmetric'));
addpath('LIBSVM\windows')
addpath(genpath('C:\Program Files\Matlab\R2018b\toolbox\WaveLab850')); % for dwt
addpath(genpath('LCKSVD\OMPbox'));
addpath(genpath('LCKSVD\ksvdbox'));
addpath(genpath('LCKSVD'));
addpath(genpath('spams-matlab-v2.6\build'));
addpath(genpath('VLFeat\toolbox\mex'));

%======================
% Set parameters
%======================
basename  = 'CYBHI';

n_recs = 1;             % How many recordings to read from dataset (1 to 3)
segment_len = 5;       % seconds
basename = sprintf('%s_%drec_%dsec', basename, n_recs, segment_len);

mkdir('data')
mkdir(['data/' basename])
mkdir(['data/' basename '/plot'])
params.Data_DIR     = 'data\immutable\';
params.Codebook_DIR = ['data\' basename '\'];
params.Feature_DIR  = ['data\' basename '\'];
params.Results_DIR  = ['data\' basename '\'];
params.Plot_DIR     = ['data\' basename '\plot\'];

% Prepare logfile for this run
starttime = datetime('now');
logfile = sprintf('%slog_%s.txt', params.Results_DIR, starttime);
logfile = strrep(logfile, ' ', '_');
logfile = strrep(logfile, ':', '_');
diary(logfile);
diary on;

fprintf('%s --- Script started\n',datetime('now'));

params.numClass=4;          % Number of classes
%
params.test_set_class_size  = 100;   % how many recordings to use for test, per class
params.train_set_class_size = 179;   % how many recordings to use for train, per class
%
params.sub_length  = 512;            % window length
params.inter_point = 32;             % window step
params.size_feat = 128;               % how many feature coeffs to keep
params.level = 7;                     % coarsest level of DWT 
%
%params.transfType='randproj';
%params.transfType='dct';
params.transfType='dwt';
%
params.codebookSize = 500;  
%params.codebookSize=400;
%params.codebookSize=40;
%
params.VQtype='sparse';     
params.dl_lambda = 0.5;
params.dl_iter   = 100;
%params.VQtype='LC-sparse';
%params.VQtype='sparseMultiD';
%params.VQtype='LC-KSVD';
%
params.encType = 'SPC';
params.poolType = 'sum-pooling';
%params.poolType = 'No';
%params.poolType = 'sum-pooling-size';
%params.poolType = 'max-pooling';
params.poolSize = 400;
params.poolStride = 200;
%params.normType = 'No';
params.normType = 'L2';                 % 'L2'; 'L1'; 'Power'; 'Power+L2'; 'Power+L1';
%params.normType = 'mean';                 % 'L2'; 'L1'; 'Power'; 'Power+L2'; 'Power+L1';
%
params.classDistType = [1 0 1];
%params.classDistType = [1 1 1];
%params.classType = 'NN';
params.classType = 'SVM';
%params.classType = 'LC-KSVD';
%params.sub_length = 128;            % window length
%params.inter_point = 4;             % window step
%


params.codebook_subsample_set_factor = 1;  % should we use less feature vectors for codebook training
%
params.global_iterations = 10;      % run everything this many times, for averaging the results
% Cleaning:
params.train_classes_to_clean = [];    % which classes to apply cleaning on
params.test_classes_to_clean  = [];    % which classes to apply cleaning on
%params.train_classes_to_clean = [1, 2, 3, 4];    % which classes to apply cleaning on
%params.test_classes_to_clean = [1, 2, 3, 4];    % which classes to apply cleaning on
%params.clean_type_detector = {'pan-tompkins', 'DWT', 'DWT', 'DWT'};
%params.clean_type_thresh = {'fixed', 'percent', 'percent', 'percent'};
%params.clean_dist_thresh = [0.5, 0.9, 0.9, 0.9];
%params.clean_type_detector = {'pan-tompkins', 'pan-tompkins', 'pan-tompkins', 'pan-tompkins'};
params.clean_type_detector = {'DWT', 'DWT', 'DWT', 'DWT'};
%params.clean_type_detector = {'pan-tompkins', 'pan-tompkins', 'pan-tompkins', 'pan-tompkins'};
params.clean_type_thresh = {'percent', 'percent', 'percent', 'percent'};
%params.clean_type_thresh = {'fixed', 'fixed', 'fixed', 'fixed'};
%params.clean_dist_thresh = [1, 1, 1, 1];
params.clean_dist_thresh = 1 * [1, 1, 1, 1];
params.clean_a_coeff  = 200;        
params.clean_a_offset = -100;
params.num_std_limit = 1;
%params.clean_function = @my_clean_ecg;
params.clean_function = @my_clean_ecg_new;
%
%params.tran_remove_mean = true;     % remove mean value from feature vectors, after feature extraction ?
params.tran_remove_mean = false;     % remove mean value from feature vectors, after feature extraction ?
%
params.plot = 0;

fprintf('%s --- Current parameters\n',datetime('now'));
disp(params);

% Initialize random number generator and save its state
rngstate_filename = [params.Feature_DIR, '0_rngstate.mat'];
if ~exist(rngstate_filename)
    rng('shuffle');
    rngstate = rng;
    fprintf('%s --- Randomly initialized RNG state, current state (JSON encoded):\n',datetime('now'));
    fprintf('%s\n',jsonencode(rngstate));
    fprintf('%s --- Saving RNG state to %s\n',datetime('now'), rngstate_filename);
    save(rngstate_filename, 'rngstate');
    changed_upstream = true;
else
    load(rngstate_filename);
    rng(rngstate);
    fprintf('%s --- Loaded RNG state from %s:\n',datetime('now'), rngstate_filename);
    fprintf('%s\n',jsonencode(rngstate));
end

if strcmp(params.transfType,'randproj')
    global Phi
    %Phi = randn(32,128);
    Phi = randn(params.size_feat, params.sub_length);
    %load([params.Data_DIR,'Phi.mat']);
end

% Set this to true in order to force rerun of everything from this point onwards
% (disable caching)
%changed_upstream = true;
changed_upstream = false;

%hit_rate_allexp = [];
conf_matrix_allexp = {};
F1_allexp = {};
for p = 1:params.global_iterations
    params.permIdx = p;
    fprintf('                      ----------\n');
    fprintf('%s Iteration %d / %d\n',datetime('now'), p, params.global_iterations);

    %================================================
    % Read data files and save data in matrix form
    %================================================
    data_save_filename = [params.Feature_DIR, '1_data_', num2str(p), '.mat'];
    if changed_upstream || ~exist(data_save_filename)
        fprintf('%s --- Reading raw data\n',datetime('now'));

        n_persons = 65;
        %n_recs = 3;
        data = {};
        keptpers = 1;
        for ipers = 1:n_persons
            try      
                for irec = 1:n_recs                
                    d = readtable(sprintf('%s/CYBHI/hand/%d_%d.txt', params.Data_DIR, ipers, irec));
                    dtemp{irec} = d{:,4};
                    if numel(dtemp{irec}) < 1000
                        error('Data too short');
                    end
                end
                for irec = 1:n_recs                
                    data{keptpers, irec} = dtemp{irec};                
                end
                
                keptpers = keptpers + 1;
            catch
                fprintf('Skipping person %d\n', ipers);
            end
        end
        n_persons = size(data,1);

        %======================================
        % Split test set and train set
        %======================================
        %segment_len = 10;   % seconds
        overlap_len = 0; % seconds
        Fs = 1000;         % Hz
        for ipers = 1:n_persons
            datamat = buffer(data{ipers,1}, segment_len * Fs, overlap_len * Fs);
            datamat(:,1) = [];    % remove incomplete first and last segments
            datamat(:,end) = [];
            data_segments{ipers} = mat2cell(datamat, size(datamat,1), ones(1, size(datamat,2)));
                        
            rp = randperm(size(data_segments{ipers}, 2));
            limit = round(0.8 * size(data_segments{ipers}, 2));
            data_train{ipers} = data_segments{ipers}(rp(1:limit));
            data_test{ipers}  = data_segments{ipers}(rp(limit+1:end));

        end
%         for ipers = 1:n_persons
%             datamat = buffer(data{ipers,2}, segment_len * Fs, overlap_len * Fs);
%             datamat(:,1) = [];    % remove incomplete first and last segments
%             datamat(:,end) = [];           
%             data_test{ipers} = mat2cell(datamat, size(datamat,1), ones(1, size(datamat,2)));
%         end

        
        fprintf('%s --- Saving raw data in %s\n',datetime('now'), data_save_filename);
        save(data_save_filename,'data', 'data_test', 'data_train', '-v7.3');
        changed_upstream = true;  % recompute everything downstream
    else
        fprintf('%s --- Loading existing raw data from %s\n',datetime('now'), data_save_filename);
        load(data_save_filename,'data_test', 'data_train');
    end

    [conf_matrix, F1]  = run_BoW(data_train, data_test, params, changed_upstream);
    %hit_rate_allexp = [hit_rate_allexp; hit_rate_all];
    conf_matrix_allexp = [conf_matrix_allexp; conf_matrix];
    F1_allexp = [F1_allexp; F1];
end

% Compute average F1 score over all runs
F1train_mean = mean(cell2mat(cellfun(@(x) x(1,:) , F1_allexp, 'UniformOutput', 0)));
F1test_mean  = mean(cell2mat(cellfun(@(x) x(2,:) , F1_allexp, 'UniformOutput', 0)));
fprintf('%s --- Average F1 scores for training set\n', datetime('now'));
disp(F1train_mean)
fprintf('%s --- Average F1 scores for test set\n', datetime('now'));
disp(F1test_mean)
    
results_filename = sprintf('%sresults_%s.mat', params.Results_DIR, starttime);
results_filename = strrep(results_filename, ' ', '_');
results_filename = strrep(results_filename, ':', '_');

fprintf('%s --- Saving classification results in %s\n',datetime('now'), results_filename);
save(results_filename, 'conf_matrix_allexp', 'F1_allexp', 'params', 'rngstate');

fprintf('%s - Finished\n',datetime('now'));
diary off