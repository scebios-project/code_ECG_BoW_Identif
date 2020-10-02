function [conf_matrix, F1] = run_BoW(data_train, data_test, params, changed_upstream)

% ========================================================
% Setup parameters
% ========================================================
% Set paths
DATA_DIR     = params.Data_DIR;
CODEBOOK_DIR = params.Codebook_DIR;
FEATURE_DIR  = params.Feature_DIR;

% Dataset parameters
num_class   = params.numClass;              % number of classes
transf_type = params.transfType;            % 'dwt'; 'randproj'; 'dct';
sub_length  = params.sub_length;            % window length
inter_point = params.inter_point;           % window step
size_feat   = params.size_feat;             % how many feature coeffs to keep
level       = params.level;                 % dwt level

% Codebook parameters
VQ.Codebook_Size  = params.codebookSize;
VQ.Max_Iterations = 20;                     % max iterations for k-means
VQ.Verbosity      = 0;
VQ.type           = params.VQtype;          % 'kmeans'; 'sparse'

% Encoding parameters
encType  = params.encType;                  % 'SPC';
poolType = params.poolType;                 % 'max-pooling';
normType = params.normType;                 % 'L2';

% Classifier parameters
classDistType = params.classDistType;       % [chi2 L2 hist_intersect]; 
classType     = params.classType;           % 'NN';

% Current iteration (experiment)
p = params.permIdx;

% Cleaning function
clean_function = params.clean_function;

% ========================================================
% Perform feature extraction
% ========================================================
feature_filename = [params.Feature_DIR, '2_feat_', num2str(p), '_', transf_type, '_',num2str(sub_length),'_',num2str(inter_point),'_','data_train_tran.mat'];
if changed_upstream || ~exist(feature_filename)

    % ========================================================
    % Clean train set
    % ========================================================
    fprintf('%s --- Cleaning train set \n',datetime('now'));
    %for i = 1:size(data_train, 2)
    %for i = 1:1  % Only for the Normal class
    data_train_nonclean = data_train;  % save a copy before cleaning
    num_det = {};
    num_det_rel = {};

    hpf = designfilt('bandpassfir', ...       % Response type
       'CutoffFrequency1',2, ...     % Frequency constraints
       'CutoffFrequency2',10, ...     % Frequency constraints
       'FilterOrder', 100, ...
       'SampleRate',300);               % Sample rate
   
    
        for ii = 1:numel(params.train_classes_to_clean)  % Only for the specified class
        i = params.train_classes_to_clean(ii);
        
        % Prepare parameters for cleaning
        clean_params.type_detector = params.clean_type_detector{i};    
        clean_params.type_thresh = params.clean_type_thresh{i};
        clean_params.dist_thresh = params.clean_dist_thresh(i);        
        clean_params.a_coeff  = params.clean_a_coeff;        
        clean_params.a_offset = params.clean_a_offset;    
        clean_params.do_plot = false;
        clean_params.num_std_limit = params.num_std_limit;
        clean_params.filter = hpf;
        
        for j = 1:size(data_train{i}, 2)
            
            % Highpass first
            %data_train{i}{j} = highpass(data_train{i}{j}, 5, 300);
            data_train{i}{j} = filter(hpf, data_train{i}{j});
            
            ecg_resampled = resample(data_train{i}{j},5,3);                % resample from 300 to 500 Hz
            ecg_clean_seg = clean_function(ecg_resampled,500, clean_params); % clean ECG
            val_clean = reshape(ecg_clean_seg, numel(ecg_clean_seg),1);    % unify all segments again
            data_train{i}{j} = val_clean;                                  % overwrite all signal
            num_det{i}(j) = size(ecg_clean_seg, 2);
            num_det_rel{i}(j) = num_det{i}(j) / numel(data_train_nonclean{i}{j});
        end
        
        % Some signals may have eliminated competely, filter all-zero signals
        % Some signals may remain too short, filter them
        %data_train{i}(cellfun(@isempty, data_train{i})) = [];
        fun = @(x) numel(x) < params.sub_length;
        data_train{i}(cellfun(fun, data_train{i})) = [];
    end

    % ========================================================
    % Limit train set to small size
    % ========================================================
    % Get minimum size of class
%     for i = 1:numel(data_train)
%         trainClassSize(i) = size(data_train{i}, 2);
%     end
%     minClassSize = min(trainClassSize);
    
    % If a class has smaller elements than in param, use the smaller size
%     train_set_class_size = min(minClassSize, params.train_set_class_size);
%         
%     % Restrict
%     fprintf('%s --- Restrict train set to %d per class \n',datetime('now'), train_set_class_size);
%     for i = 1:numel(data_train)
%         data_train{i} = data_train{i}(1:train_set_class_size);
%     end

    % ========================================================
    % Clean test set
    % ========================================================
    fprintf('%s --- Cleaning test set \n',datetime('now'));
    %for i = 1:size(data_test, 2)
    %for i = 1:1  % Only for the Normal class
    data_test_nonclean = data_test;  % save a copy before cleaning
    for ii = 1:numel(params.test_classes_to_clean)  % Only for the specified class
        i = params.test_classes_to_clean(ii);

        % Prepare parameters for cleaning
        clean_params.type_detector = params.clean_type_detector{i};    
        clean_params.type_thresh = params.clean_type_thresh{i};
        clean_params.dist_thresh = params.clean_dist_thresh(i);        
        clean_params.a_coeff  = params.clean_a_coeff;        
        clean_params.a_offset = params.clean_a_offset;      
        clean_params.do_plot = false;
        clean_params.num_std_limit = params.num_std_limit;
        
        for j = 1:size(data_test{i}, 2)
            
            %Highpass first
            %data_test{i}{j} = highpass(data_test{i}{j}, 5, 300);
            data_test{i}{j} = filter(hpf, data_test{i}{j});
            
            ecg_resampled = resample(data_test{i}{j},5,3);                 % resample from 300 to 500 Hz
            ecg_clean_seg = clean_function(ecg_resampled,500, clean_params); % clean ECG
            val_clean = reshape(ecg_clean_seg, numel(ecg_clean_seg),1);    % unify all segments again

            % Keep segment only if larger than one window
            if val_clean >= params.sub_length
                data_test{i}{j} = val_clean;                             % overwrite signal
            end
            
        end
    end


    % ========================================================
    % Feature extraction
    % ========================================================
    fprintf('%s --- Feature extraction\n',datetime('now'))
    % For train set
    for i = 1:size(data_train,2)
        for j = 1:size(data_train{i},2)
            % sliding a subwindows along a time series and extract a feature vector for each sub sequences by transformation (e.g., 'wavelet')
             data_train_tran{i}{j} = series_transform(data_train{i}{j}',sub_length,inter_point,transf_type, size_feat, level);
             
             % Nic: 2020-08-20: Remove NaN
             remove_indices = logical(sum(isnan(data_train_tran{i}{j})));
             data_train_tran{i}{j}(:, remove_indices) = [];
             
             % NiC: Remove mean value from features
             if params.tran_remove_mean
                 for k = 1:size(data_train_tran{i}{j}, 2)
                     data_train_tran{i}{j}(:,k) = data_train_tran{i}{j}(:,k) - mean(data_train_tran{i}{j}(:,k));
                 end
             end
        end
    end
    % For test set
    for i = 1:size(data_test,2)
        for j = 1:size(data_test{i},2)
            % sliding a subwindows along a time series and extract a feature vector for each sub sequences by transformation (e.g., 'wavelet')
             data_test_tran{i}{j} = series_transform(data_test{i}{j}',sub_length,inter_point,transf_type, size_feat, level);
             
             % Nic: 2020-08-20: Remove NaN
             remove_indices = logical(sum(isnan(data_test_tran{i}{j})));
             data_test_tran{i}{j}(:, remove_indices) = [];             
             
             % NiC: Remove mean value from features
             if params.tran_remove_mean
                 for k = 1:size(data_test_tran{i}{j}, 2)
                     data_test_tran{i}{j}(:,k) = data_test_tran{i}{j}(:,k) - mean(data_test_tran{i}{j}(:,k));
                 end
             end             
        end
    end
    
    % Plot sample
    if params.plot
        plot_train_sigs = [1, 10, 100, 150];
        plot_test_sigs = [1, 25, 50, 75];
        plot_maxlen = 1000;
        plot_windows = [1, 2, 3, 4, 5, 6, 7, 8];
        close all
        for i = 1:size(data_train,2)
            for j = 1:numel(plot_train_sigs)
                f = figure('visible','off');
                plot(data_train{i}{plot_train_sigs(j)}(1:plot_maxlen));
                filename = sprintf('%s2_plot_train_%d_%d', params.Plot_DIR, i,plot_train_sigs(j));
                saveas(f,filename,'png');
                for k = 1:numel(plot_windows)
                    f = figure('visible','off');
                    plot(data_train_tran{i}{plot_train_sigs(j)}(:, plot_windows(k)));
                    filename = sprintf('%s2_plot_train_tran_%d_%d_%d', params.Plot_DIR, i,plot_train_sigs(j), plot_windows(k));
                    saveas(f,filename,'png');
                end
            end
        end
    end
    
    % Remove empty features (e.g. shorter than 1 window length)
    for i = 1:4
        data_train_tran{i}(cellfun(@isempty, data_train_tran{i})) = [];
    end    
    for i = 1:4
        data_test_tran{i}(cellfun(@isempty, data_test_tran{i})) = [];
    end     
    
    du_mkdir(params.Feature_DIR);
    fprintf('%s --- Saving features to %s\n',datetime('now'), feature_filename);
    save(feature_filename, 'data_train', 'data_train_nonclean', 'data_test', 'data_test_nonclean',...
        'data_train_tran', 'data_test_tran', 'num_det', 'num_det_rel', '-v7.3');
    changed_upstream = true;
else
    fprintf('%s --- Loading existing features from %s\n',datetime('now'), feature_filename);
    load(feature_filename);
end
    
% ==========================================================
% Generate codebook
% ==========================================================
codebook_filename = [params.Codebook_DIR, '4_codebook_', num2str(p), '_', transf_type, '_', VQ.type, '_', num2str(VQ.Codebook_Size), '.mat'];
if changed_upstream || ~exist(codebook_filename)
    
    % ==========================================================
    % Generate data for codebook training
    % ==========================================================
    data_mat_filename = [params.Feature_DIR, '3_' ,transf_type, '_', num2str(p), '_', 'data_tran_mat.mat'];
    if changed_upstream || ~exist(data_mat_filename)
        fprintf('%s --- Preparing data for codebook train and test \n',datetime('now'));
        data_train_tran_codebook_mat = [];      % matrix with all training data to be used in dict learning
        cat_train_codebook = [];                % stores the class index of all data in the coresponding matrix

        % Train data
        for i = 1:size(data_train_tran,2) % for each class
            % For codebook training
            newdata = cell2mat(data_train_tran{i});
            data_train_tran_codebook_mat = [data_train_tran_codebook_mat, newdata];
            cat_train_codebook = [cat_train_codebook, i * ones(1, size(newdata, 2))];
        end

        % Save mat data in a single file
        du_mkdir(params.Feature_DIR);
        fprintf('%s --- Saving features in matrix form in %s \n',datetime('now'), data_mat_filename);
        save(data_mat_filename,'data_train_tran_codebook_mat', 'cat_train_codebook','-v7.3');
        changed_upstream = true;
    else

        fprintf('%s --- Loading existing features matrix for codebook training from %s \n',datetime('now'), data_mat_filename);
        load(data_mat_filename);
    end

    fprintf('%s --- Generating codebook, type %s  \n',datetime('now'), VQ.type);
    if strcmp(VQ.type,'kmeans')
        codebook_size = VQ.Codebook_Size;
        cluster_options.maxiters = VQ.Max_Iterations;
        cluster_options.verbose  = VQ.Verbosity;

        data_train_codebook_mat = data_train_codebook_mat(:,1:20:size(data_train_codebook_mat,2));
        [centers]=vl_kmeans(double(data_train_codebook_mat), codebook_size, 'distance', 'L2', 'initialization','randsel','NumRepetitions',20,'algorithm', 'ANN');

        du_mkdir(params.Codebook_DIR);
        fname = [params.Codebook_DIR , transf_type, '_', VQ.type, '_', num2str(p), '_' , num2str(codebook_size) , '.mat']; 
        save(fname,'centers');

    elseif strcmp(VQ.type,'kmedians')
        codebook_size = VQ.Codebook_Size;
        cluster_options.maxiters = VQ.Max_Iterations;
        cluster_options.verbose  = VQ.Verbosity;

        data_train_codebook_mat = data_train_codebook_mat(:,1:20:size(data_train_codebook_mat,2));
        [centers]=vl_kmeans(double(data_train_codebook_mat), codebook_size, 'distance', 'L1', 'initialization','randsel','NumRepetitions',20,'algorithm', 'ANN');

        du_mkdir(params.Codebook_DIR);
        fname = [params.Codebook_DIR , transf_type, '_', VQ.type, '_', num2str(p), '_', num2str(codebook_size) , '.mat']; 
        save(fname,'centers');

    elseif strcmp(VQ.type,'sparse')
        codebook_size = VQ.Codebook_Size;
        subsample_set_factor = params.codebook_subsample_set_factor;
        data_train_tran_codebook_mat = data_train_tran_codebook_mat(:,1:subsample_set_factor:size(data_train_tran_codebook_mat,2));
        dlparam.mode = 1;
        dlparam.K=codebook_size;
        dlparam.lambda=params.dl_lambda;
        %param.iter=100;
        %param.lambda=0.9;
        %param.iter=500;
        dlparam.iter=params.dl_iter;
        dlparam.verbose = true;
        % 2020-08-20: commented out
%         dlparam.gamma1=0.5;
%         dlparam.modeD = 3;
%         dlparam.posD = true;
        
        fprintf('%s --- --- Using %d vectors for dictionary learning\n',datetime('now'), size(data_train_tran_codebook_mat,2));
        fprintf('%s --- --- Dict learning parameters:\n',datetime('now'));
        disp(dlparam);

        % Train dictionary
        [centers]=mexTrainDL(data_train_tran_codebook_mat,dlparam);
        %tic; [centers]=mexTrainDL(DT,param); toc;
        
        %[centers, model]=mexTrainDL(data_train_tran_codebook_mat,param);
%         alpha=mexLasso(data_train_tran_codebook_mat,centers,param);
%         R=mean(0.5*sum((data_train_tran_codebook_mat-centers*alpha).^2)+param.lambda*sum(abs(alpha)));
%         disp(R)
%         for rep = 1:10
%             param.D = centers;
%             [centers, model] = mexTrainDL(data_train_tran_codebook_mat,param,model);
%             alpha=mexLasso(data_train_tran_codebook_mat,centers,param);
%             R=mean(0.5*sum((data_train_tran_codebook_mat-centers*alpha).^2)+param.lambda*sum(abs(alpha)));
%             disp(R)
%         end

        % Save dictionary
        du_mkdir(params.Codebook_DIR);
        save(codebook_filename,'centers');        
        
    elseif strcmp(VQ.type,'LC-sparse')
        codebook_size = VQ.Codebook_Size;
        subsample_set_factor = params.codebook_subsample_set_factor;
        data_train_tran_codebook_mat = data_train_tran_codebook_mat(:,1:subsample_set_factor:size(data_train_tran_codebook_mat,2));
        param.mode = 2;
        param.K=codebook_size;
        param.lambda=0.15;
        %param.iter=100;
        %param.lambda=0.9;
        param.iter=500;
        param.verbose = true;
        
        fprintf('%s --- --- Using %d vectors for dictionary learning\n',datetime('now'), size(data_train_tran_codebook_mat,2));
        fprintf('%s --- --- Dict learning parameters:\n',datetime('now'));
        disp(param);

        % LC
        Y = data_train_tran_codebook_mat;
        Q = zeros(codebook_size, size(data_train_tran_codebook_mat, 2));
        chunk = codebook_size / (1 + size(data_train_tran,2));
        for i = 1:size(data_train_tran,2) % for each class
            rowsidx = chunk*(i-1)+1 : i*chunk;
            colsidx = (cat_train_codebook == i);
            Q(rowsidx, colsidx) = 1;
        end
        rowsidx = (size(Q, 1) - chunk + 1) : size(Q, 1); % last chunk
        Q(rowsidx, :) = 1;
        beta = 0.1;
        YT = [Y; sqrt(beta)*Q];
        
        
        % Train dictionary
        %[centers]=mexTrainDL(data_train_tran_codebook_mat,param);
        tic; [centers]=mexTrainDL(YT,param); toc;
        
        centers = centers(1:size(Y,1), :);  % get rid of A matrix below D
        
        %[centers, model]=mexTrainDL(data_train_tran_codebook_mat,param);
%         alpha=mexLasso(data_train_tran_codebook_mat,centers,param);
%         R=mean(0.5*sum((data_train_tran_codebook_mat-centers*alpha).^2)+param.lambda*sum(abs(alpha)));
%         disp(R)
%         for rep = 1:10
%             param.D = centers;
%             [centers, model] = mexTrainDL(data_train_tran_codebook_mat,param,model);
%             alpha=mexLasso(data_train_tran_codebook_mat,centers,param);
%             R=mean(0.5*sum((data_train_tran_codebook_mat-centers*alpha).^2)+param.lambda*sum(abs(alpha)));
%             disp(R)
%         end

        % Save dictionary
        du_mkdir(params.Codebook_DIR);
        save(codebook_filename,'centers');
        
    elseif strcmp(VQ.type,'sparseMultiD')
        codebook_size = VQ.Codebook_Size / 4;
        subsample_set_factor = params.codebook_subsample_set_factor;
        data_train_tran_codebook_mat = data_train_tran_codebook_mat(:,1:subsample_set_factor:size(data_train_tran_codebook_mat,2));
        %
        param.mode = 2;
        %param.lambda=0.15;
        param.lambda=0.90;
        %param.mode = 3;
        %param.lambda=2;
        %
        param.K=codebook_size;
        param.iter=1000;

        fprintf('%s --- --- Dict learning parameters:\n',datetime('now'));
        disp(param);
                
        % Train a dictionary for each class
        for i = 1:size(data_train_tran,2) % for each class
            traindata_for_class = data_train_tran_codebook_mat(:, cat_train_codebook == i);
            % clean NaN and other indesirables
            remove_indices = logical(sum(isnan(traindata_for_class)));
            traindata_for_class(:, remove_indices) = [];
            fprintf('%s --- --- Using %d vectors for dictionary learning for class %d\n',datetime('now'), size(traindata_for_class,2), i);
            centers{i}=mexTrainDL(traindata_for_class, param);
        end
        centers = cell2mat(centers);

        % Save dictionary
        du_mkdir(params.Codebook_DIR);
        save(codebook_filename,'centers');        
        
    elseif strcmp(VQ.type,'LC-KSVD')
        %codebook_size = VQ.Codebook_Size;
        %all_descriptors_train = all_descriptors_train(:,1:20:size(all_descriptors_train,2));
        %param.mode = 2;
        %param.K=codebook_size;
        %param.lambda=0.15;
        %param.iter=100;

        %[centers]=mexTrainDL(all_descriptors_train,param);

        %du_mkdir(params.Codebook_DIR);
        %fname = [params.Codebook_DIR , transf_type, '_', VQ.type, '_', num2str(p), '_', num2str(VQ.Codebook_Size) , '.mat']; 
        %save(fname,'centers');        
        
        % 
        H_train = ind2vec(cat_train_codebook);
        subsample_set_factor = params.codebook_subsample_set_factor;
        training_feats = data_train_tran_codebook_mat(:,1:subsample_set_factor:size(data_train_tran_codebook_mat,2));
        H_train_small = H_train(:, 1:subsample_set_factor:size(H_train,2));
        fprintf('%s --- --- Using %d vectors for dictionary learning\n',datetime('now'), size(training_feats,2));
        
        % constant
        %sparsitythres = 30; % sparsity prior
        sparsitythres = 15; % sparsity prior
        %sqrt_alpha = 4; % weights for label constraint term
        %sqrt_beta = 2; % weights for classification err term
        sqrt_alpha = 0.1; % weights for label constraint term
        sqrt_beta = 0.1; % weights for classification err term
        %dictsize = 570; % dictionary size
        dictsize = params.codebookSize; % dictionary size
        iterations = 50; % iteration number
        iterations4ini = 20; % iteration number for initialization

        % dictionary learning process
        % get initial dictionary Dinit and Winit
        fprintf('\nLC-KSVD initialization ');
        [Dinit,Tinit,Winit,Q_train] = initialization4LCKSVD(training_feats,H_train_small,dictsize,iterations4ini,sparsitythres);
        fprintf('done!');

        % % run LC K-SVD Training (reconstruction err + class penalty)
        %fprintf('\nDictionary learning by LC-KSVD1');
        [centers,X1,T1,W1] = labelconsistentksvd1(training_feats,Dinit,Q_train,Tinit,H_train_small,iterations,sparsitythres,sqrt_alpha);
        %%save('.\trainingdata\dictionarydata1.mat','D1','X1','W1','T1');
        %fprintf('done!');
        
        % run LC k-svd training (reconstruction err + class penalty + classifier err)
        %fprintf('\nDictionary and classifier learning by LC-KSVD2')
        %[centers,X2,T2,W2] = labelconsistentksvd2(training_feats,Dinit,Q_train,Tinit,H_train_small,Winit,iterations,sparsitythres,sqrt_alpha,sqrt_beta);
        % D=dict, X=codes,T=transform matrix, W=classifier matrix
        %save('.\trainingdata\dictionarydata2.mat','D2','X2','W2','T2');
        %fprintf('done!');
        
        du_mkdir(params.Codebook_DIR);
        %fname = [params.Codebook_DIR , transf_type, '_', VQ.type, '_', num2str(p), '_', num2str(VQ.Codebook_Size) , '.mat']; 
        save(codebook_filename, 'centers','X1','W1','T1', 'sparsitythres');
        %save(fname, 'centers','X2','W2','T2', 'sparsitythres');
    
    end
    %clear centers all_descriptors_train sse temp_index codebook_size centersMat
    clear sse temp_index centersMat
    
    changed_upstream = true;
    
    % Plot some dictionary atoms
    if params.plot
        plot_atoms = (1:100:params.codebookSize);
        close all
        f = figure('visible','off');
        imshow(centers, []);
        filename = sprintf('%s4_dict', params.Plot_DIR);
        saveas(f,filename,'png');    
        for i = 1:numel(plot_atoms)
            f = figure('visible','off');
            plot(centers(:, plot_atoms(i)));
            filename = sprintf('%s4_dict_atom_%d', params.Plot_DIR, plot_atoms(i));
            saveas(f,filename,'png');
        end    
        close all
    end
    
else

    fprintf('%s --- Loading existing codebook from %s \n',datetime('now'), codebook_filename);
    load(codebook_filename);
end


% ======================================================================
% Encoding (assign codewords + pooling !!): 
% ======================================================================

% Encode training data
code_train_filename = [FEATURE_DIR,'5_code_train','_', num2str(p), '_', transf_type,'_',encType,'_',poolType,'_',normType,'_',num2str(VQ.Codebook_Size),'.mat'];
if changed_upstream || ~exist(code_train_filename)
    fprintf('%s --- Encoding train data\n',datetime('now'));
    for i=1:size(data_train_tran,2)
        fprintf('%s --- --- Encoding class %d, %d signals, %d feature vectors\n',datetime('now'), i, size(data_train_tran{i}, 2), sum(cellfun('length',data_train_tran{i})));
        wb = waitbar(0,'Encoding...');

        for j = 1:size(data_train_tran{i},2)
            if strcmp(encType,'VQ')
                %% Standard vector quantization
                [codeTrain]=Encodeing(data_train_tran{i}{j}',encType,centers',[],[],[],[],poolType,normType,0.5,[]);
            elseif strcmp(encType,'LLC')
                %% LLC coding
                [codeTrain]=Encodeing(data_train_tran{i}{j}',encType,centers',[],[],[],[],poolType,normType,[],[]);
            elseif strcmp(encType,'SA-all')
                %% Soft assignment all centers
                [codeTrain]=Encodeing(data_train_tran{i}{j}','SA',centers',size(centers,2),1,[],[],poolType,normType,[],[]);
            elseif strcmp(encType,'SA-k')
                %% Soft assignment k-nearest centers
                [codeTrain]=Encodeing(data_train_tran{i}{j}','SA',centers',5,1,[],[],poolType,normType,[],[]);
            elseif strcmp(encType,'SPC')    
                %% Sparse coding
                [codeTrain]=Encodeing(data_train_tran{i}{j}',encType,centers',[],[],[],[],...
                                        poolType,params.poolSize, params.poolStride,normType,[],[]);
            end
            %code_train{i}(:,j)=codeTrain;
            code_train{i}{j}=codeTrain';
            %fprintf('%s --- --- --- Finished signal %d, %d feature vectors\n', datetime('now'), j, size(data_train_tran{i}{j}, 2));
            waitbar(j/size(data_train_tran{i},2),wb,'Encoding...');
        end
        close(wb)
        %code_train{i} = cell2mat(code_train{i});
        
    end
    clear codeTrain
    fprintf('%s --- Saving train data codes to %s\n',datetime('now'), code_train_filename);
    save(code_train_filename,'code_train');
    
    changed_upstream = true;
    
    % Plot sample
    if params.plot
        plot_train_sigs = [1, 10, 100, 150];
        plot_windows = [1, 2, 3, 4, 5, 6, 7, 8];
        close all
        for i = 1:size(code_train,2)
            for j = 1:numel(plot_train_sigs)
                if ~iscell(code_train{i}{plot_train_sigs(j)})
                    f = figure('visible','off');
                    plot(code_train{i}{plot_train_sigs(j)});
                    filename = sprintf('%s5_plot_code_train_%d_%d', params.Plot_DIR, i,plot_train_sigs(j));
                    saveas(f,filename,'png');
                else
                    for k = 1:numel(plot_windows)
                        f = figure('visible','off');
                        plot(code_train{i}{plot_train_sigs(j)}(:, plot_windows(k)));
                        filename = sprintf('%s5_plot_code_train_%d_%d_%d', params.Plot_DIR, i,plot_train_sigs(j), plot_windows(k));
                        saveas(f,filename,'png');
                    end
                end
            end
        end
    end
    
    
else
    fprintf('%s --- Loading train data codes from %s\n',datetime('now'), code_train_filename);
    load(code_train_filename);
end

% Encoding test data
code_test_filename = [FEATURE_DIR,'6_code_test','_', num2str(p), '_', transf_type,'_',encType,'_',poolType,'_',normType,'_',num2str(VQ.Codebook_Size),'.mat'];
if changed_upstream || ~exist(code_test_filename)
    fprintf('%s --- Encoding test data\n',datetime('now'));
    for i=1:size(data_test_tran,2)
        fprintf('%s --- --- Encoding class %d, %d signals, %d feature vectors\n',datetime('now'), i, size(data_test_tran{i}, 2), sum(cellfun('length',data_test_tran{i})));
        wb = waitbar(0,'Encoding...');
        
        for j = 1:size(data_test_tran{i},2)
            if strcmp(encType,'VQ')
                %% Standard vector quantization
                [codeTest]=Encodeing(data_test_tran{i}{j}',encType,centers',[],[],[],[],poolType,normType,0.5,[]);
            elseif strcmp(encType,'LLC')
                %% LLC coding
                [codeTest]=Encodeing(data_test_tran{i}{j}',encType,centers',[],[],[],[],poolType,normType,[],[]);
            elseif strcmp(encType,'SA-all')
                %% Soft assignment all centers
                [codeTest]=Encodeing(data_test_tran{i}{j}','SA',centers',size(centers,2),1,[],[],poolType,normType,[],[]);
            elseif strcmp(encType,'SA-k')
                %% Soft assignment k-nearest centers
                [codeTest]=Encodeing(data_test_tran{i}{j}','SA',centers',5,1,[],[],poolType,normType,[],[]);
            elseif strcmp(encType,'SPC') 
                %% Sparse coding
                [codeTest]=Encodeing(data_test_tran{i}{j}',encType,centers',[],[],[],[],...
                                        poolType,params.poolSize, params.poolStride,normType,[],[]);
            end
            %code_test{i}(:,j)=codeTest;
            code_test{i}{j}=codeTest';
            waitbar(j/size(data_train_tran{i},2),wb,'Encoding...');
        end
        close(wb);
        %code_test{i} = cell2mat(code_test{i});
        
    end
    clear codeTest

    fprintf('%s --- Saving test data codes to %s\n',datetime('now'), code_train_filename);
    save(code_test_filename,'code_test');
    
    changed_upstream = true;
    
    % Plot sample
    if params.plot
        plot_test_sigs = [1, 30, 60, 90];
        plot_windows = [1, 2, 3, 4, 5, 6, 7, 8];
        close all
        for i = 1:size(code_test,2)
            for j = 1:numel(plot_test_sigs)
                if ~iscell(code_test{i}{plot_test_sigs(j)})
                    f = figure('visible','off');
                    plot(code_test{i}{plot_test_sigs(j)});
                    filename = sprintf('%s5_plot_code_test_%d_%d', params.Plot_DIR, i,plot_test_sigs(j));
                    saveas(f,filename,'png');
                else
                    for k = 1:numel(plot_windows)
                        f = figure('visible','off');
                        plot(code_test{i}{plot_test_sigs(j)}(:, plot_windows(k)));
                        filename = sprintf('%s5_plot_code_test_%d_%d_%d', params.Plot_DIR, i,plot_test_sigs(j), plot_windows(k));
                        saveas(f,filename,'png');
                    end
                end
            end
        end
    end
    
else
    fprintf('%s --- Loading test data codes from %s\n',datetime('now'), code_train_filename);
    load(code_test_filename);
end

% =========================================================
% Perform classification
% =========================================================
classif_filename = [FEATURE_DIR,'7_classif','_', num2str(p), '_', transf_type,'_',encType,'_',poolType,'_',normType,'_',num2str(VQ.Codebook_Size),'_',classType,'.mat'];
if changed_upstream || ~exist(classif_filename)

    fprintf('%s --- Classification with %s\n',datetime('now'), classType);

    % Get number of classes
    numclasses = size(code_train, 2);  
    
    % Get number of elements in train set and test set
    sizes_train_set = cellfun(@(x) size(x,2), code_train);
    sizes_test_set = cellfun(@(x) size(x,2), code_test);
    
    fprintf(['%s --- --- Total number of training codes: ', repmat( '%g ', 1, numclasses), '\n'], datetime('now'), sizes_train_set);
    fprintf(['%s --- --- Total number of test codes: ', repmat('%g ', 1, numclasses), '\n'], datetime('now'), sizes_test_set);

    % For train set
    set = 1;
    num_CV_splits = 5;        
    traindev_percent = 0.2;
    fprintf('%s --- --- Classifying training set, %dx cross-validation, %g - %g splits\n',...
        datetime('now'), num_CV_splits, (1-traindev_percent), traindev_percent);

    % Run classification on train set with multiple random splits
    for splititer = 1:num_CV_splits  % split iterations

        % Randomly split train-train and train-dev

        for i = 1:numclasses
            [train_train{i}, train_dev{i}] = randsplit(code_train{i}, (1-traindev_percent));
            
            % Concatenate codes from all signals in a class
            train_train{i} = cell2mat(train_train{i});
            train_dev{i}   = cell2mat(train_dev{i});
        end

        % for each distance
        for idist = 1:3
            if classDistType(idist) == 1
                %[conf_matrix{set}{splititer}{idist}, F1{set}{splititer}{idist}] = ...
                [conf_matrix{set}{splititer}{idist}, F1iter(set,splititer,idist)] = ...
                    classify(train_train, train_dev, classType, idist);
                %fprintf('%s --- --- --- CV %d/%d, dist %d, confusion matrix, F1 score:\n',datetime('now'), splititer, num_CV_splits, idist);
                %disp(conf_matrix{set}{splititer}{idist});
                %disp(F1iter(set,splititer,idist));
            end
        end
        fprintf('%s --- --- --- CV %d/%d, F1 score: %g %g %g \n',...
            datetime('now'), splititer, num_CV_splits, ...
            F1iter(set, splititer, 1), F1iter(set, splititer, 2), F1iter(set, splititer, 3));
    end
    F1 = reshape(mean(F1iter, 2), size(F1iter,1), size(F1iter,3));
            
    fprintf('%s --- --- Training set average F1 scores: \n', datetime('now'));
    disp(F1(set,:));

    % For test set
    fprintf('%s --- --- Classifying testing set\n',datetime('now'));
    set = 2;

    % Concatenate codes from all signals in a class
    for i = 1:numclasses
        code_train{i} = cell2mat(code_train{i});
        code_test{i}  = cell2mat(code_test{i});
    end
        
    % For each distance
    for idist = 1:3
        if classDistType(idist) == 1
            [conf_matrix{set}{idist}, F1(set,idist)] = ...
                classify(code_train, code_test, classType, idist);
        end
    end
    
    fprintf('%s --- --- Test set F1 scores: \n', datetime('now'));
    disp(F1(set,:));

    fprintf('%s --- F1 scores\n',datetime('now'));
    disp(F1)
    
    fprintf('%s --- Saving classification results in %s\n',datetime('now'), classif_filename);
    save(classif_filename, 'conf_matrix', 'F1');
else
    load(classif_filename);
end

end

function [conf_matrix, F1] = classify(trainset, testset, type, dist)

    %fprintf('%s --- Classification with %s\n',datetime('now'), type);

    % Number of classes
    numclasses = size(trainset, 2);

    % Get number of elements in train set
    sizes_train_set = cellfun(@(x) size(x,2), trainset);
    sizes_train_set = [0 sizes_train_set];
    cum_sizes = cumsum(sizes_train_set);    
    
    % Preallocate output
    conf_matrix = zeros(numclasses, numclasses);
    
    if strcmp(type,'NN')

        distTypesNN = {'chisq', 'eucdist', 'intersectdis'};
        
        % Prepare train code for comparing distances to
        z1 = cell2mat(trainset);
        z1 = full(z1);           
        
        % for each class
        for i = 1:numclasses

            % Data for evaluation, for this class
            x = full(testset{i});

            % Compute distances
            dist_chi = slmetric_pw(x, z1, distTypesNN{dist});

            [~,idx_min] = min(dist_chi, [], 2);

            for j = 1:size(trainset,2) 
                conf_matrix(i,j) = sum(double(idx_min >= 1 + cum_sizes(j) & idx_min <= cum_sizes(j+1)));
            end
        end

    elseif strcmp(type,'SVM')
    
        approx_order = 5;

        % Prepare data
        train_data = cell2mat(trainset);
        eval_data  = cell2mat(testset);
        %train_data=train_data';
        %eval_data = eval_data';    

        % Scale all data
        Lower = 0;
        Upper = 1;
        [train_data, eval_data] = scaleSVM(train_data',eval_data',Lower,Upper);
        train_data = train_data';
        eval_data  = eval_data';

        % Get number of elements in train set
        sizes_train_data = cellfun(@(x) size(x,2), trainset);
        sizes_eval_data  = cellfun(@(x) size(x,2), testset);

        % Create the label vectors
        train_label = [];
        eval_label  = [];
        for i = 1:numclasses
               train_label = [train_label; i * ones(sizes_train_data(i),1)];
               eval_label  = [eval_label;  i * ones(sizes_eval_data(i) ,1)];
        end

        if dist == 1
            psiTrain = vl_homkermap(train_data, approx_order, 'kchi2');
            psiTest  = vl_homkermap(eval_data, approx_order, 'kchi2');
            psiTrain=psiTrain';
            psiTest=psiTest';
            % Scale data
            Lower = 0;
            Upper = 1;
            [psiTrain,psiTest] = scaleSVM(psiTrain,psiTest,Lower,Upper);

            model = svmtrain(train_label, psiTrain, ['-t 0 -c 2 -q']);
            [predict_label, hit_rate, dec_values] = svmpredict(eval_label, psiTest, model); % test the training data

        elseif dist == 2
            
            % Get best parameters
            bestcv = 0;
            bestc = 0;
            for log2c = -1:4
              for log2g = -5:1
                cmd = ['-v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g) '-q']; % Nic: added -q
                cv = svmtrain(train_label, train_data', cmd);
                if (cv > bestcv)
                  bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
                end

                fprintf('%g %g %g (best c=%g, g=%g, rate=%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
              end
            end

            model = svmtrain(train_label, train_data', ['-c ' num2str(bestc) ' -g ' num2str(bestg) ' -q']);
            [predict_label, hit_rate, dec_values] = svmpredict(eval_label, eval_data', model); % test the training data
            
        elseif dist == 3

            psiTrain = vl_homkermap(train_data, approx_order, 'kinters');
            psiTest = vl_homkermap(eval_data, approx_order, 'kinters');
            psiTrain=psiTrain';
            psiTest=psiTest';
            % Scale data
            Lower = 0;
            Upper = 1;
            [psiTrain,psiTest]=scaleSVM(psiTrain,psiTest,Lower,Upper);

            model = svmtrain(train_label, psiTrain, ['-t 0 -c 2 -q']);
            [predict_label, hit_rate, dec_values] = svmpredict(eval_label, psiTest, model); % test the training data

        else
            error('Unknown distance type');
        end
            
        % Confusion matrix from SVM predictions
        for i = 1:numclasses               % for each class
            idx_i = (eval_label==i);
            pred_i = predict_label(idx_i);

            for j = 1:numclasses           % for each class
                conf_matrix(i,j) = sum(pred_i==j);
            end
        end
        
%     elseif strcmp(classType,'LC-KSVD')    
% 
%         % classification process
% 
%         dictfilename = [params.Codebook_DIR , transf_type, '_', VQ.type, '_', num2str(p), '_', num2str(VQ.Codebook_Size) , '.mat']; 
%         load(dictfilename);
% 
%         %testing_feats = cell2mat(code_test);
%         for i=1:size(data_tran_test,2)
%             for j = 1:size(data_tran_test{i},2)
%                 [codeTest] = Encodeing(data_tran_test{i}{j}',encType,centers',[],[],[],[],'sum-pooling','No',[],[]);
%                 code_test{i}(:,j)=full(codeTest);
%                 disp(j)
%             end
%         end
%         testing_feats = cell2mat(data_tran_test);
% 
%         for i = 1:size(centers,2)
%             centers(:,i) = centers(:,i) / norm(centers(:,i),2);
%         end
% 
%         [prediction2, accuracy2] = classification(centers, W2, testing_feats, H_test, sparsitythres);
%         fprintf('\nFinal recognition rate for LC-KSVD2 is : %.03f ', accuracy2);
%     end
    
    end

    % Compute F1 scores
    for k = 1:numclasses
        Pk = conf_matrix(k,k) / sum(conf_matrix(:,k));
        Rk = conf_matrix(k,k) / sum(conf_matrix(k,:));
        F1class(k) = 2 * Pk * Rk / (Pk + Rk);
    end
    F1class(isnan(F1class)) = []; % Remove NaN values
    F1 = sum(F1class) / numel(F1class);
end


function [data1, data2] = randsplit(data, percent, dim)
    % Split data randomply into two sets alongside dimension
    
    % Direction
    if nargin == 2
        dim = 2; % default dimension is 2 (split horizontally)
    end
    
    numdata = size(data, dim);
    limit = round(percent * numdata);
    rp = randperm(numdata);
    indicesA = rp(1:limit);
    indicesB = rp(limit+1 : numdata);

    if dim == 1
        data1 = data(indicesA, :);
        data2 = data(indicesB, :);
    elseif dim == 2
        data1 = data(:, indicesA);
        data2 = data(:, indicesB);
    else
        error('Can only split along first or second dimension');
    end
end