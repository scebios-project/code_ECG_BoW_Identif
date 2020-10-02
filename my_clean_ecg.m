function [ecg_clean] = my_clean_ecg(ecg_signal,fs, param)

% Setup parameters
% Set tau value (20 ms)
tau = 10;                %10 for fs=500 Hz; 4 for fs=200 Hz;
% Set segment duration  (0.4 ms --> 200 samples at 500 Hz)
offset_1 = 39;           %29 for fs=500 Hz; 12 for fs=200 Hz [3D]; 39 for fs=500 Hz [4D]
offset_2 = 170+2*tau;    %170 for fs=500 Hz; 68 for fs=200 Hz; 

% ATTENTION!!! DEBUG HACK NiCl 28.09.2019
extra = 256 - offset_1 - offset_2 -1;
offset_1 = offset_1 + extra;  % ensure total length = 256

% Set normalization parameters
%a_coeff = 180;
%a_offset = 10;
a_coeff = param.a_coeff;
a_offset = param.a_offset;
% Type of R detector
type_detector = param.type_detector;    % 'pan-tompkins'; 'DWT'
% Type of threshold
type_thresh = param.type_thresh;        % 'fixed' ; 'percent'
% Set distance threshold values for quality check
%dist_thresh=0.25;
dist_thresh = param.dist_thresh;
%type_split = param_
num_std_limit = param.num_std_limit;  % how many std's to allow

% Amplitude saturation
mean_ecg = mean(ecg_signal);
zeroMean_ecg_signal = ecg_signal - mean_ecg;
std_ecg = std(zeroMean_ecg_signal);
limit_ecg_signal = mean_ecg + sign(zeroMean_ecg_signal).*min(abs(zeroMean_ecg_signal), 3*std_ecg);

% Detect R-peaks
%type_detector = 'pan-tompkins';   % 'pan-tompkins'; 'DWT'
if strcmp(type_detector,'pan-tompkins')
    [qrs_amp_raw,ind,delay]=pan_tompkin(limit_ecg_signal,fs,0);

    diff_idxR=[mean(diff(ind)) diff(ind)];
    idx_cleanR = find(abs(diff_idxR-mean(diff(ind))) < 0.15*mean(diff(ind)));
    ind_clean = ind(idx_cleanR);
    clear ind
    qrspeaks = ind_clean;
    clear ind_clean
elseif strcmp(type_detector,'DWT')
    wt = modwt(limit_ecg_signal,5);
    wtrec = zeros(size(wt));
    wtrec(4:5,:) = wt(4:5,:);
    y = imodwt(wtrec,'sym4');
    y = abs(y).^2;
    [qrs,qrspeaks] = findpeaks(y,'MinPeakHeight',0.2*mean(y), 'MinPeakDistance',200);
    clear wt wrec y
end

% Normalize data
type_norm='std';   % 'range'; 'std'
s = [];
for k = 1:length(qrspeaks)
    if qrspeaks(k) > offset_1 & (qrspeaks(k) + offset_2) < length(limit_ecg_signal)
        s1 = limit_ecg_signal(qrspeaks(k)-offset_1:qrspeaks(k)+offset_2);
        
        % % Filter s
        % s1 = filter(param.filter, s1);        
        
        if strcmp(type_norm,'range')
            norm_s = a_offset + a_coeff*(s1-min(s1))/(max(s1)-min(s1));
        elseif strcmp(type_norm,'std')
            norm_s = (s1-mean(s1))/std(s1);
            norm_s = norm_s + abs(min(norm_s));
            norm_s = a_offset + (a_coeff/max(norm_s))*norm_s;
        end
        s = [s norm_s];
    else
    end
end



% Quality check for s
type_quality='norm';   % 'norm'; 'pca'; 'NCCC'
if strcmp(type_quality,'norm')
    if length(s) ~= 0
        size_s=size(s);
        mean_s=mean(s')';

        dist_s=dist(s',mean_s)/norm(mean_s);

        % Nic: moved inside the if()
        % Nic: add fixed / percentage option
        if  strcmp(type_thresh, 'fixed')
            idx_clean = find(dist_s < dist_thresh);   % set minimum value for dist_s!
        elseif strcmp(type_thresh, 'percent')
            dist_s_sorted = sort(dist_s);
            last_in = floor(dist_thresh * numel(dist_s_sorted));
            
            if last_in == 0  % check & fix boundary
                thresh = dist_s_sorted(1) - 1;  % no elements go in, just ensure thresh is smaller than min
            elseif last_in == numel(dist_s_sorted) % check & fix boundary
                thresh = dist_s_sorted(last_in) + 1;  % all elements go in, just ensure thresh is bigger
            else
                thresh = (dist_s_sorted(last_in) + dist_s_sorted(last_in + 1))/2;  % take middle between last inside and first outside
            end
            
            idx_clean = find(dist_s < thresh); 
        else
            error('Not implemented yet!');
        end

        s_clean = s(:,idx_clean);   
    
    else
        s_clean = s; % Nic
    end


elseif strcmp(type_quality,'pca')
    mean_s=mean(s')';
    s_center=s-repmat(mean_s,1,size(s,2));

    thresh_pca=0.1;
    numvecs=round(thresh_pca*size(s,1));
    [V_pca,eigvalues_all]=pc_evectors(s_center,numvecs);
    
    size_eigvalues=size(eigvalues_all);
    s_reconstruct=V_pca*V_pca'*s_center + repmat(mean_s,1,size(s,2));

    for j=1:size(s,2)
        err_reconstruct(j)=norm(s(:,j)-s_reconstruct(:,j))/norm(s(:,j));
    end

    thresh_clean=0.004;
    idx_clean = find(err_reconstruct < thresh_clean);   % set minimum value for dist_s!
    s_clean = s(:,idx_clean);

elseif strcmp(type_quality,'NCCC')
    init_cluster=min(3,size(s,2));
    NCCC=zeros(size(s,2),size(s,2));
    for p = 1:size(s,2)
        NCCC(p,:)=max(xcorr2(s(:,p),s)/norm(s(:,p))^2);
    end
    ACORR=mean(NCCC');
    [sortACORR,idx_ACORR]=sort(ACORR,'descend');
    s_clean=s(:,idx_ACORR(1:init_cluster));
    mean_sortACORR=mean(sortACORR(1:init_cluster));

    sortACORR_clean = sortACORR(1:init_cluster);

    thresh_NCCC=0.06*mean_sortACORR;

    for m=init_cluster+1:length(idx_ACORR)
       %if (sortACORR(m) > 0.5) && (mean_sortACORR-sortACORR(m) < thresh_NCCC)
       if (sortACORR(m) > 0.5) && (mean_sortACORR-sortACORR(m) < thresh_NCCC) && (dist(s(:,idx_ACORR(m))',mean(s_clean')')/norm(mean(s_clean'))<0.35)
           s_clean=[s_clean s(:,idx_ACORR(m))];
           sortACORR_clean = [sortACORR_clean sortACORR(m)];
           mean_sortACORR=mean(sortACORR_clean);
           thresh_NCCC=0.06*mean_sortACORR;
       else
       end
    end
end
ecg_clean=s_clean;
clear s_clean

% %% Plot data
% %Plot original ECG signal and R peaks
% figure
% plot(limit_ecg_signal)
% hold on
% plot(qrspeaks, limit_ecg_signal(qrspeaks), 'ro')
% hold off
% 
% % Plot original data
% figure
% plot(s)
% 
% % Plot cleaned data
% figure
% plot(ecg_clean)   