function tran = series_transform(data,sub_length,inter_point,transf_type, size_feat, level)
%% sliding a subwindows along a time series and extract a feature
%% vector for each sub sequences by transformation (e.g.,'wavelet')
%% Created by Jin Wang    06/09/2011

% data    1*D matrix    D is the length of time series
% sub_length   the length of the subwindow


tran=[];
for i = 1:inter_point:(length(data)-sub_length+1)
    sub_sequence = data(i:i+sub_length-1);
    
    sub_sequence = (sub_sequence-mean(sub_sequence))/std(sub_sequence);
    
    if strcmp(transf_type,'dwt')
        %[sub_tran aa] = dwt(sub_sequence,'db3');
        qmf = MakeONFilter('Daubechies',6);
        [sub_tran] = FWT_PO(sub_sequence,level,qmf);
        size_dwt = size_feat;   % 32; 64;
        sub_tran=sub_tran(1:size_dwt);
    elseif strcmp(transf_type,'randproj')
        global Phi
        [sub_tran] = sub_sequence*Phi';
    elseif strcmp(transf_type,'dct')
        size_dct = size_feat;   % 32; 64;
        [sub_tran] = dct(sub_sequence);
        sub_tran=sub_tran(1:size_dct);
    elseif strcmp(transf_type,'combined')
        qmf = MakeONFilter('Daubechies',6);
        [sub_tran1] = FWT_PO(sub_sequence,level,qmf);
        size_dwt = size_feat;   % 32; 64;
        sub_tran1=sub_tran1(1:size_dwt);
        
        size_dct = size_feat;   % 32; 64;
        [sub_tran2] = dct(sub_sequence);
        sub_tran2=sub_tran2(1:size_dct);
        
        global Phi
        [sub_tran3] = sub_sequence*Phi';
        
        [sub_tran]=[sub_tran1 sub_tran2 sub_tran3];
    elseif strcmp(transf_type,'none')
        sub_tran = sub_sequence;
    end
    
    bb=norm(sub_tran);
    sub_tran = sub_tran/bb;
    tran=[tran,sub_tran'];
end