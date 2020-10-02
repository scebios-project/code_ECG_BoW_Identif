function [code]=Encodeing(feature,Method,codebook,KNN,beta,lambda,sigma,pooling, poolSize, poolStride, Normalization,alpha,A)
%% input feature :N*dim matrix
%% input codebook : N*dim matrix
%% KNN a integer,belong [1 size(codebook,1)]
%% beat : param for SA Encodeing
%% lambda : param for SPC Encodeing
%% sigma : param for KCB Encodeing
%% pooling  select 'sum-pooling' or 'max-pooling',the VQ and FK Encodeing default value is 'sum-pooling'
%% Normalization : 'L1','L2','P+L1','P+L2'
%% alpha : param for Power Normalization
%% A : speed the SPC encodeing at big dataset,
[feature_vote]=Vote(feature,Method,codebook,KNN,beta,lambda,sigma,A);
switch pooling
    case 'max-pooling'
        code=max(feature_vote,[],1);
    case 'sum-pooling'    
        code=sum(feature_vote,1);
    case 'var-pooling'    
        code=var(feature_vote,1);        
    case 'sum-pooling-size'
        % pool in groups of `pooling_size`, stride by `pooling_stride`
        code = zeros(1 + floor((size(feature_vote, 1) - poolSize)/poolStride), size(feature_vote, 2)); % preallocate
        k = 1;
        for i = 1:poolStride:(size(feature_vote, 1) - poolSize)
            code(k, :)=sum(feature_vote(i:(i+poolSize-1), :),1);
            k=k+1;
        end
    case 'mix-ord-pooling'
    case 'sum-max-pooling'
    case 'No'
        code = feature_vote;
end

switch Normalization
    case 'Power'
        code=sign(code).*(abs(code).^alpha);
    case 'L1'
        code=code./(sum(abs(code),2)+eps);
    case 'L2'
        for i = 1:size(code, 1)
            code(i,:) = code(i,:) ./ (sum(code(i,:).^2,2).^(0.5)+eps);
        end
    case 'Power+L1'
        code=sign(code).*(abs(code).^alpha);
        code=code./(sum(abs(code),2)+eps);
    case 'Power+L2'
        code=sign(code).*(abs(code).^alpha);
        code=code./(sum(code.^2,2).^(0.5)+eps);
    case 'mean'
        code = code / size(feature_vote, 1);
    case 'No'
end
