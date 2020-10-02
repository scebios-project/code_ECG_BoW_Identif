function [scale_train_data,scale_test_data]=scaleSVM(train_data,test_data,Lower,Upper)
%--------------------------------------------------------------------------
% DESCRIPTION: Used to Scale data uniformly
%--------------------------------------------------------------------------
Data=train_data;
[MaxV, I]=max(Data);
[MinV, I]=min(Data);
[R,C]= size(Data);
scaled=(Data-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
for i=1:size(Data,2)
    if(all(isnan(scaled(:,i))))
        scaled(:,i)=0;
    end
end
scale_train_data=scaled;

%###### SCALE THE TEST DATA TO THE RANGE OF TRAINING DATA ###########
Data=test_data;
[R,C]= size(Data);
scaled=(Data-ones(R,1)*MinV).*(ones(R,1)*((Upper-Lower)*ones(1,C)./(MaxV-MinV)))+Lower;
for i=1:size(Data,2)
    if(all(isnan(scaled(:,i))))
        scaled(:,i)=0;
    end
end
scale_test_data=scaled;
