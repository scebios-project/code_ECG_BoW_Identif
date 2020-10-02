close all
clear all

filename = '../data/CYBHI/2_feat_1_dwt_512_32_data_train_tran.mat';

load(filename)


s1 = data_train_tran{1}{1}(:, 1);
s2 = data_train_tran{2}{1}(:, 1);
s3 = data_train_tran{3}{1}(:, 1);
s4 = data_train_tran{4}{1}(:, 1);

figure
subplot(4,1,1); plot(s1);
title('Exemple de trasaturi primare');
subplot(4,1,2); plot(s2);
subplot(4,1,3); plot(s3);
subplot(4,1,4); plot(s4);


% Save as fig
savefig('Features.fig')

% Save as png
saveas(gcf, 'Features.png')

% Save as pdf
ps = get(gcf, 'Position');
ratio = (ps(4)-ps(2)) / (ps(3)-ps(1));
paperWidth = 10;
paperHeight = paperWidth*ratio;
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'papersize', [paperWidth paperHeight]);
set(gcf, 'PaperPosition', [0    0   paperWidth paperHeight]);
print(gcf, '-dpdf', 'Features.pdf');
