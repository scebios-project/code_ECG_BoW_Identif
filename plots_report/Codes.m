close all
clear all

filename = '../data/CYBHI/5_code_train_1_dwt_SPC_sum-pooling_L2_500.mat';

load(filename)

s1 = code_train{1}{:, 1};
s2 = code_train{2}{:, 1};
s3 = code_train{3}{:, 1};
s4 = code_train{4}{:, 1};

figure
subplot(4,1,1); plot(s1);
title('Exemple de histograme ale unor persoane diferite');
subplot(4,1,2); plot(s2);
subplot(4,1,3); plot(s3);
subplot(4,1,4); plot(s4);

% Save as fig
savefig('Codes.fig')

% Save as png
saveas(gcf, 'Codes.png')

% Save as pdf
ps = get(gcf, 'Position');
ratio = (ps(4)-ps(2)) / (ps(3)-ps(1));
paperWidth = 10;
paperHeight = paperWidth*ratio;
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'papersize', [paperWidth paperHeight]);
set(gcf, 'PaperPosition', [0    0   paperWidth paperHeight]);
print(gcf, '-dpdf', 'Codes.pdf');
