%this script is used to calculate the pearson linear correlation
%coefficient and root mean sqaured error after regression

%get the objective scores computed by the IQA metric and the subjective
%scores provided by the dataset
matData = load('SSIMOnLIVE.mat');
SSIMOnLIVE = matData.SSIMOnLIVE;
ssimValues = SSIMOnLIVE(:,1);
mos = SSIMOnLIVE(:,2);

%plot objective-subjective score pairs
p = plot(ssimValues,mos,'+');
set(p,'Color','blue','LineWidth',1);

%initialize the parameters used by the nonlinear fitting function
beta(1) = max(mos);
beta(2) = min(mos);
beta(3) = mean(ssimValues);
beta(4) = 0.1;
beta(5) = 0.1;
%fitting a curve using the data
[bayta ehat,J] = nlinfit(ssimValues,mos,@logistic,beta);
%given a ssim value, predict the correspoing mos (ypre) using the fitted curve
[ypre junk] = nlpredci(@logistic,ssimValues,bayta,ehat,J);

RMSE = sqrt(sum((ypre - mos).^2) / length(mos));%root meas squared error
corr_coef = corr(mos, ypre, 'type','Pearson'); %pearson linear coefficient

%draw the fitted curve
t = min(ssimValues):0.01:max(ssimValues);
[ypre junk] = nlpredci(@logistic,t,bayta,ehat,J);
hold on;
p = plot(t,ypre);
set(p,'Color','black','LineWidth',2);
legend('Images in LIVE','Curve fitted with logistic function', 'Location','SouthWest');
xlabel('Objective score by SSIM');
ylabel('MOS');

