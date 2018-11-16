close all;
clear
clc

%% Constants
g = 9.807;
dataNum = 85;
files = 512; %510, the dir command consider '.' and '..' as files
startHour = 2795;
sensorSensivity = 97.4;

%% Variables
rmsData = zeros(dataNum,1);
t = zeros(dataNum,1);

%% Load data
list = dir('/home/magdiel/Documentos/TCC/matlab/data/vib_25h_RACAL/'); 
cont = 1;
% figure;
for i = 3:files %starts at 3 to disconsider '.' and '..' 
   name = list(i).name;
   split = strsplit(name, 'h');
   if strcmp(split(2),'_Track5.mat')
        %% Feature extraction
        t(cont,1) = str2double(split(1)) - startHour;
        temp = load(strcat('/home/magdiel/Documentos/TCC/matlab/data/vib_25h_RACAL/',name));
        normalized = (temp.X/sensorSensivity)*g; 
        
        rmsData(cont,1) = rms(normalized(256:end-256));
%         
%         Fs = 25000;
%         Fd = fft(temp.X);
%         N=length(Fd);
%         S2 = abs(Fd/N);
%         S1 = S2(1:(N/2+1));
%         S1(2:end-1) = 2*S1(2:end-1);
% 
%         f = Fs*(0:(N/2))/N;
% 
%         
%         plot(f,S1) 
%         title('Single-Sided Amplitude Spectrum of Y(t)')
%         xlabel('f (Hz)')
%         ylabel('|S1(f)|')
%         hold on
%         t(cont,1) = str2double(split(1));
%         webwrite(['http://localhost:9000/' 'readings'],'sensor','5be9bf941cbcdd3a95bf2b09','value',rmsData(cont,1),'date',t(cont,1));
        cont = cont + 1;
   end
end

%% Select data to analysis
firstPoint = 18;
midPoint = 28;
endPoint = 35;

% construction rmsData
P = t(firstPoint:midPoint)';
Y = rmsData(firstPoint:midPoint)';
N = length(P);

% for i = 1:length(P)
%     webwrite(['http://localhost:9000/' 'readings'],'sensor','5be9bfa31cbcdd3a95bf2b0a','value',Y(i),'date',P(i));
% end

% Ptest = t(N+1:N+4)';
% Ytest = rmsData(N+1:N+4)';

Ptest=t(midPoint+1:endPoint)';
Ytest=rmsData(midPoint+1:endPoint)';

Pl = [P, Ptest];
Yl = [Y, Ytest];
Nl = length(Pl);

%% FFT 

T = 25 * 3600;
Fs = 1/T;
Fd = fft(Y);

S2 = abs(Fd/N);
S1 = S2(1:(N/2+1));
S1(2:end-1) = 2*S1(2:end-1);

f = Fs*(0:(N/2))/N;

% plot result
figure;
plot(f,S1) 
title('Single-Sided Amplitude Spectrum of Y(t)')
xlabel('f (Hz)')
ylabel('|S1(f)|')

%% RNN - Recurrent Neural Network

%create rnn
nn = [1 5 5 1];
dIn = [1,2,3];
dIntern=[];
dOut=[1,2,3,4];

net = CreateNN(nn,dIn,dIntern,dOut);
net = train_LM(P,Y,net,400,1e-4);

%test
ytest = NNOut(Pl,net);

% Plot results

figure;
hold on
plot(Pl,Yl,'*');
plot(Pl,ytest, 'r');
hold off;

%% Least squares Polinomial

% Polinomial
A = [P'.^2 P'.^1 P'.^0 cos(P')];
Atest = [Pl'.^2 Pl'.^1 Pl'.^0 cos(Pl')];

th = pinv(A)*Y';

y_ap = Atest*th;

%plot results
figure;
hold on
plot(Pl,Yl,'.');
plot(Pl,y_ap, 'r');
hold off;

%% Least squares arx

% Number of parameter to estimate
na = 3;
nb = 2;

dim = na+nb+1;
m = max(na,nb+1);

yest = zeros(N,1);

% Construction of observation matrix
phi = zeros(N,dim);
for i=m+1:N
    phi(i,:) = [-Y(i-(1:na)) P(i-(0:nb))];
end

% estimated parameters
theta2 = (phi'*phi\phi')*Y';

print('antes');
yest2 = zeros(1,Nl);
yest2(1:(m+1)) = Y(1:(m+1));
for i = m+1:Nl
   yest2(i) = -yest2(i-1)*theta2(1)-yest2(i-2)*theta2(2)-yest2(i-3)*theta2(3)+Pl(i)*theta2(4)+Pl(i-1)*theta2(5)+Pl(i-2)*theta2(6);
end
print('dps');
%plot results
figure;
hold on
plot(Pl,Yl,'*');
plot(Pl,yest2, 'r');
hold off;

%% Alisamento Exponencial Simples

% define interval to iterate
interval = 0.1:0.001:0.99;
e = zeros(length(interval),1);
cont =1 ;

% calculate for all alfas and locate the minimum error
for alfa = interval
    e2=0;
    prev= zeros(length(Y),1);
    prev(1) = Y(1);
    for i = 2:length(Y)
       prev(i) = alfa*Y(i-1)+(1-alfa)*prev(i-1);
       e2 = e2 + (Y(i)-prev(i))^2;
    end
    e(cont) = e2;
    cont= cont+1;
end

figure;
plot(interval,e);
title('Soma dos quadrados dos erros');

%find the best alfa
[minError, indexMin] = min(e);
alfa = interval(indexMin);

%calculate the results and predict
y_aes = zeros(length(Yl)+1,1);
y_aes(1) = Yl(1);
for i = 2:length(Yl)+1
    y_aes(i) = alfa*Yl(i-1)+(1-alfa)*y_aes(i-1);
end

figure;
hold on;
plot(Pl,Yl,'*');
plot([Pl Pl(end)+25],y_aes);
hold off;
