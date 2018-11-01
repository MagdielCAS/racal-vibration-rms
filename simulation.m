close all;
clear
clc

%% Constants
g = 9.807;
dataNum = 85;
files = 512; %510, the dir command consider '.' and '..' as files
startHour = 2795;

%% Variables
data = zeros(dataNum,1);
t = zeros(dataNum,1);

%% Load data
list = dir('/home/magdiel/Documentos/TCC/matlab/data/vib_25h_RACAL/'); 
cont = 1;
for i = 3:files %starts at 3 to disconsider '.' and '..' 
   name = list(i).name;
   split = strsplit(name, 'h');
   if strcmp(split(2),'_Track1.mat')
        temp = load(strcat('/home/magdiel/Documentos/TCC/matlab/data/vib_25h_RACAL/',name));
        data(cont,1) = rms(temp.X);
%         t(cont,1) = str2double(split(1)) - startHour;
        t(cont,1) = str2double(split(1));
        cont = cont + 1;
   end
end

%% RNN - Recurrent Neural Network

% construction data
N = 20;
P = t(1:N)';
Y = data(1:N)';

Ptest = t(N+1:N+4)';
Ytest = data(N+1:N+4)';

%create rnn
nn = [1 4 4 1];
dIn = [1,2,3];
dIntern=[];
dOut=[1,2,3,4];

net = CreateNN(nn,dIn,dIntern,dOut);
net = train_LM(P,Y,net,400,1e-4);

%test
ytest = NNOut([P, Ptest],net);

%% Plot Data

figure;
hold on
plot(t(1:N+4),data(1:N+4),'*');
plot([P, Ptest],ytest, 'r');
hold off;

%% Least squares approximation

% Polinomial
Pl = [P, Ptest];
Yl = [Y, Ytest];

A = [Pl'.^5 Pl'.^4 Pl'.^3 Pl'.^2 Pl'.^1 Pl'.^0];

th = pinv(A)*Yl';

y_ap = A*th;

%plot results
figure;
hold on
plot(Pl,Yl,'.');
plot(Pl,y_ap, 'r');
hold off;

%% MNQR

% Number of parameter to estimate
na = 3;
nb = 1;

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

yest2 = phi*theta2;
% Getting TF parameters
den = [1 theta2(1:na)'];
num = [0 theta2(na+1:dim)'];
% Discrete TF
Hs2 = tf(num,den,1,'variable','z^-1');
% Continuous TF
Hc2 = d2c(Hs2);

%plot results
figure;
hold on
plot(Pl,Yl,'.');
plot(P,yest2, 'r');
hold off;
figure;
plot(lsim(Hc2,Pl,(1:(N+4))'));
