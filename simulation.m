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
deltaRms = zeros(dataNum-1,1);
t = zeros(dataNum,1);

%% Load data
list = dir('/home/magdiel/Documentos/TCC/matlab/data/vib_25h_RACAL/'); 
cont = 1;

for i = 3:files %starts at 3 to disconsider '.' and '..' 
   name = list(i).name;
   split = strsplit(name, 'h');
   if strcmp(split(2),'_Track5.mat')
        %% Feature extraction
        temp = load(strcat('/home/magdiel/Documentos/TCC/matlab/data/vib_25h_RACAL/',name));
        normalized = (temp.X/sensorSensivity)*g; 
        
        rmsData(cont,1) = rms(normalized(256:end-256));

        t(cont,1) = str2double(split(1));
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


Ptest=t(midPoint+1:endPoint)';
Ytest=rmsData(midPoint+1:endPoint)';

Pl = [P, Ptest];
Yl = [Y, Ytest];
Nl = length(Pl);

%% Autocorrelation

[acf, lags, bounds] = autocorr(Y);

figure;
hold on;
plot(lags,acf,'-r');
plot(lags,bounds(1)*ones(size(lags)),'-b','LineWidth',1.5);
plot(lags,bounds(2)*ones(size(lags)),'-b','LineWidth',1.5);
ylim([-1 1]);
grid on
title('Autocorrelação dos dados de treinamento')
ylabel('Autocorrelação')
xlabel('Atrasos (Lags)')

%% Delta RMS


deltaRms = abs(Yl(2:end) - Yl(1:end-1));
figure;
plot(Pl(2:end),deltaRms);
title('Delta RMS da janela de dados para análise.');
xlabel('Horas em funcionamento');
ylabel('Delta RMS');

%% RNN - Recurrent Neural Network

%create rnn
nn = [1 5 5 1];
dIn = [1,2,3];
dIntern=[];
dOut=[1,2,3,4];

init = P(1);
last = P(end);
miny = min(Y);
maxy = max(Y)+300;



net = CreateNN(nn,dIn,dIntern,dOut);
net = train_LM((P-init)/(last-init),(Y-miny)/(maxy-miny),net,5000,1e-10);

ytest = NNOut((Pl-init)/(last-init),net)*(maxy-miny)+miny;

% Plot results
figure;
subplot(3,1,1);
plot(Pl,Yl,'*');
hold on
plot(Pl,ytest, '--r');
legend('Observado','Estimado')
title('Rede Neural Recorrente')
xlabel('Horas em funcionamento')
ylabel('RMS')
hold off
subplot(3,1,2);
deltaRms = abs(ytest(2:end) - ytest(1:end-1));
plot(Pl(2:end),deltaRms);
title('Delta RMS dos dados estimados.');
xlabel('Horas em funcionamento');
ylabel('Delta RMS');
subplot(3,1,3);
plot(Pl,sqrt((Yl-ytest).^2));
title('Erro de estimação');
xlabel('Horas em funcionamento');
ylabel('Erro');
str(1) = {'RMSE: '};
str(2) = {sqrt(sum((Y-ytest(1:N)).^2)/N)};
str(3) = {'MAE: '};
str(4) = {sum(abs(Y-ytest(1:N)))/N};
x = 3220;
str(5) = {'MAPE: '};
str(6) = {(sum(abs((Y-ytest(1:N))./Y))/N)*100};
y1 = ylim;
y=(y1(2)-y1(1))/2;
text(x,y,str);

%% Least squares Polinomial

% Polinomial
A = [P'.^2 P'.^1 P'.^0 cos(P')];
Atest = [Pl'.^2 Pl'.^1 Pl'.^0 cos(Pl')];

th = pinv(A)*Y';

y_ap = Atest*th;

%plot results
figure;
subplot(3,1,1);
hold on
plot(Pl,Yl,'*');
plot(Pl,y_ap, '--r');
legend('Observado','Estimado')
title('Modelo Polinomial')
xlabel('Horas em funcionamento')
ylabel('RMS')
hold off;
subplot(3,1,2);
deltaRms = abs(y_ap(2:end) - y_ap(1:end-1));
plot(Pl(2:end),deltaRms);
title('Delta RMS dos dados estimados.');
xlabel('Horas em funcionamento');
ylabel('Delta RMS');
subplot(3,1,3);
plot(Pl,sqrt((Yl'-y_ap).^2));
title('Erro de estimação');
xlabel('Horas em funcionamento');
ylabel('Erro');
str(1) = {'RMSE: '};
str(2) = {sqrt(sum((Y-y_ap(1:N)').^2)/N)};
str(3) = {'MAE: '};
str(4) = {sum(abs(Y-y_ap(1:N)'))/N};
str(5) = {'MAPE: '};
str(6) = {(sum(abs((Y-y_ap(1:N)')./Y))/N)*100};
y1 = ylim;
y=(y1(2)-y1(1))/2;
text(x,y,str);

%% Least squares arx

% Number of parameter to estimate
na = 3;
nb = 3;

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


yest2 = zeros(1,Nl);
yest2(1:m) = Y(1:m);
for i = m:Nl
   yest2(i) = -yest2(i-1)*theta2(1)-yest2(i-2)*theta2(2)-yest2(i-3)*theta2(3)+Pl(i)*theta2(4)+Pl(i-1)*theta2(5)+Pl(i-2)*theta2(6)+Pl(i-3)*theta2(7);
end
%plot results
figure;
subplot(3,1,1);
hold on
plot(Pl,Yl,'*');
plot(Pl,yest2, '--r');
legend('Observado','Estimado')
title('Modelo ARX')
xlabel('Horas em funcionamento')
ylabel('RMS')
hold off;
subplot(3,1,2);
deltaRms = abs(yest2(2:end) - yest2(1:end-1));
plot(Pl(2:end),deltaRms);
title('Delta RMS dos dados estimados.');
xlabel('Horas em funcionamento');
ylabel('Delta RMS');
subplot(3,1,3);
plot(Pl,sqrt((Yl-yest2).^2));
title('Erro de estimação');
xlabel('Horas em funcionamento');
ylabel('Erro');
str(1) = {'RMSE: '};
str(2) = {sqrt(sum((Y-yest2(1:N)).^2)/N)};
str(3) = {'MAE: '};
str(4) = {sum(abs(Y-yest2(1:N)))/N};
str(5) = {'MAPE: '};
str(6) = {(sum(abs((Y-yest2(1:N))./Y))/N)*100};
y1 = ylim;
y=(y1(2)-y1(1))/2;
text(x,y,str);


%% Estimador recursivo aplicando filtro de Kalman - modelo ARMAX

na = 3;
nb = 3;
nc = 1;
d = 0;

dim = na+nb+1+nc;
m = max([na nb+1 nc]);

% Geracao de ruido aleatorio (gaussiano, média 0, variância 1)
er = randn(N, 1);
el = randn(Nl,1);

% vetor de parametros
theta = [zeros(1,na) zeros(1,nb+1) zeros(1,nc)]';

%valores iniciais
erro = zeros(N,1);
phi = zeros(dim,1);
yMqrEst = zeros(N,1);
yMqtEst(1:m)=Y(1:m);
p = 1000*eye(dim,dim);

for t = m : N
   phi = [-yMqrEst(t-(1:na)); P(t-d-(0:nb))'; er(t-(1:nc))]; % alterando matriz de observação
   
   K = (p*phi)/(phi'*p*phi+1); % calculando ganho

   theta = theta+(K*(Y(t)-phi'*theta)); % nova matriz de estimadores
 
   p = p - K*(phi'*p); % calculando matriz de covariância
   
   err = Y(t)-phi'*theta;
   yMqrEst(t) = phi'*theta + err; %obtendo valores de saída
end

pred = zeros(size(Yl));
pred(1:m) = Y(1:m);
for t = m : Nl
    pred(t) = sum([-pred(t-(1:na)).*theta(1:na)' Pl(t-d-(0:nb)).*theta(na+1:na+nb+1)' el(t-(1:nc))'.*theta(na+nb+2:end)']);
end
%plot results
figure;
subplot(3,1,1);
hold on
plot(Pl,Yl,'*');
plot(Pl,pred, '--r');
legend('Observado','Estimado')
title('Modelo ARMAX (Kalman Recursivo)')
xlabel('Horas em funcionamento')
ylabel('RMS')
hold off;
subplot(3,1,2);
deltaRms = abs(pred(2:end) - pred(1:end-1));
plot(Pl(2:end),deltaRms);
title('Delta RMS dos dados estimados.');
xlabel('Horas em funcionamento');
ylabel('Delta RMS');
subplot(3,1,3);
plot(Pl,sqrt((Yl-pred).^2));
title('Erro de estimação');
xlabel('Horas em funcionamento');
ylabel('Erro');
str(1) = {'RMSE: '};
str(2) = {sqrt(sum((Y-pred(1:N)).^2)/N)};
str(3) = {'MAE: '};
str(4) = {sum(abs(Y-pred(1:N)))/N};
str(5) = {'MAPE: '};
str(6) = {(sum(abs((Y-pred(1:N))./Y))/N)*100};
y1 = ylim;
y=(y1(2)-y1(1))/2;
text(x,y,str);